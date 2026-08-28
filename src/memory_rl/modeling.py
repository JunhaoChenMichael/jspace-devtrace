"""Model helpers for workspace-guided memory-admission training.

The RL policy deliberately reuses the paper's verbal yes/no interface.  We
restrict rollouts to the two valid actions instead of allowing arbitrary text;
this removes a formatting confound while still optimizing the language model's
own next-token probabilities.  The original (adapter-disabled) model is used as
the fixed KL reference.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as F


ADMISSION_PROMPT = (
    "{ctx}\n\nBased only on the passage above, is the concept \"{concept}\" "
    "one of the most important things to remember in order to answer possible "
    "future questions? Answer with a single word: yes or no."
)


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype: {name}") from exc


def render_admission_prompt(tokenizer, context: str, concept: str) -> str:
    """Render the exact prompt shared by SFT, RL, and evaluation."""
    body = ADMISSION_PROMPT.format(ctx=context, concept=concept)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return body + "\nAnswer:"


def yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Return deduplicated first-token variants for (No, Yes), in that order."""
    def ids(words: Iterable[str]) -> list[int]:
        out = []
        for word in words:
            encoded = tokenizer.encode(word, add_special_tokens=False)
            if encoded:
                out.append(int(encoded[0]))
        return sorted(set(out))

    no = ids(("no", " no", "No", " No", "NO", " NO"))
    yes = ids(("yes", " yes", "Yes", " Yes", "YES", " YES"))
    if not no or not yes:
        raise ValueError("tokenizer has no usable yes/no token variants")
    if set(no) & set(yes):
        raise ValueError("yes/no token sets unexpectedly overlap")
    return no, yes


def tokenize_prompts(tokenizer, prompts: list[str], device: str, max_length: int):
    old_padding = tokenizer.padding_side
    old_truncation = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    try:
        batch = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_padding
        tokenizer.truncation_side = old_truncation
    return {key: value.to(device) for key, value in batch.items()}


def binary_action_logits(
    model,
    tokenizer,
    prompts: list[str],
    token_ids: tuple[list[int], list[int]],
    device: str,
    max_length: int,
) -> torch.Tensor:
    """Return logits over canonical actions [No, Yes] for each prompt.

    Each action logit is logsumexp over common capitalization/leading-space
    variants, matching the paper's probability aggregation while giving RL an
    exact two-action policy.
    """
    batch = tokenize_prompts(tokenizer, prompts, device, max_length)
    vocab_logits = model(**batch).logits[:, -1].float()
    no_ids, yes_ids = token_ids
    no = torch.logsumexp(vocab_logits[:, no_ids], dim=-1)
    yes = torch.logsumexp(vocab_logits[:, yes_ids], dim=-1)
    return torch.stack((no, yes), dim=-1)


def selection_logits(action_logits: torch.Tensor) -> torch.Tensor:
    """Candidate score log P(Yes) - log P(No), invariant to joint normalization."""
    logp = F.log_softmax(action_logits, dim=-1)
    return logp[..., 1] - logp[..., 0]


def disable_dropout(model) -> dict[str, object]:
    """Disable stochastic dropout while retaining train-mode checkpointing.

    PPO/GRPO requires the rollout and the first policy update to represent the
    same distribution.  Calling ``model.eval()`` would also disable gradient
    checkpointing in common decoder implementations, so we instead zero all
    dropout modules and the standard config-level functional-dropout fields.
    """
    def audit_number(value: int | float) -> int | float | str:
        """Return a strict-JSON-safe representation for an audit scalar."""
        number = float(value)
        return number if math.isfinite(number) else repr(number)

    dropout_modules_found = []
    dropout_modules_modified = []
    for raw_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Dropout):
            continue
        name = raw_name or "<root>"
        before = module.p
        modified = before != 0.0
        if modified:
            module.p = 0.0
            dropout_modules_modified.append(name)
        dropout_modules_found.append(
            {
                "name": name,
                "type": type(module).__qualname__,
                "before": audit_number(before),
                "after": audit_number(module.p),
                "modified": modified,
            }
        )

    config_fields_found = []
    config_fields_modified = []
    config = getattr(model, "config", None)
    for name in (
        "attention_dropout",
        "hidden_dropout",
        "hidden_dropout_prob",
        "activation_dropout",
        "classifier_dropout",
        "resid_pdrop",
        "embd_pdrop",
        "attn_pdrop",
    ):
        value = getattr(config, name, None)
        if not isinstance(value, (int, float)):
            continue
        modified = value != 0
        if modified:
            setattr(config, name, 0.0)
            config_fields_modified.append(name)
        config_fields_found.append(
            {
                "name": name,
                "before": audit_number(value),
                "after": audit_number(getattr(config, name)),
                "modified": modified,
            }
        )

    # Scan the live model again instead of inferring success from attempted
    # assignments.  This makes the artifact evidence of the postcondition and
    # catches custom modules/config properties that silently reject mutation.
    remaining_nonzero = []
    for raw_name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
            remaining_nonzero.append(
                {
                    "kind": "module",
                    "name": raw_name or "<root>",
                    "value": audit_number(module.p),
                }
            )
    for name in (
        "attention_dropout",
        "hidden_dropout",
        "hidden_dropout_prob",
        "activation_dropout",
        "classifier_dropout",
        "resid_pdrop",
        "embd_pdrop",
        "attn_pdrop",
    ):
        value = getattr(config, name, None)
        if isinstance(value, (int, float)) and value != 0:
            remaining_nonzero.append(
                {
                    "kind": "config",
                    "name": name,
                    "value": audit_number(value),
                }
            )

    return {
        # Backward-compatible fields consumed by existing run artifacts/tests.
        "dropout_modules_zeroed": len(dropout_modules_modified),
        "config_fields_zeroed": config_fields_modified,
        # Detailed discovery/mutation records plus an independent post-scan.
        "dropout_modules_found": dropout_modules_found,
        "dropout_modules_modified": dropout_modules_modified,
        "config_fields_found": config_fields_found,
        "config_fields_modified": config_fields_modified,
        "remaining_nonzero": remaining_nonzero,
        "postcondition_satisfied": not remaining_nonzero,
    }


@contextmanager
def adapter_disabled(model):
    """Temporarily expose the frozen base checkpoint behind a PEFT policy."""
    was_training = model.training
    model.eval()
    disable = getattr(model, "disable_adapter", None)
    if disable is None:
        try:
            yield
        finally:
            model.train(was_training)
        return
    try:
        with disable():
            yield
    finally:
        model.train(was_training)


@dataclass
class PolicyBundle:
    model: object
    tokenizer: object
    action_token_ids: tuple[list[int], list[int]]
    device: str


def load_lora_policy(
    model_name: str,
    device: str,
    dtype: str = "bfloat16",
    lora_rank: int = 32,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.0,
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
) -> PolicyBundle:
    """Load a frozen causal LM plus trainable LoRA admission policy."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "memory RL requires PEFT; install the repository requirements first"
        ) from exc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch_dtype = resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.config.use_cache = False
    # Non-reentrant checkpointing works with a frozen backbone whose only
    # trainable tensors are LoRA weights; the reentrant variant may otherwise
    # drop the graph because token IDs/embedding outputs do not require grad.
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:  # compatibility with older Transformers releases
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha or 2 * lora_rank,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[x.strip() for x in target_modules.split(",") if x.strip()],
    )
    model = get_peft_model(model, config)
    model.train()
    return PolicyBundle(model, tok, yes_no_token_ids(tok), device)


def load_policy_for_eval(
    model_name: str,
    adapter_path: str | None,
    device: str,
    dtype: str = "bfloat16",
) -> PolicyBundle:
    """Load the original model or a saved LoRA adapter for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter_path or model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=resolve_dtype(dtype)
    ).to(device)
    if adapter_path:
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError("loading an RL adapter requires PEFT") from exc
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        configured_base = getattr(adapter_config, "base_model_name_or_path", None)
        if (
            configured_base
            and str(configured_base).rstrip("/") != str(model_name).rstrip("/")
        ):
            raise ValueError(
                f"adapter {adapter_path!r} was trained from {configured_base!r}, "
                f"not requested base {model_name!r}"
            )
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.eval()
    return PolicyBundle(model, tok, yes_no_token_ids(tok), device)
