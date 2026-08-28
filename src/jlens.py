"""
jlens.py — A workspace ("J-space") readout for open-weight causal LMs.

Goal: reimplement, on models we can actually inspect, the core move from
Anthropic's "A global workspace in Claude" (2026): for a given internal state,
read out the *silent words in the model's mind* — concepts that are active in
the residual stream but need not appear in the surface text.

Two readouts are provided:

  1. LogitLens (zero training): project an intermediate residual state through
     the model's own final-norm + unembedding. Classic logit-lens. Cheap, and
     already surfaces some silent intermediate concepts. Used for fast demos.

  2. FutureLens (the principled dual of the "Jacobian lens"): a per-layer affine
     probe h -> logits trained so that its softmax puts mass on tokens that
     appear LATER in the same sequence (a future window), not just the next
     token. This operationalizes the J-lens definition — "the internal pattern
     that raises the future probability of saying token v" — as its dual: the
     linear readout whose high-scoring tokens are the ones this state is pushing
     toward saying at some future point. This is what captures genuinely silent
     multi-step concepts. Training is one linear layer per layer, minutes on CPU.

Design choices for portability across many checkpoints (base/SFT/RL/VLM):
  - We read the residual stream via output_hidden_states=True, so the same code
    runs on GPT-2, Pythia/OLMo, Llama, Qwen, Gemma, ... without per-arch hooks.
  - The final norm + unembedding are located by common attribute names, so the
    lens is model-family-agnostic.

This is an approximation of Anthropic's exact (unpublished) method; the
experiments in this repo are designed to be robust to that — they compare the
SAME lens across checkpoints, so systematic lens error cancels in the contrast.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_final_norm(model) -> nn.Module:
    """Locate the model's final pre-unembedding norm across architectures."""
    m = model
    for path in [
        ("model", "norm"),          # Llama, Qwen2, Mistral, OLMo
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),    # GPT-2, GPT-NeoX-ish
        ("gpt_neox", "final_layer_norm"),
        ("model", "final_norm"),
    ]:
        obj = m
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.Module):
            return obj
    # Fallback: identity (classic logit lens without final norm).
    return nn.Identity()


@dataclass
class Readout:
    """Top-k silent words at one (layer, position)."""
    layer: int
    position: int
    tokens: list[str]
    probs: list[float]


class WorkspaceLens:
    def __init__(self, model_name: str, device: str | None = None,
                 dtype=torch.float32, adapter_path: str | None = None,
                 model_revision: str | None = None,
                 tokenizer_revision: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.sharded = device == "auto"
        self.device = "cuda:0" if self.sharded else (device or pick_device())
        # When only the model revision is supplied, use that same immutable
        # revision for the tokenizer.  A separate tokenizer revision remains
        # available for checkpoints whose tokenizer is intentionally pinned to
        # a different commit.
        effective_tokenizer_revision = tokenizer_revision or model_revision
        tokenizer_kwargs = {}
        if effective_tokenizer_revision is not None:
            tokenizer_kwargs["revision"] = effective_tokenizer_revision
        model_kwargs = {}
        if model_revision is not None:
            model_kwargs["revision"] = model_revision

        self.model_revision_requested = model_revision
        self.tokenizer_revision_requested = tokenizer_revision
        self.tokenizer_revision_effective = effective_tokenizer_revision
        self.tok = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if self.sharded:
            # multi-GPU sharding for models beyond one card (e.g. 13B+ on A5000s)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype,
                device_map="auto", **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, **model_kwargs
            ).to(self.device)
        if adapter_path:
            try:
                from peft import PeftConfig, PeftModel
            except ImportError as exc:
                raise RuntimeError("adapter-backed workspace measurement requires PEFT") from exc
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
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        get_base = getattr(self.model, "get_base_model", None)
        readout_model = get_base() if callable(get_base) else self.model
        self.final_norm = _find_final_norm(readout_model)
        if not self.sharded:
            self.final_norm = self.final_norm.to(self.device)
        self.unembed = readout_model.get_output_embeddings()  # nn.Linear(d, vocab)
        self.n_layers = readout_model.config.num_hidden_layers
        self.d_model = readout_model.config.hidden_size
        self.vocab = readout_model.config.vocab_size
        self.adapter_path = adapter_path
        self.model_revision_resolved = getattr(
            readout_model.config, "_commit_hash", None
        )
        tokenizer_init_kwargs = getattr(self.tok, "init_kwargs", {}) or {}
        self.tokenizer_revision_resolved = tokenizer_init_kwargs.get("_commit_hash")
        self.future_lens: dict[int, nn.Linear] = {}

    # ---- forward pass capturing the residual stream ----
    @torch.no_grad()
    def hidden_states(self, text: str) -> tuple[torch.Tensor, list[int]]:
        """Return hs [n_layers+1, seq, d] and the token ids."""
        enc = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model(**enc, output_hidden_states=True)
        layers = [h.to(self.device) for h in out.hidden_states]
        hs = torch.stack(layers, dim=0).squeeze(1)             # [L+1, seq, d]
        return hs, enc["input_ids"][0].tolist()

    def decode_ids(self, ids: list[int]) -> list[str]:
        return [self.tok.decode([i]).strip() for i in ids]

    # ---- readout 1: logit lens (zero training) ----
    @torch.no_grad()
    def logitlens(self, h: torch.Tensor) -> torch.Tensor:
        """Map a residual vector [.., d] to vocab logits via final-norm+unembed."""
        w = self.unembed.weight
        norm_dev = next(self.final_norm.parameters(), w).device
        h = h.to(device=norm_dev, dtype=w.dtype)
        return self.unembed(self.final_norm(h).to(w.device))

    # ---- readout 2: future-token tuned lens (Jacobian-lens dual) ----
    def fit_future_lens(self, corpus: list[str], layers: list[int] | None = None,
                        future_window: int = 12, steps: int = 400, lr: float = 5e-3,
                        max_positions: int = 20000, verbose: bool = True):
        """
        Train per-layer affine probes so softmax(W h + b) puts mass on tokens
        appearing in the next `future_window` positions. This is the readout dual
        of "the state that raises the FUTURE probability of saying token v".
        """
        layers = layers or list(range(1, self.n_layers + 1))
        # Collect (hidden, future-token-multiset) pairs.
        feats = {L: [] for L in layers}
        targets = []  # list of tensors of future token ids per position
        with torch.no_grad():
            for text in corpus:
                hs, ids = self.hidden_states(text)
                seq = len(ids)
                for p in range(seq - 1):
                    fut = ids[p + 1: min(seq, p + 1 + future_window)]
                    if not fut:
                        continue
                    targets.append(torch.tensor(fut))
                    for L in layers:
                        feats[L].append(hs[L, p].float().cpu())
                    if len(targets) >= max_positions:
                        break
                if len(targets) >= max_positions:
                    break
        n = len(targets)
        if verbose:
            print(f"[future-lens] {n} positions, {len(layers)} layers")
        for L in layers:
            X = torch.stack(feats[L]).to(self.device)             # [n, d]
            probe = nn.Linear(self.d_model, self.vocab).to(self.device)
            opt = torch.optim.Adam(probe.parameters(), lr=lr)
            for step in range(steps):
                idx = torch.randint(0, n, (min(256, n),))
                logits = probe(X[idx])
                # multi-hot future targets -> mean CE over the future window
                loss = 0.0
                lp = F.log_softmax(logits, dim=-1)
                for j, i in enumerate(idx.tolist()):
                    fut = targets[i].to(self.device)
                    loss = loss - lp[j, fut].mean()
                loss = loss / len(idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.future_lens[L] = probe.eval()
            if verbose:
                print(f"[future-lens] layer {L} trained, final loss {loss.item():.3f}")
        return self

    @torch.no_grad()
    def futurelens(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        probe = self.future_lens.get(layer)
        if probe is None:
            raise RuntimeError(f"future lens for layer {layer} not trained")
        return probe(h.to(self.device).float())

    # ---- high-level: silent words across layers at a position ----
    @torch.no_grad()
    def silent_words(self, text: str, position: int | None = None, topk: int = 8,
                     method: str = "logit", layers: list[int] | None = None,
                     exclude_surface: bool = True) -> list[Readout]:
        hs, ids = self.hidden_states(text)
        seq = len(ids)
        pos = position if position is not None else seq - 1
        layers = layers or list(range(1, self.n_layers + 1))
        surface = set(ids) if exclude_surface else set()
        out = []
        for L in layers:
            if method == "future":
                logits = self.futurelens(hs[L, pos], L)
            else:
                logits = self.logitlens(hs[L, pos])
            probs = F.softmax(logits.float(), dim=-1)
            top = torch.topk(probs, topk + len(surface))
            toks, ps = [], []
            for tid, p in zip(top.indices.tolist(), top.values.tolist()):
                if exclude_surface and tid in surface:
                    continue
                toks.append(self.tok.decode([tid]).strip())
                ps.append(round(p, 4))
                if len(toks) >= topk:
                    break
            out.append(Readout(layer=L, position=pos, tokens=toks, probs=ps))
        return out

    def token_direction(self, word: str) -> torch.Tensor:
        """
        Unit UNEMBEDDING direction for a token. NOTE: this is the *output* direction
        (what makes the model SAY the token), not the input-side workspace direction
        that downstream computations READ. Steering with it is a deliberately-weak
        baseline that fails to flip answers — see concept_vector() for the correct,
        activation-derived (Jacobian-spirit) input-side direction.
        """
        ids = self.tok.encode(word, add_special_tokens=False)
        if not ids:
            raise ValueError(f"no token for {word!r}")
        vec = self.unembed.weight[ids[0]].detach().float()
        return vec / (vec.norm() + 1e-8)

    @torch.no_grad()
    def concept_vector(self, prompts_pos: list[str], prompts_neg: list[str],
                       layer: int, last_only: bool = True) -> torch.Tensor:
        """
        Input-side concept direction via difference-of-means of the residual
        stream (the RepE / function-vector construction). This is the direction
        the network actually READS, so steering along it flips downstream answers
        — unlike the unembedding row. Returns a raw (unnormalized) delta at `layer`.
        """
        def mean_act(prompts):
            accs = []
            for t in prompts:
                hs, ids = self.hidden_states(t)
                if last_only:
                    accs.append(hs[layer, -1].float())
                else:
                    accs.append(hs[layer].float().mean(0))
            return torch.stack(accs).mean(0)
        return mean_act(prompts_pos) - mean_act(prompts_neg)
