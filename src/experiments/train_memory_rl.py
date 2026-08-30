"""Train workspace-guided memory-admission policies.

Implements the README's minimum viable experiment with one entry point:

  sft-w     hard top-k workspace distillation (the matched SFT baseline)
  rl-w      constrained Yes/No GRPO with fixed W_ref reward (Stage A)
  rl-qa     exact-budget set GRPO with frozen-recall QA reward (Stage B)
  rl-hybrid exact-budget set GRPO with W_ref + QA reward (Stage C)

The reference checkpoint is the adapter-disabled base model.  Workspace rewards
are loaded from immutable result JSONs and fingerprinted in split_manifest.json;
the trainable model can therefore not hack its own latent reward.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import get_cosine_schedule_with_warmup

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from memory_rl.data import (  # noqa: E402
    TrainingSpec,
    build_training_bundle,
    default_training_specs,
    write_split_manifest,
)
from memory_rl.modeling import (  # noqa: E402
    adapter_disabled,
    binary_action_logits,
    disable_dropout,
    load_lora_policy,
    render_admission_prompt,
    selection_logits,
)
from memory_rl.objectives import (  # noqa: E402
    categorical_kl,
    exact_set_entropy,
    exact_set_kl,
    grpo_clipped_loss,
    hybrid_reward,
    normalize_advantages,
    sample_gumbel_topk,
    set_logprob,
    workspace_action_reward,
    workspace_set_reward,
)
from memory_rl.recall import FrozenRecall  # noqa: E402
from memory_rl.reporter_correlations import (  # noqa: E402
    summarize_reporter_correlations,
    within_episode_utility_auc,
)
from memory_rl.training_diagnostics import (  # noqa: E402
    summarize_selector_training,
    summarize_selector_window,
)


MODES = ("sft-w", "rl-w", "rl-qa", "rl-hybrid")

# Released workspace files predate artifact metadata, so bind their historical
# tags to the checkpoint IDs recorded by the measurement scripts/logs.
KNOWN_TEACHER_MODELS = {
    "7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "7B-base": "Qwen/Qwen2.5-7B",
    "qwen3-1.7B": "Qwen/Qwen3-1.7B",
    # Capable-scale replication target for the RL-QA track; its W_ref rows are
    # measured by this project (data/results/results_v*_qwen3-8B.json), not
    # inherited from the Qwen2.5 campaign.
    "qwen3-8B": "Qwen/Qwen3-8B",
    # 32B scale point for the seed-0 scaling gate; dense, same family. Its
    # W_ref rows are measured by this project, never inherited from 8B.
    "qwen3-32B": "Qwen/Qwen3-32B",
    "olmo1b-rlvr": "allenai/OLMo-2-0425-1B-Instruct",
    "olmo7b-Instruct": "allenai/OLMo-2-1124-7B-Instruct",
}


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, default=json_default, allow_nan=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def software_versions() -> dict[str, str | None]:
    versions = {"python": sys.version.split()[0], "cuda": torch.version.cuda}
    for package in ("torch", "transformers", "peft", "numpy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def parse_spec(text: str) -> TrainingSpec:
    """Parse NAME=RESULTS::BATTERY without making ':' in paths ambiguous."""
    if "=" not in text or "::" not in text:
        raise argparse.ArgumentTypeError(
            "train spec must be NAME=RESULTS_JSON::BATTERY_JSON"
        )
    name, paths = text.split("=", 1)
    results, battery = paths.split("::", 1)
    return TrainingSpec(source=name, results_path=results, battery_path=battery)


def candidate_prompts(tokenizer, candidates) -> list[str]:
    return [
        render_admission_prompt(tokenizer, candidate.context, candidate.concept)
        for candidate in candidates
    ]


def selector_candidate_prompts(tokenizer, episode) -> list[str]:
    """Render selector prompts and fail closed if the future probe leaks."""

    prompts = [
        render_admission_prompt(tokenizer, episode.context, candidate.concept)
        for candidate in episode.candidates
    ]
    leaked = [
        index
        for index, prompt in enumerate(prompts)
        if episode.probe_question in prompt
    ]
    if leaked:
        raise RuntimeError(
            f"probe leaked into admission prompt for {episode.uid}, candidates {leaked}"
        )
    return prompts


def grouped_candidates(episodes):
    return [candidate for episode in episodes for candidate in episode.candidates]


def within_episode_auc(candidates, scores: list[float]) -> float:
    by_episode = defaultdict(list)
    for candidate, score in zip(candidates, scores):
        by_episode[candidate.episode_uid].append((candidate.label == "load_bearing", score))
    values = []
    for rows in by_episode.values():
        positives = [s for y, s in rows if y]
        negatives = [s for y, s in rows if not y]
        if not positives or not negatives:
            continue
        wins = sum((a > b) + 0.5 * (a == b) for a in positives for b in negatives)
        values.append(wins / (len(positives) * len(negatives)))
    return float(np.mean(values)) if values else float("nan")


def containment(episodes, score_by_uid: dict[str, float], budget: int) -> float:
    hits = []
    for episode in episodes:
        ordered = sorted(
            episode.candidates,
            key=lambda candidate: score_by_uid[candidate.uid],
            reverse=True,
        )[:budget]
        hits.append(any(x.label == "load_bearing" for x in ordered))
    return float(np.mean(hits)) if hits else float("nan")


@torch.no_grad()
def score_candidates(bundle, candidates, max_length: int, batch_size: int = 8):
    was_training = bundle.model.training
    bundle.model.eval()
    values = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        logits = binary_action_logits(
            bundle.model,
            bundle.tokenizer,
            candidate_prompts(bundle.tokenizer, chunk),
            bundle.action_token_ids,
            bundle.device,
            max_length,
        )
        values.extend(F.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    bundle.model.train(was_training)
    return values


def evaluate_reporter(bundle, episodes, budget: int, max_length: int) -> dict:
    candidates = grouped_candidates(episodes)
    scores = score_candidates(bundle, candidates, max_length)
    labels = [candidate.label == "load_bearing" for candidate in candidates]
    pooled = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")
    score_map = {candidate.uid: score for candidate, score in zip(candidates, scores)}
    return {
        "auc": float(pooled),
        "within_episode_auc": within_episode_auc(candidates, scores),
        "containment": containment(episodes, score_map, budget),
        "yes_rate": float(np.mean(np.asarray(scores) >= 0.5)),
    }


@torch.no_grad()
def evaluate_selector(
    bundle,
    episodes,
    recall,
    budget: int,
    max_length: int,
    max_episodes: int = 0,
    reporter_bootstrap_samples: int = 4000,
    reporter_bootstrap_seed: int = 0,
) -> tuple[dict, list[dict]]:
    subset = episodes[:max_episodes] if max_episodes else episodes
    correct, contains, workspace = [], [], []
    reporter_rows: list[dict] = []
    was_training = bundle.model.training
    bundle.model.eval()
    for episode in subset:
        logits = binary_action_logits(
            bundle.model,
            bundle.tokenizer,
            selector_candidate_prompts(bundle.tokenizer, episode),
            bundle.action_token_ids,
            bundle.device,
            max_length,
        )
        yes_probabilities = F.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        reporter_rows.extend(
            {
                "episode_id": episode.uid,
                "candidate_id": candidate.uid,
                "source": episode.source,
                "v_rl": float(v_rl),
                "w_ref": float(candidate.w_ref),
                "y_utility": int(candidate.label == "load_bearing"),
            }
            for candidate, v_rl in zip(episode.candidates, yes_probabilities)
        )
        chosen = torch.topk(selection_logits(logits), budget).indices.cpu().tolist()
        record = recall.evaluate_sets(episode, [chosen])[0]
        correct.append(record["correct"])
        contains.append(any(episode.candidates[i].label == "load_bearing" for i in chosen))
        workspace.append(float(np.mean([episode.candidates[i].w_percentile for i in chosen])))
    bundle.model.train(was_training)
    correlations = summarize_reporter_correlations(
        reporter_rows,
        bootstrap_samples=reporter_bootstrap_samples,
        bootstrap_seed=reporter_bootstrap_seed,
    )
    return {
        "qa_accuracy": float(np.mean(correct)) if correct else float("nan"),
        "containment": float(np.mean(contains)) if contains else float("nan"),
        "workspace_set_reward": float(np.mean(workspace)) if workspace else float("nan"),
        "verbal_auc": correlations["utility_auc"],
        "verbal_within_episode_auc": within_episode_utility_auc(reporter_rows),
        "verbal_yes_rate": correlations["yes_rate"],
        "reporter_correlations": correlations,
        "episodes": len(subset),
    }, reporter_rows


class ReferenceCache:
    def __init__(self, bundle, max_length: int):
        self.bundle = bundle
        self.max_length = max_length
        self.values: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def action_logits(self, candidates) -> torch.Tensor:
        missing = [candidate for candidate in candidates if candidate.uid not in self.values]
        if missing:
            with adapter_disabled(self.bundle.model):
                logits = binary_action_logits(
                    self.bundle.model,
                    self.bundle.tokenizer,
                    candidate_prompts(self.bundle.tokenizer, missing),
                    self.bundle.action_token_ids,
                    self.bundle.device,
                    self.max_length,
                )
            for candidate, row in zip(missing, logits):
                self.values[candidate.uid] = row.detach().cpu()
        return torch.stack([self.values[c.uid] for c in candidates]).to(self.bundle.device)


def save_adapter(bundle, directory: Path, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    bundle.model.save_pretrained(directory)
    bundle.tokenizer.save_pretrained(directory)
    write_json(directory / "training_state.json", metadata)


def diagnose(history: list[dict], mode: str) -> list[str]:
    if not history:
        return ["no training records"]
    warnings = []
    recent = history[-min(100, len(history)):]
    def applicable_values(key: str) -> np.ndarray:
        values = [row.get(key) for row in recent]
        return np.asarray(
            [float(value) for value in values if value is not None], dtype=float
        )

    rewards = applicable_values("reward_mean")
    variances = applicable_values("reward_std")
    kls = applicable_values("kl")
    if len(rewards) >= 100 and rewards[-50:].mean() <= rewards[-100:-50].mean():
        warnings.append("training reward did not increase between steps -100:-50 and -50:end")
    if len(variances) and float(np.mean(variances < 0.1)) > 0.5:
        if mode == "rl-qa":
            warnings.append("reward std < 0.1 in most groups; inspect recall ceiling or use hybrid reward")
        else:
            warnings.append("reward std < 0.1 in most groups; inspect sampling temperature")
    if len(kls) and len(rewards) and np.mean(kls) > max(0.1, abs(np.mean(rewards))):
        warnings.append("KL magnitude is comparable to or larger than reward; consider lower beta")
    if mode == "rl-w":
        yes_rates = np.asarray([x.get("yes_rate", 0.5) for x in recent])
        if np.mean((yes_rates < 0.05) | (yes_rates > 0.95)) > 0.5:
            warnings.append("reporter collapsed toward always-yes or always-no")
    if mode in ("rl-qa", "rl-hybrid"):
        unique_sets = applicable_values("number_unique_selected_sets")
        if len(unique_sets) and float(np.mean(unique_sets <= 1.0)) > 0.5:
            warnings.append("selection diversity collapsed to one exact-k set in most groups")
        if len(unique_sets) >= 50:
            width = min(50, len(unique_sets) // 2)
            if unique_sets[-width:].mean() < unique_sets[:width].mean() - 0.5:
                warnings.append("selection diversity declined materially during training")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--teacher-tag", default="7B-Instruct")
    parser.add_argument(
        "--workspace-teacher-model",
        default=None,
        help="checkpoint that produced W_ref; inferred for known teacher tags",
    )
    parser.add_argument(
        "--allow-teacher-mismatch",
        action="store_true",
        help="explicit smoke/ablation override; never use for the primary comparison",
    )
    parser.add_argument("--train-spec", action="append", type=parse_spec,
                        help="NAME=RESULTS_JSON::BATTERY_JSON; repeatable")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="fixed episode-split seed; keep constant across optimization seeds",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--workspace-top-k", type=int, default=2)
    parser.add_argument(
        "--workspace-objective",
        choices=("rank-continuous", "top-k"),
        default="rank-continuous",
        help="shared SFT/RL-W teacher objective; top-k reproduces legacy SFT-W",
    )
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--grpo-epochs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--lambda-qa", type=float, default=1.0)
    parser.add_argument("--lambda-w", type=float, default=0.5)
    parser.add_argument("--workspace-set-reward", choices=("mean", "contrastive"),
                        default="mean")
    parser.add_argument("--advantage-normalization",
                        choices=("auto", "center", "zscore"), default="auto")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="SFT candidate batch size")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--answer-tokens", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--diagnostic-every",
        type=int,
        default=25,
        help="selector rolling-diagnostic interval in optimizer steps",
    )
    parser.add_argument(
        "--reporter-correlation-bootstrap-samples",
        type=int,
        default=4000,
        help="episode-cluster bootstrap draws for ID reporter correlations",
    )
    parser.add_argument(
        "--reporter-correlation-bootstrap-seed",
        type=int,
        default=0,
        help="fixed bootstrap seed for ID reporter correlations",
    )
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--val-eval-episodes", type=int, default=0)
    parser.add_argument("--limit-train-episodes", type=int, default=0,
                        help="smoke tests only; applied after the split")
    parser.add_argument("--limit-validation-episodes", type=int, default=0,
                        help="smoke tests only; applied after the split")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if args.limit_train_episodes < 0 or args.limit_validation_episodes < 0:
        parser.error("smoke-test episode limits must be >= 0")
    if args.mode != "sft-w" and args.group_size < 2:
        parser.error("--group-size must be >= 2 for group-relative advantages")
    if args.grpo_epochs < 1 or args.batch_size < 1:
        parser.error("--grpo-epochs and --batch-size must be >= 1")
    if args.workspace_top_k < 1 or args.lora_rank < 1:
        parser.error("--workspace-top-k and --lora-rank must be >= 1")
    if args.learning_rate <= 0 or args.max_length < 1 or args.answer_tokens < 1:
        parser.error("learning rate, max length, and answer tokens must be positive")
    if (
        args.eval_every < 0
        or args.save_every < 0
        or args.diagnostic_every < 1
        or args.early_stop_patience < 1
    ):
        parser.error("eval/save intervals must be non-negative and patience >= 1")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.reporter_correlation_bootstrap_samples < 0:
        parser.error("--reporter-correlation-bootstrap-samples must be >= 0")
    if args.beta < 0 or args.lambda_qa < 0 or args.lambda_w < 0:
        parser.error("reward and KL coefficients must be non-negative")
    if args.mode == "rl-qa" and args.lambda_qa <= 0:
        parser.error("rl-qa requires --lambda-qa > 0")
    if args.mode == "rl-hybrid" and args.lambda_qa + args.lambda_w <= 0:
        parser.error("rl-hybrid requires at least one non-zero reward coefficient")
    if args.mode in ("rl-qa", "rl-hybrid") and args.budget < 1:
        parser.error("selector modes require --budget >= 1")
    if args.mode == "rl-w" and args.advantage_normalization == "zscore":
        print("WARNING: zscore removes |2r-1| for binary groups; center is recommended", flush=True)

    teacher_model = args.workspace_teacher_model or KNOWN_TEACHER_MODELS.get(
        args.teacher_tag
    )
    if teacher_model is None:
        parser.error(
            "unknown --teacher-tag: provide --workspace-teacher-model so W_ref "
            "provenance can be recorded"
        )
    teacher_matches_policy = args.model.rstrip("/") == teacher_model.rstrip("/")
    if not teacher_matches_policy and not args.allow_teacher_mismatch:
        parser.error(
            f"W_ref teacher {teacher_model!r} does not match policy/reference "
            f"checkpoint {args.model!r}; pass matching artifacts, or use "
            "--allow-teacher-mismatch only for a smoke test/declared ablation"
        )

    if args.advantage_normalization == "auto":
        effective_advantage_mode = (
            "center"
            if args.mode == "rl-w" and args.workspace_objective == "rank-continuous"
            else "zscore"
        )
    else:
        effective_advantage_mode = args.advantage_normalization

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    specs = tuple(args.train_spec or default_training_specs(args.teacher_tag))
    data_bundle = build_training_bundle(
        specs,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
        top_k=args.workspace_top_k,
    )
    train_episodes = data_bundle.train_episodes
    val_episodes = data_bundle.validation_episodes
    write_split_manifest(out_dir / "split_manifest.json", data_bundle.split_manifest)
    if args.limit_train_episodes:
        train_episodes = train_episodes[:args.limit_train_episodes]
    if args.limit_validation_episodes:
        val_episodes = val_episodes[:args.limit_validation_episodes]
    if not train_episodes:
        raise ValueError("training split is empty")
    if args.mode in ("rl-qa", "rl-hybrid"):
        too_small = [
            episode.uid
            for episode in (*train_episodes, *val_episodes)
            if len(episode.candidates) <= args.budget
        ]
        if too_small:
            raise ValueError(
                f"selector training requires budget < candidate count; failed at "
                f"{too_small[0]}"
            )
    run_config = vars(args).copy()
    run_config["train_spec"] = [
        {
            "source": spec.canonical_source,
            "battery_path": str(spec.battery_path),
            "results_path": str(spec.results_path),
        }
        for spec in specs
    ]
    run_config["train_episode_count"] = len(train_episodes)
    run_config["validation_episode_count"] = len(val_episodes)
    run_config["fixed_workspace_teacher"] = True
    run_config["workspace_teacher_model"] = teacher_model
    run_config["teacher_matches_policy_reference"] = teacher_matches_policy
    run_config["teacher_mismatch_override"] = bool(args.allow_teacher_mismatch)
    run_config["effective_advantage_normalization"] = effective_advantage_mode
    run_config["constrained_yes_no_rollouts"] = True
    run_config["probe_visible_to_policy"] = False
    run_config["frozen_recall_adapter_disabled"] = args.mode in ("rl-qa", "rl-hybrid")
    run_config["workspace_reward_used_for_optimization"] = (
        args.mode == "rl-hybrid" and args.lambda_w > 0
    )
    run_config["reporter_correlation_diagnostics"] = args.mode in (
        "rl-qa", "rl-hybrid"
    )
    run_config["reporter_correlation_scope"] = "fixed ID validation candidates"
    run_config["reporter_correlations_used_for_selection"] = False
    run_config["reporter_correlation_artifact"] = (
        "reporter_correlations.jsonl"
        if run_config["reporter_correlation_diagnostics"]
        else None
    )
    run_config["software_versions"] = software_versions()
    write_json(out_dir / "run_config.json", run_config)
    print(f"mode={args.mode} train={len(train_episodes)} val={len(val_episodes)} "
          f"seed={args.seed} split_seed={args.split_seed} out={out_dir}", flush=True)
    if args.dry_run:
        print("dry run: split and fingerprints validated; model not loaded", flush=True)
        return

    bundle = load_lora_policy(
        args.model,
        device=args.device,
        dtype=args.dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha or None,
        lora_dropout=args.lora_dropout,
    )
    dropout_audit = None
    if args.mode != "sft-w":
        dropout_audit = disable_dropout(bundle.model)
        write_json(out_dir / "dropout_audit.json", dropout_audit)
    get_base = getattr(bundle.model, "get_base_model", None)
    resolved_model = get_base() if callable(get_base) else bundle.model
    run_config["resolved_model_name_or_path"] = getattr(
        resolved_model.config, "_name_or_path", args.model
    )
    run_config["resolved_model_commit"] = getattr(
        resolved_model.config, "_commit_hash", None
    )
    run_config["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in bundle.model.parameters()
        if parameter.requires_grad
    )
    run_config["total_parameter_count"] = sum(
        parameter.numel() for parameter in bundle.model.parameters()
    )
    write_json(out_dir / "run_config.json", run_config)
    bundle.model.print_trainable_parameters()
    reference = ReferenceCache(bundle, args.max_length)
    recall = None
    if args.mode in ("rl-qa", "rl-hybrid"):
        recall = FrozenRecall(bundle.model, bundle.tokenizer, bundle.device,
                              max_new_tokens=args.answer_tokens)

    parameters = [p for p in bundle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    updates_per_rollout = 1 if args.mode == "sft-w" else args.grpo_epochs
    total_updates = max(1, args.max_steps * updates_per_rollout)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(20, max(1, total_updates // 10)),
        num_training_steps=total_updates,
    )
    rng = random.Random(args.seed)
    torch_generator = torch.Generator(device=torch.device(args.device))
    torch_generator.manual_seed(args.seed)
    train_candidates = grouped_candidates(train_episodes)
    history: list[dict] = []
    rollout_file = out_dir / "rollouts.jsonl"
    metric_file = out_dir / "metrics.jsonl"
    reporter_correlation_file = out_dir / "reporter_correlations.jsonl"
    best_metric = -math.inf
    non_improving = 0
    best_checkpoint = None
    best_validation = None

    def validate(step: int, event: str = "validation") -> dict:
        if args.mode in ("sft-w", "rl-w"):
            metrics = evaluate_reporter(bundle, val_episodes, args.budget, args.max_length)
            score = metrics["auc"]
        else:
            metrics, reporter_rows = evaluate_selector(
                bundle, val_episodes, recall, args.budget, args.max_length,
                args.val_eval_episodes,
                args.reporter_correlation_bootstrap_samples,
                args.reporter_correlation_bootstrap_seed,
            )
            reward_weight = args.lambda_qa + args.lambda_w
            metrics["combined_reward"] = (
                args.lambda_qa * metrics["qa_accuracy"]
                + args.lambda_w * metrics["workspace_set_reward"]
            ) / reward_weight
            append_jsonl(
                reporter_correlation_file,
                {
                    "schema_version": 1,
                    "scope": "fixed_id_validation",
                    "event": event,
                    "step": step,
                    "summary": metrics["reporter_correlations"],
                    "rows": reporter_rows,
                },
            )
            score = metrics["qa_accuracy"]
        metrics.update({"event": event, "step": step, "selection_metric": score})
        append_jsonl(metric_file, metrics)
        print(f"[{event} step={step}] {json.dumps(metrics, default=json_default)}", flush=True)
        return metrics

    baseline = validate(0, event="baseline")
    last_validation = baseline

    started = time.time()
    for step in range(1, args.max_steps + 1):
        bundle.model.train()
        if args.mode == "sft-w":
            batch = rng.sample(train_candidates, min(args.batch_size, len(train_candidates)))
            logits = binary_action_logits(
                bundle.model, bundle.tokenizer,
                candidate_prompts(bundle.tokenizer, batch), bundle.action_token_ids,
                bundle.device, args.max_length,
            )
            if args.workspace_objective == "top-k":
                targets = torch.tensor(
                    [int(candidate.workspace_target) for candidate in batch],
                    device=bundle.device,
                )
                loss = F.cross_entropy(logits, targets)
            else:
                margins = torch.tensor(
                    [2.0 * candidate.w_percentile - 1.0 for candidate in batch],
                    device=bundle.device,
                )
                targets = (margins > 0).long()
                per_example = F.cross_entropy(logits, targets, reduction="none")
                weights = margins.abs()
                loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step(); scheduler.step()
            record = {
                "event": "train", "step": step, "loss": float(loss.detach()),
                "reward_mean": None, "reward_std": None,
                "kl": 0.0, "yes_rate": float(targets.float().mean()),
                "grad_norm": float(grad_norm),
            }
        elif args.mode == "rl-w":
            candidate = rng.choice(train_candidates)
            prompt = candidate_prompts(bundle.tokenizer, [candidate])
            with torch.no_grad():
                old_logits = binary_action_logits(
                    bundle.model, bundle.tokenizer, prompt, bundle.action_token_ids,
                    bundle.device, args.max_length,
                )[0] / args.temperature
                ref_logits = reference.action_logits([candidate])[0] / args.temperature
                probs = F.softmax(old_logits, dim=-1)
                actions = torch.multinomial(
                    probs, args.group_size, replacement=True, generator=torch_generator
                )
                if args.workspace_objective == "top-k":
                    target = int(candidate.workspace_target)
                    rewards = torch.where(
                        actions == target,
                        torch.ones_like(actions, dtype=torch.float32),
                        -torch.ones_like(actions, dtype=torch.float32),
                    )
                else:
                    rewards = workspace_action_reward(actions, candidate.w_percentile)
                advantages = normalize_advantages(
                    rewards, mode=effective_advantage_mode
                )
                old_logp = F.log_softmax(old_logits, dim=-1)[actions]
            losses, kls, grad_norms = [], [], []
            for _ in range(args.grpo_epochs):
                current = binary_action_logits(
                    bundle.model, bundle.tokenizer, prompt, bundle.action_token_ids,
                    bundle.device, args.max_length,
                )[0] / args.temperature
                new_logp = F.log_softmax(current, dim=-1)[actions]
                kl = categorical_kl(current, ref_logits)
                loss = grpo_clipped_loss(
                    new_logp,
                    old_logp,
                    advantages,
                    clip_epsilon=args.clip_epsilon,
                    kl=kl,
                    beta=args.beta,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step(); scheduler.step()
                losses.append(float(loss.detach())); kls.append(float(kl.detach()))
                grad_norms.append(float(grad_norm))
            record = {
                "event": "train", "step": step, "candidate_id": candidate.uid,
                "loss": float(np.mean(losses)), "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std(unbiased=False)),
                "kl": float(np.mean(kls)),
                "grad_norm": float(np.mean(grad_norms)),
                "yes_rate": float((actions == 1).float().mean()),
                "workspace_percentile": candidate.w_percentile,
            }
            append_jsonl(rollout_file, {
                "step": step, "episode_id": candidate.episode_uid,
                "candidate_ids": [candidate.uid],
                "actions": ["Yes" if int(x) else "No" for x in actions.cpu()],
                "rewards": rewards.cpu().tolist(), "kl": record["kl"],
            })
        else:
            episode = rng.choice(train_episodes)
            if args.budget > len(episode.candidates):
                raise ValueError(f"budget {args.budget} exceeds candidates in {episode.uid}")
            prompts = selector_candidate_prompts(bundle.tokenizer, episode)
            with torch.no_grad():
                old_actions = binary_action_logits(
                    bundle.model, bundle.tokenizer, prompts, bundle.action_token_ids,
                    bundle.device, args.max_length,
                )
                old_scores = selection_logits(old_actions)
                ref_actions = reference.action_logits(episode.candidates)
                ref_scores = selection_logits(ref_actions)
                selected = sample_gumbel_topk(
                    old_scores,
                    args.budget,
                    num_samples=args.group_size,
                    temperature=args.temperature,
                    generator=torch_generator,
                )
                selected_rows = selected.cpu().tolist()
                canonical_sets = [tuple(indices) for indices in selected_rows]
                unique_set_count = len(set(canonical_sets))
                yes_probabilities = F.softmax(old_actions, dim=-1)[:, 1]
                policy_set_entropy = exact_set_entropy(
                    old_scores,
                    args.budget,
                    temperature=args.temperature,
                )
                normalized_policy_set_entropy = exact_set_entropy(
                    old_scores,
                    args.budget,
                    temperature=args.temperature,
                    normalize=True,
                )
                old_logp = torch.stack(
                    [
                        set_logprob(old_scores, subset, temperature=args.temperature)
                        for subset in selected
                    ]
                )
                recalls = recall.evaluate_sets(episode, selected_rows)
                qa_rewards = torch.tensor(
                    [float(row["correct"]) for row in recalls], device=bundle.device
                )
                percentiles = torch.tensor(
                    [x.w_percentile for x in episode.candidates], device=bundle.device
                )
                w_rewards = workspace_set_reward(
                    percentiles, selected,
                    contrastive=args.workspace_set_reward == "contrastive",
                )
                if args.mode == "rl-qa":
                    rewards = qa_rewards
                else:
                    rewards = hybrid_reward(
                        qa_rewards,
                        w_rewards,
                        lambda_qa=args.lambda_qa,
                        lambda_workspace=args.lambda_w,
                        normalize_weights=True,
                    )
                advantages = normalize_advantages(
                    rewards, mode=effective_advantage_mode
                )
                refs = recall.references(episode, args.budget)
                contains_by_group = [
                    any(
                        episode.candidates[index].label == "load_bearing"
                        for index in indices
                    )
                    for indices in selected_rows
                ]
            losses, kls, grad_norms = [], [], []
            for _ in range(args.grpo_epochs):
                current_actions = binary_action_logits(
                    bundle.model, bundle.tokenizer, prompts, bundle.action_token_ids,
                    bundle.device, args.max_length,
                )
                current_scores = selection_logits(current_actions)
                new_logp = torch.stack(
                    [
                        set_logprob(
                            current_scores, subset, temperature=args.temperature
                        )
                        for subset in selected
                    ]
                )
                kl = exact_set_kl(
                    current_scores,
                    ref_scores,
                    args.budget,
                    temperature=args.temperature,
                )
                loss = grpo_clipped_loss(
                    new_logp,
                    old_logp,
                    advantages,
                    clip_epsilon=args.clip_epsilon,
                    kl=kl,
                    beta=args.beta,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step(); scheduler.step()
                losses.append(float(loss.detach())); kls.append(float(kl.detach()))
                grad_norms.append(float(grad_norm))
            record = {
                "event": "train", "step": step, "episode_id": episode.uid,
                "loss": float(np.mean(losses)), "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std(unbiased=False)),
                "qa_reward": float(qa_rewards.mean()),
                "workspace_reward": float(w_rewards.mean()),
                "kl": float(np.mean(kls)), "yes_rate": None,
                "grad_norm": float(np.mean(grad_norms)),
                "candidate_count": len(episode.candidates),
                "number_unique_selected_sets": unique_set_count,
                "fraction_unique_selected_sets": unique_set_count / args.group_size,
                "mixed_QA_reward_group": len(set(qa_rewards.cpu().tolist())) > 1,
                "containment_rate": float(np.mean(contains_by_group)),
                "mixed_containment_group": len(set(contains_by_group)) > 1,
                "policy_set_entropy": float(policy_set_entropy),
                "normalized_policy_set_entropy": float(normalized_policy_set_entropy),
                "yes_probabilities": yes_probabilities.cpu().tolist(),
            }
            for group_index, (indices, result) in enumerate(zip(selected_rows, recalls)):
                indices = sorted(set(indices))
                append_jsonl(rollout_file, {
                    "step": step, "group_index": group_index,
                    "episode_id": episode.uid,
                    "candidate_ids": [x.uid for x in episode.candidates],
                    "selection_probability": float(old_logp[group_index].exp()),
                    "selection_log_probability": float(old_logp[group_index]),
                    "selected_set": [episode.candidates[i].uid for i in indices],
                    "selected_concepts": result["selected_concepts"],
                    "contains_load_bearing": contains_by_group[group_index],
                    "answer": result["answer"], "QA_correct": result["correct"],
                    "oracle_QA_correct": refs["oracle_correct"],
                    "full_context_correct": refs["full_context_correct"],
                    "workspace_scores": [x.w_ref for x in episode.candidates],
                    "verbal_scores": [x.v_ref for x in episode.candidates],
                    "QA_reward": float(qa_rewards[group_index]),
                    "workspace_reward": float(w_rewards[group_index]),
                    "reward": float(rewards[group_index]), "KL": record["kl"],
                    "failure_type": (
                        "selection" if not any(
                            episode.candidates[i].label == "load_bearing" for i in indices
                        ) else ("recall_or_composition" if not result["correct"] else "none")
                    ),
                })

        history.append(record)
        append_jsonl(metric_file, record)
        if step == 1 or step % 10 == 0:
            reward_text = (
                "n/a" if record["reward_mean"] is None
                else f"{record['reward_mean']:.3f}"
            )
            reward_std_text = (
                "n/a" if record["reward_std"] is None
                else f"{record['reward_std']:.3f}"
            )
            print(f"[step {step}/{args.max_steps}] loss={record['loss']:.4f} "
                  f"reward={reward_text} std={reward_std_text} "
                  f"kl={record['kl']:.4f} grad={record['grad_norm']:.4f}", flush=True)

        if args.mode in ("rl-qa", "rl-hybrid") and (
            step == 1 or step % args.diagnostic_every == 0 or step == args.max_steps
        ):
            selector_rows = [
                row
                for row in history[-min(args.diagnostic_every, len(history)) :]
                if row.get("number_unique_selected_sets") is not None
            ]
            diagnostic = summarize_selector_window(selector_rows)
            print(
                f"[selector-diagnostic step={step}] "
                f"{json.dumps(diagnostic, allow_nan=False)}",
                flush=True,
            )

        should_eval = args.eval_every > 0 and step % args.eval_every == 0
        if should_eval or step == args.max_steps:
            metrics = validate(step)
            last_validation = metrics
            score = metrics["selection_metric"]
            if score > best_metric:
                best_metric, non_improving = score, 0
                best_validation = metrics
                checkpoint = out_dir / f"best-step-{step}"
                save_adapter(bundle, checkpoint, {"step": step, "metric": score})
                best_checkpoint = str(checkpoint)
                write_json(out_dir / "best_checkpoint.json",
                           {"path": best_checkpoint, "step": step, "metric": score})
            else:
                non_improving += 1
            if non_improving >= args.early_stop_patience:
                print(f"early stopping after {non_improving} non-improving validations", flush=True)
                break
        if args.save_every > 0 and step % args.save_every == 0 and step != args.max_steps:
            save_adapter(bundle, out_dir / f"checkpoint-{step}", {"step": step})

    final_dir = out_dir / "final_adapter"
    save_adapter(bundle, final_dir, {"step": history[-1]["step"],
                                    "elapsed_seconds": time.time() - started})
    selector_diagnostics = (
        summarize_selector_training(history, window_size=args.diagnostic_every)
        if args.mode in ("rl-qa", "rl-hybrid")
        else None
    )
    summary = {
        "mode": args.mode,
        "steps_completed": history[-1]["step"],
        "best_validation_metric": best_metric,
        "best_checkpoint": best_checkpoint,
        "final_adapter": str(final_dir),
        "baseline_validation": baseline,
        "last_validation": last_validation,
        "best_validation": best_validation,
        "reporter_correlation_artifact": (
            str(reporter_correlation_file)
            if args.mode in ("rl-qa", "rl-hybrid")
            else None
        ),
        "workspace_teacher_model": teacher_model,
        "teacher_matches_policy_reference": teacher_matches_policy,
        "effective_advantage_normalization": effective_advantage_mode,
        "dropout_audit": dropout_audit,
        "elapsed_seconds": time.time() - started,
        "diagnostics": diagnose(history, args.mode),
        "selector_training_diagnostics": selector_diagnostics,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
