#!/usr/bin/env python3
"""One-shot, lock-bound M1 OOD evaluation for Metacognitive Alignment.

This command is intentionally limited to Decoupled and Compositional.  It
loads the ID-selected LoRA checkpoint once, measures its verbal/workspace
channels, evaluates adapter-disabled/enabled full-context QA in identical item
order, and writes paired episode-cluster bootstrap intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from experiments.measure import (  # noqa: E402
    concept_token_ids,
    verbal_salience,
    workspace_salience,
    yes_no_ids,
)
from jlens import WorkspaceLens  # noqa: E402
from memory_rl.recall import full_context_prompt, grade_answer  # noqa: E402
from run_metacog_alignment_campaign import (  # noqa: E402
    DEFAULT_BOOTSTRAP_DRAWS,
    EXPECTED_GPU,
    EXPECTED_MODEL,
    CampaignError,
    sha256_file,
    validate_orchestrator_lock,
)


RESULT_SCHEMA = "metacog-alignment-m1-ood/v1"


class EvaluationError(RuntimeError):
    pass


def _load_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"missing or unsafe {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read {label} {path}: {exc}") from exc


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binary_label(row: Mapping[str, Any]) -> int:
    return int(row.get("label") == "load_bearing")


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Tie-aware Mann-Whitney AUC without a heavyweight metric dependency."""

    if len(scores) != len(labels) or not scores:
        raise EvaluationError("AUC vectors must be non-empty and equal length")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise EvaluationError("AUC requires both utility classes")
    order = sorted(range(len(scores)), key=lambda index: float(scores[index]))
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        score = float(scores[order[cursor]])
        while end < len(order) and float(scores[order[end]]) == score:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for offset in range(cursor, end):
            ranks[order[offset]] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise EvaluationError("cannot take percentile of an empty distribution")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _interval(estimate: float, draws: Sequence[float]) -> dict[str, Any]:
    return {
        "estimate": float(estimate),
        "ci_95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "bootstrap_samples_effective": len(draws),
    }


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        episode = row.get("episode")
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
            raise EvaluationError("measurement row has invalid episode")
        grouped.setdefault(episode, []).append(row)
    return grouped


def summarize_channel(
    rows: Sequence[Mapping[str, Any]], key: str, *, include_yes_rate: bool
) -> dict[str, Any]:
    scores = [float(row[key]) for row in rows]
    if not all(math.isfinite(score) for score in scores):
        raise EvaluationError(f"{key} contains non-finite values")
    labels = [_binary_label(row) for row in rows]
    grouped = _group_rows(rows)
    episode_aucs: list[float] = []
    for episode_rows in grouped.values():
        episode_labels = [_binary_label(row) for row in episode_rows]
        if 0 < sum(episode_labels) < len(episode_labels):
            episode_aucs.append(auc([float(row[key]) for row in episode_rows], episode_labels))
    result = {
        "pooled_auc": auc(scores, labels),
        "within_episode_auc": sum(episode_aucs) / len(episode_aucs),
        "n_within_episode_auc": len(episode_aucs),
        "candidate_count": len(rows),
        "episode_count": len(grouped),
    }
    if include_yes_rate:
        result["yes_rate"] = sum(score >= 0.5 for score in scores) / len(scores)
    return result


def paired_episode_bootstrap(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    key: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    before_by_episode = _group_rows(before)
    after_by_episode = _group_rows(after)
    episodes = sorted(before_by_episode)
    if episodes != sorted(after_by_episode):
        raise EvaluationError("before/after episode sets differ")
    rng = random.Random(seed)
    before_draws: list[float] = []
    after_draws: list[float] = []
    difference_draws: list[float] = []
    for _ in range(samples):
        sampled = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        before_rows = [row for episode in sampled for row in before_by_episode[episode]]
        after_rows = [row for episode in sampled for row in after_by_episode[episode]]
        labels = [_binary_label(row) for row in before_rows]
        try:
            before_auc = auc([float(row[key]) for row in before_rows], labels)
            after_auc = auc([float(row[key]) for row in after_rows], labels)
        except EvaluationError:
            continue
        before_draws.append(before_auc)
        after_draws.append(after_auc)
        difference_draws.append(after_auc - before_auc)
    if len(difference_draws) != samples:
        raise EvaluationError(
            f"episode bootstrap produced {len(difference_draws)}/{samples} valid draws"
        )
    before_point = auc([float(row[key]) for row in before], [_binary_label(row) for row in before])
    after_point = auc([float(row[key]) for row in after], [_binary_label(row) for row in after])
    return {
        "before": _interval(before_point, before_draws),
        "after": _interval(after_point, after_draws),
        "after_minus_before": {
            **_interval(after_point - before_point, difference_draws),
            "probability_gt_zero": sum(value > 0 for value in difference_draws)
            / len(difference_draws),
        },
    }


def _align_before_after(
    baseline: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> None:
    def identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("episode"),
            row.get("candidate_index"),
            row.get("concept"),
            row.get("label"),
        )

    before_ids = [identity(row) for row in baseline]
    after_ids = [identity(row) for row in after]
    if before_ids != after_ids:
        raise EvaluationError("before/after candidate identity or order differs")
    if not baseline:
        raise EvaluationError("baseline measurement is empty")
    for key in ("V", "W_rr"):
        if any(key not in row for row in baseline) or any(key not in row for row in after):
            raise EvaluationError(f"before/after rows must all contain {key}")


@torch.inference_mode()
def measure_adapter_condition(
    lens: WorkspaceLens,
    battery: Sequence[Mapping[str, Any]],
    condition: str,
) -> list[dict[str, Any]]:
    yes_ids, no_ids = yes_no_ids(lens)
    rows: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(battery):
        context = str(episode["context"])
        items = episode["items"]
        by_concept: dict[str, list[int]] = {}
        for item in items:
            concept = str(item["concept"])
            token_ids = concept_token_ids(lens, concept)
            if not token_ids:
                raise EvaluationError(
                    f"{condition} episode {episode_index} concept has no token ids: {concept!r}"
                )
            by_concept[concept] = token_ids
        all_ids = sorted({token_id for ids in by_concept.values() for token_id in ids})
        _, _, reciprocal_ranks = workspace_salience(lens, context, all_ids, end_only=True)
        for candidate_index, item in enumerate(items):
            concept = str(item["concept"])
            ids = by_concept[concept]
            rows.append(
                {
                    "episode": episode_index,
                    "candidate_index": candidate_index,
                    "concept": concept,
                    "label": item["label"],
                    "V": verbal_salience(lens, context, concept, yes_ids, no_ids),
                    "W_rr": max(reciprocal_ranks[token_id] for token_id in ids),
                }
            )
        if (episode_index + 1) % 5 == 0:
            print(f"  [{condition}] measured {episode_index + 1}/{len(battery)} episodes", flush=True)
    return rows


@torch.inference_mode()
def _generate_full_context(lens: WorkspaceLens, context: str, question: str, max_tokens: int) -> str:
    prompt = full_context_prompt(context, question)
    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = lens.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = lens.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    encoded = lens.tok(rendered, return_tensors="pt").to(lens.device)
    generated = lens.model.generate(
        **encoded,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=lens.tok.pad_token_id,
    )
    continuation = generated[0, encoded["input_ids"].shape[1] :]
    return lens.tok.decode(continuation, skip_special_tokens=True).strip()


@contextmanager
def _adapter_disabled(model: Any):
    disable = getattr(model, "disable_adapter", None)
    if not callable(disable):
        raise EvaluationError("loaded checkpoint model cannot disable its adapter")
    with disable():
        yield


def evaluate_full_context(
    lens: WorkspaceLens,
    battery: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []
    with _adapter_disabled(lens.model):
        for episode_index, episode in enumerate(battery):
            answer = _generate_full_context(
                lens, str(episode["context"]), str(episode["probe_question"]), max_tokens
            )
            before.append(
                {
                    "episode": episode_index,
                    "answer": answer,
                    "correct": grade_answer(answer, str(episode["answer"])),
                }
            )
    for episode_index, episode in enumerate(battery):
        answer = _generate_full_context(
            lens, str(episode["context"]), str(episode["probe_question"]), max_tokens
        )
        after.append(
            {
                "episode": episode_index,
                "answer": answer,
                "correct": grade_answer(answer, str(episode["answer"])),
            }
        )
    before_accuracy = sum(row["correct"] for row in before) / len(before)
    after_accuracy = sum(row["correct"] for row in after) / len(after)
    return {
        "prompt": "full context + concise probe; identical before/after item order",
        "generation": {"do_sample": False, "max_new_tokens": max_tokens},
        "before_accuracy": before_accuracy,
        "after_accuracy": after_accuracy,
        "after_minus_before": after_accuracy - before_accuracy,
        "drop_percentage_points": (before_accuracy - after_accuracy) * 100.0,
        "per_episode": {"before": before, "after": after},
    }


def _paired_accuracy_bootstrap(
    full_context: Mapping[str, Any], samples: int, seed: int
) -> dict[str, Any]:
    before = full_context["per_episode"]["before"]
    after = full_context["per_episode"]["after"]
    if [row["episode"] for row in before] != [row["episode"] for row in after]:
        raise EvaluationError("full-context before/after episode order differs")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(before)) for _ in before]
        before_acc = sum(bool(before[index]["correct"]) for index in indices) / len(indices)
        after_acc = sum(bool(after[index]["correct"]) for index in indices) / len(indices)
        draws.append(after_acc - before_acc)
    return _interval(float(full_context["after_minus_before"]), draws)


def summarize_condition(
    baseline: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    full_context: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    _align_before_after(baseline, after)
    channels: dict[str, Any] = {}
    for output_name, row_key in (("verbal", "V"), ("workspace", "W_rr")):
        include_yes_rate = row_key == "V"
        before_summary = summarize_channel(
            baseline, row_key, include_yes_rate=include_yes_rate
        )
        after_summary = summarize_channel(
            after, row_key, include_yes_rate=include_yes_rate
        )
        channels[output_name] = {
            "before": before_summary,
            "after": after_summary,
            "delta_pooled_auc": after_summary["pooled_auc"] - before_summary["pooled_auc"],
            "delta_within_episode_auc": (
                after_summary["within_episode_auc"] - before_summary["within_episode_auc"]
            ),
            "paired_episode_bootstrap": paired_episode_bootstrap(
                baseline,
                after,
                row_key,
                samples=bootstrap_samples,
                seed=bootstrap_seed + (0 if row_key == "V" else 1),
            ),
        }
        if include_yes_rate:
            channels[output_name]["delta_yes_rate"] = (
                after_summary["yes_rate"] - before_summary["yes_rate"]
            )
    full_context_payload = dict(full_context)
    full_context_payload["paired_episode_bootstrap"] = _paired_accuracy_bootstrap(
        full_context, bootstrap_samples, bootstrap_seed + 2
    )
    return {
        "verbal": channels["verbal"],
        "workspace": channels["workspace"],
        "full_context_qa": full_context_payload,
        "candidate_identity_sha256": _canonical_hash(
            [
                [row["episode"], row["candidate_index"], row["concept"], row["label"]]
                for row in baseline
            ]
        ),
        "per_item": {"before": list(baseline), "after": list(after)},
    }


def gate_decision(decoupled: Mapping[str, Any]) -> tuple[str, list[str], bool, bool]:
    verbal = decoupled["verbal"]
    workspace = decoupled["workspace"]
    qa = decoupled["full_context_qa"]
    delta_v = float(verbal["delta_pooled_auc"])
    v_after = float(verbal["after"]["pooled_auc"])
    delta_w = float(workspace["delta_pooled_auc"])
    qa_drop_pp = float(qa["drop_percentage_points"])
    no_harm = abs(delta_w) < 0.03 and qa_drop_pp <= 2.0
    reasons: list[str] = []
    if delta_v <= 0.05:
        reasons.append("Decoupled Delta V <= +0.05")
    if delta_w < -0.03:
        reasons.append("Decoupled workspace AUC drop > 0.03")
    if qa_drop_pp > 2.0:
        reasons.append("Decoupled full-context QA drop > 2pp")
    if reasons:
        return "RED", reasons, False, False
    if delta_v >= 0.15 and v_after > 0.50 and abs(delta_w) < 0.03 and no_harm:
        return (
            "GREEN",
            ["all preregistered M1 GREEN conditions passed"],
            delta_v >= 0.25,
            False,
        )
    if 0.05 < delta_v < 0.15 and no_harm:
        return (
            "AMBER",
            ["directional gain with no-harm passed, below GREEN effect size"],
            False,
            True,
        )
    return (
        "AMBER",
        ["no RED condition fired, but at least one GREEN condition did not pass"],
        False,
        False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-manifest", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--baseline-decoupled", type=Path, required=True)
    parser.add_argument("--baseline-compositional", type=Path, required=True)
    parser.add_argument("--decoupled-battery", type=Path, required=True)
    parser.add_argument("--compositional-battery", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--qa-max-new-tokens", type=int, default=64)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bootstrap_samples != DEFAULT_BOOTSTRAP_DRAWS:
        parser.error(f"--bootstrap-samples must be exactly {DEFAULT_BOOTSTRAP_DRAWS}")
    if args.bootstrap_seed < 0:
        parser.error("--bootstrap-seed must be non-negative")
    if args.qa_max_new_tokens <= 0:
        parser.error("--qa-max-new-tokens must be positive")
    if not re.fullmatch(r"[0-9a-f]{32}", args.attempt_id):
        parser.error("--attempt-id must be the launcher's 32-hex precommitment")
    if args.out_json.exists() or args.out_json.is_symlink():
        parser.error(f"refusing to overwrite existing output: {args.out_json}")

    lock_path = args.lock_manifest.resolve()
    run_dir = lock_path.parent.parent
    try:
        lock = validate_orchestrator_lock(lock_path, run_dir)
    except CampaignError as exc:
        parser.error(str(exc))
    lock_hash = sha256_file(lock_path)
    attempt_marker_path = run_dir / "ood" / "attempt_started.json"
    marker = _load_json(attempt_marker_path, "OOD attempt marker")
    if marker.get("attempt_id") != args.attempt_id:
        parser.error("OOD attempt marker does not match --attempt-id")
    if marker.get("lock_manifest_sha256") != lock_hash:
        parser.error("OOD attempt marker does not match the validated lock")

    model = lock.get("model")
    model_revision = lock.get("model_revision")
    tokenizer_revision = lock.get("tokenizer_revision")
    if model != EXPECTED_MODEL:
        parser.error(f"lock model must be {EXPECTED_MODEL}")
    if not isinstance(model_revision, str) or not isinstance(tokenizer_revision, str):
        parser.error("lock must contain pinned model/tokenizer revisions")
    checkpoint_path = run_dir / str(lock["checkpoint_path"])

    inputs = {
        "decoupled": {
            "baseline_path": args.baseline_decoupled.resolve(),
            "battery_path": args.decoupled_battery.resolve(),
        },
        "compositional": {
            "baseline_path": args.baseline_compositional.resolve(),
            "battery_path": args.compositional_battery.resolve(),
        },
    }
    locked_conditions = lock["m0_artifacts"]["conditions"]
    for condition in ("decoupled", "compositional"):
        binding = locked_conditions[condition]
        expected_baseline = (run_dir / binding["raw_path"]).resolve()
        expected_battery = Path(binding["battery_path"]).resolve()
        if inputs[condition]["baseline_path"] != expected_baseline:
            parser.error(f"{condition} baseline path differs from the campaign lock")
        if inputs[condition]["battery_path"] != expected_battery:
            parser.error(f"{condition} battery path differs from the campaign lock")
        if sha256_file(expected_baseline) != binding["raw_sha256"]:
            parser.error(f"{condition} baseline hash differs from the campaign lock")
        if sha256_file(expected_battery) != binding["battery_sha256"]:
            parser.error(f"{condition} battery hash differs from the campaign lock")
    loaded: dict[str, dict[str, Any]] = {}
    for condition, paths in inputs.items():
        baseline = _load_json(paths["baseline_path"], f"{condition} baseline")
        battery = _load_json(paths["battery_path"], f"{condition} battery")
        if not isinstance(baseline, list) or not isinstance(battery, list) or not battery:
            parser.error(f"{condition} baseline and battery must be non-empty arrays")
        loaded[condition] = {"baseline": baseline, "battery": battery}

    print(f"loading locked adapter once: {checkpoint_path}", flush=True)
    lens = WorkspaceLens(
        EXPECTED_MODEL,
        device="cuda",
        dtype=torch.bfloat16,
        adapter_path=str(checkpoint_path),
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    )
    actual_gpu_name = torch.cuda.get_device_name(0)
    if actual_gpu_name != EXPECTED_GPU:
        parser.error(f"OOD evaluator requires {EXPECTED_GPU}, found {actual_gpu_name!r}")
    if lens.model_revision_resolved not in {None, model_revision}:
        parser.error("resolved OOD model revision differs from the campaign lock")
    if lens.tokenizer_revision_resolved not in {None, tokenizer_revision}:
        parser.error("resolved OOD tokenizer revision differs from the campaign lock")
    parameter_dtype = str(next(lens.model.parameters()).dtype)
    if parameter_dtype != "torch.bfloat16":
        parser.error(f"OOD evaluator requires bf16 model weights, found {parameter_dtype}")
    yes_ids, no_ids = yes_no_ids(lens)
    chat_template = getattr(lens.tok, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        parser.error("OOD tokenizer has no chat template")
    evaluator_provenance = {
        "gpu_name": actual_gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parameter_dtype": parameter_dtype,
        "model_revision_requested": model_revision,
        "model_revision_resolved": lens.model_revision_resolved,
        "tokenizer_revision_requested": tokenizer_revision,
        "tokenizer_revision_resolved": lens.tokenizer_revision_resolved,
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        "measure_source_sha256": sha256_file(SRC_ROOT / "experiments" / "measure.py"),
        "workspace_lens_source_sha256": sha256_file(SRC_ROOT / "jlens.py"),
        "recall_source_sha256": sha256_file(SRC_ROOT / "memory_rl" / "recall.py"),
    }
    condition_payloads: dict[str, Any] = {}
    for index, condition in enumerate(("decoupled", "compositional")):
        battery = loaded[condition]["battery"]
        after = measure_adapter_condition(lens, battery, condition)
        full_context = evaluate_full_context(
            lens, battery, max_tokens=args.qa_max_new_tokens
        )
        condition_payloads[condition] = summarize_condition(
            loaded[condition]["baseline"],
            after,
            full_context,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + index * 10,
        )

    decision, reasons, strong_green, controlled_branch = gate_decision(
        condition_payloads["decoupled"]
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "stage": "M1_OOD",
        "attempt_id": args.attempt_id,
        "lock_manifest": str(lock_path),
        "lock_manifest_sha256": lock_hash,
        "model": EXPECTED_MODEL,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_tree_sha256": lock["checkpoint_tree_sha256"],
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "episode_cluster",
            "paired_before_after": True,
        },
        "conditions": condition_payloads,
        "decision": decision,
        "decision_reasons": reasons,
        "strong_green": strong_green,
        "controlled_branch_authorized": controlled_branch,
        "inputs": {
            condition: {
                "baseline_path": str(paths["baseline_path"]),
                "baseline_sha256": sha256_file(paths["baseline_path"]),
                "battery_path": str(paths["battery_path"]),
                "battery_sha256": sha256_file(paths["battery_path"]),
            }
            for condition, paths in inputs.items()
        },
        "protocol": {
            "ood_conditions": ["decoupled", "compositional"],
            "checkpoint_selected_on": "id_validation_only",
            "workspace_readout": "W_rr at final context position, max over layers",
            "verbal_readout": "constrained Yes/No probability",
            "full_context_before": "same loaded model with adapter disabled",
            "full_context_after": "same loaded model with locked adapter enabled",
            "qa_prompt_and_item_order_identical": True,
            "h100_used": False,
        },
        "provenance": evaluator_provenance,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out_json.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        parser.error(f"refusing to overwrite existing output: {args.out_json}")
    print(f"M1 decision={decision}; saved one-shot OOD result -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
