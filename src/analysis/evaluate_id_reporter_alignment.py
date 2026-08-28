"""Compare locked admission reporters on the fixed training validation split.

This is deliberately an ID-only diagnostic.  It loads the three predeclared
training sources through :func:`build_training_bundle`, verifies every run's
sealed split manifest, and never accepts an evaluation/OOD battery path.
Correlation results are descriptive and must not be used for checkpoint or
Stage-C coefficient selection.

Example::

    python src/analysis/evaluate_id_reporter_alignment.py \
      --run sft-w=data/results/.../formal_sft-w... \
      --run rl-qa=data/results/.../formal_rl-qa... \
      --run hybrid-lw0.5=data/results/.../formal_rl-hybrid... \
      --batch-size 1 --out data/results/.../reports/id_reporters.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


SRC = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory_rl.data import build_training_bundle  # noqa: E402
from memory_rl.modeling import (  # noqa: E402
    binary_action_logits,
    load_policy_for_eval,
    render_admission_prompt,
)
from memory_rl.reporter_correlations import (  # noqa: E402
    compare_reporter_correlations,
    summarize_reporter_correlations,
)


EXPECTED_ID_SOURCES = {"explicit", "evoked", "evoked_g2"}
DEFAULT_MODEL_COMMIT = "a09a35458c702b33eeacc393d103063234e8bc28"
COMMON_CONDITION_LOCK = {
    "seed": 0,
    "split_seed": 0,
    "budget": 2,
    "max_steps": 300,
    "temperature": 5,
    "group_size": 8,
    "learning_rate": 1e-6,
    "lora_rank": 32,
    "lora_dropout": 0,
    "dtype": "bfloat16",
    "max_length": 2048,
    "workspace_objective": "rank-continuous",
    "workspace_top_k": 2,
    "teacher_mismatch_override": False,
}
CONDITION_LOCKS = {
    "sft-w": {"mode": "sft-w", "beta": 0, "lambda_qa": 1, "lambda_w": 0.5},
    "rl-qa": {"mode": "rl-qa", "beta": 0.03, "lambda_qa": 1, "lambda_w": 0},
    "hybrid-lw0.5": {
        "mode": "rl-hybrid",
        "beta": 0.03,
        "lambda_qa": 1,
        "lambda_w": 0.5,
    },
    "hybrid-lw0.25": {
        "mode": "rl-hybrid",
        "beta": 0.03,
        "lambda_qa": 1,
        "lambda_w": 0.25,
    },
    "hybrid-lw1.0": {
        "mode": "rl-hybrid",
        "beta": 0.03,
        "lambda_qa": 1,
        "lambda_w": 1,
    },
}


def _strict_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parse_named_run(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--run must use NAME=RUN_DIR")
    name, raw_path = (part.strip() for part in text.split("=", 1))
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("--run must use non-empty NAME=RUN_DIR")
    if name == "original":
        raise argparse.ArgumentTypeError("'original' is a reserved condition name")
    return name, Path(raw_path)


def _resolve_artifact_path(raw_path: Any, run_dir: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"best checkpoint path is missing in {run_dir}")
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, run_dir / path]
    candidates.append(run_dir / path.name)
    for candidate in candidates:
        if candidate.is_dir():
            resolved = candidate.resolve()
            try:
                resolved.relative_to(run_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"best checkpoint escapes its run directory: {resolved}"
                ) from exc
            return resolved
    raise ValueError(f"best checkpoint directory does not exist: {raw_path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_locked_run(
    name: str,
    run_dir: Path,
    *,
    model: str,
    teacher_tag: str,
    split_seed: int,
    workspace_top_k: int,
    expected_manifest: Mapping[str, Any],
    expected_model_commit: str,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist for {name!r}: {run_dir}")
    config = _strict_load(run_dir / "run_config.json")
    manifest = _strict_load(run_dir / "split_manifest.json")
    best = _strict_load(run_dir / "best_checkpoint.json")
    summary = _strict_load(run_dir / "summary.json")

    condition_lock = CONDITION_LOCKS.get(name)
    if condition_lock is None:
        raise ValueError(
            f"unknown locked condition {name!r}; expected one of "
            f"{sorted(CONDITION_LOCKS)}"
        )

    expected_fields = {
        "model": model,
        "teacher_tag": teacher_tag,
        "split_seed": split_seed,
        "workspace_top_k": workspace_top_k,
        "validation_episode_count": 45,
        "resolved_model_commit": expected_model_commit,
        **COMMON_CONDITION_LOCK,
        **condition_lock,
    }
    for key, expected in expected_fields.items():
        if config.get(key) != expected:
            raise ValueError(
                f"run {name!r} has {key}={config.get(key)!r}, expected {expected!r}"
            )
    probe_flag = config.get("probe_visible_to_policy")
    legacy_probe_flag_missing = probe_flag is None and config.get("mode") == "sft-w"
    if probe_flag is not False and not legacy_probe_flag_missing:
        raise ValueError(
            f"run {name!r} has invalid probe_visible_to_policy={probe_flag!r}"
        )
    if manifest != expected_manifest:
        raise ValueError(
            f"run {name!r} split manifest differs from the rebuilt fixed ID split"
        )
    adapter_path = _resolve_artifact_path(best.get("path"), run_dir)
    adapter_weights = adapter_path / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise ValueError(f"run {name!r} best checkpoint lacks adapter weights")
    step = best.get("step")
    metric = best.get("metric")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"run {name!r} has an invalid best checkpoint step")
    if (
        isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(float(metric))
    ):
        raise ValueError(f"run {name!r} has an invalid best checkpoint metric")
    if adapter_path.name != f"best-step-{step}":
        raise ValueError(
            f"run {name!r} checkpoint basename does not match step {step}"
        )
    summary_path = _resolve_artifact_path(summary.get("best_checkpoint"), run_dir)
    if summary_path != adapter_path:
        raise ValueError(f"run {name!r} summary selects a different checkpoint")
    summary_metric = summary.get("best_validation_metric")
    if (
        isinstance(summary_metric, bool)
        or not isinstance(summary_metric, (int, float))
        or float(summary_metric) != float(metric)
    ):
        raise ValueError(f"run {name!r} summary metric disagrees with best checkpoint")
    training_state_path = adapter_path / "training_state.json"
    training_state = _strict_load(training_state_path)
    if training_state.get("step") != step or training_state.get("metric") != metric:
        raise ValueError(f"run {name!r} adapter training_state disagrees with lock")
    return {
        "name": name,
        "run_dir": str(run_dir),
        "mode": config.get("mode"),
        "adapter_path": str(adapter_path),
        "adapter_weights_sha256": _sha256_file(adapter_weights),
        "checkpoint_step": step,
        "checkpoint_selection_metric": float(metric),
        "resolved_model_commit": config.get("resolved_model_commit"),
        "legacy_config_probe_flag_missing": legacy_probe_flag_missing,
        "training_protocol": {
            key: config.get(key) for key in sorted({*COMMON_CONDITION_LOCK, *condition_lock})
        },
        "artifact_sha256": {
            "run_config.json": _sha256_file(run_dir / "run_config.json"),
            "split_manifest.json": _sha256_file(run_dir / "split_manifest.json"),
            "best_checkpoint.json": _sha256_file(run_dir / "best_checkpoint.json"),
            "summary.json": _sha256_file(run_dir / "summary.json"),
            "training_state.json": _sha256_file(training_state_path),
        },
    }


@torch.no_grad()
def _score_id_candidates(
    policy,
    episodes: Sequence[object],
    *,
    batch_size: int,
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch size must be a positive integer")
    records: list[tuple[object, object, str]] = []
    token_id_rows: list[list[int]] = []
    old_truncation = policy.tokenizer.truncation_side
    policy.tokenizer.truncation_side = "left"
    try:
        for episode in episodes:
            for candidate in episode.candidates:
                prompt = render_admission_prompt(
                    policy.tokenizer, episode.context, candidate.concept
                )
                if episode.probe_question in prompt:
                    raise RuntimeError(
                        f"probe leaked into admission prompt for {candidate.uid}"
                    )
                records.append((episode, candidate, prompt))
                token_id_rows.append(
                    policy.tokenizer.encode(
                        prompt,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_length,
                    )
                )
    finally:
        policy.tokenizer.truncation_side = old_truncation

    rows: list[dict[str, Any]] = []
    policy.model.eval()
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        logits = binary_action_logits(
            policy.model,
            policy.tokenizer,
            [record[2] for record in chunk],
            policy.action_token_ids,
            policy.device,
            max_length,
        )
        probabilities = F.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        for (episode, candidate, _), probability in zip(chunk, probabilities):
            rows.append(
                {
                    "episode_id": episode.uid,
                    "candidate_id": candidate.uid,
                    "source": episode.source,
                    "v_rl": float(probability),
                    "w_ref": float(candidate.w_ref),
                    "y_utility": int(candidate.label == "load_bearing"),
                }
            )
        completed = min(start + len(chunk), len(records))
        if completed == len(records) or completed % 50 == 0:
            print(f"  scored {completed}/{len(records)} candidates", flush=True)
    return rows, {
        "episode_order_sha256": _sha256_json(
            list(dict.fromkeys(row["episode_id"] for row in rows))
        ),
        "candidate_order_sha256": _sha256_json(
            [row["candidate_id"] for row in rows]
        ),
        "rendered_prompt_sha256": _sha256_json(
            [record[2] for record in records]
        ),
        "prompt_token_ids_sha256": _sha256_json(token_id_rows),
    }


def _loaded_model_commit(policy) -> str | None:
    get_base = getattr(policy.model, "get_base_model", None)
    base = get_base() if callable(get_base) else policy.model
    for model in (base, policy.model):
        config = getattr(model, "config", None)
        commit = getattr(config, "_commit_hash", None)
        if isinstance(commit, str) and commit:
            return commit
    return None


def _active_adapters(policy) -> list[str]:
    active = getattr(policy.model, "active_adapters", None)
    if callable(active):
        try:
            active = active()
        except ValueError as exc:
            # Recent Transformers mixes PEFT helpers into plain base models;
            # the method is present but deliberately raises when no adapter is
            # loaded.  That is the expected state for the Original condition.
            if "No adapter loaded" not in str(exc):
                raise
            return []
    if active is None:
        active = getattr(policy.model, "active_adapter", None)
    if active is None:
        return []
    if isinstance(active, str):
        return [active]
    return [str(value) for value in active]


def _release(policy) -> None:
    if policy is not None:
        del policy.model
        del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_named_run)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--expected-model-commit", default=DEFAULT_MODEL_COMMIT)
    parser.add_argument("--teacher-tag", default="7B-Instruct")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--workspace-top-k", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    named_runs = dict(args.run)
    if len(named_runs) != len(args.run):
        parser.error("--run condition names must be unique")
    if set(named_runs) != set(CONDITION_LOCKS):
        parser.error(
            "locked Stage-C reporter comparison requires exactly these conditions: "
            + ", ".join(sorted(CONDITION_LOCKS))
        )
    if args.model != "Qwen/Qwen2.5-7B-Instruct":
        parser.error("this locked diagnostic requires Qwen/Qwen2.5-7B-Instruct")
    if args.teacher_tag != "7B-Instruct":
        parser.error("this locked diagnostic requires teacher tag 7B-Instruct")
    if args.expected_model_commit != DEFAULT_MODEL_COMMIT:
        parser.error(f"this locked diagnostic requires model commit {DEFAULT_MODEL_COMMIT}")
    if args.workspace_top_k != 2:
        parser.error("this locked Stage-C ID diagnostic requires workspace top-k=2")
    if args.split_seed != 0:
        parser.error("this locked Stage-C ID diagnostic requires split seed 0")
    if args.batch_size != 1:
        parser.error("this locked diagnostic requires admission batch size 1")
    if args.max_length != 2048:
        parser.error("this locked diagnostic requires max length 2048")
    if args.dtype != "bfloat16":
        parser.error("this locked diagnostic requires bfloat16")
    if args.bootstrap_samples < 0:
        parser.error("bootstrap samples must be non-negative")
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")

    data = build_training_bundle(
        repo_root=REPO_ROOT,
        teacher_tag=args.teacher_tag,
        seed=args.split_seed,
        top_k=args.workspace_top_k,
    )
    sources = {episode.source for episode in data.validation_episodes}
    if sources != EXPECTED_ID_SOURCES:
        raise RuntimeError(
            f"ID source guard failed: observed={sorted(sources)}, "
            f"expected={sorted(EXPECTED_ID_SOURCES)}"
        )
    if len(data.validation_episodes) != 45:
        raise RuntimeError("locked ID validation split must contain exactly 45 episodes")
    candidate_count = sum(len(episode.candidates) for episode in data.validation_episodes)
    if candidate_count != 215:
        raise RuntimeError("locked ID validation split must contain exactly 215 candidates")

    locked_runs = [
        _load_locked_run(
            name,
            path,
            model=args.model,
            teacher_tag=args.teacher_tag,
            split_seed=args.split_seed,
            workspace_top_k=args.workspace_top_k,
            expected_manifest=data.split_manifest,
            expected_model_commit=args.expected_model_commit,
        )
        for name, path in args.run
    ]
    commits = {run["resolved_model_commit"] for run in locked_runs}
    if commits != {args.expected_model_commit}:
        raise RuntimeError(f"locked runs disagree on model revision: {sorted(map(str, commits))}")

    condition_specs = [
        {
            "name": "original",
            "mode": "original",
            "adapter_path": None,
            "checkpoint_step": None,
            "checkpoint_selection_metric": None,
            "run_dir": None,
            "adapter_weights_sha256": None,
            "legacy_config_probe_flag_missing": False,
            "training_protocol": None,
            "artifact_sha256": None,
        },
        *locked_runs,
    ]
    conditions: dict[str, dict[str, Any]] = {}
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    common_scoring_hashes: dict[str, str] | None = None
    expected_commit = args.expected_model_commit
    for condition in condition_specs:
        name = condition["name"]
        print(f"scoring fixed ID reporter: {name}", flush=True)
        policy = None
        try:
            policy = load_policy_for_eval(
                args.model,
                condition["adapter_path"],
                args.device,
                args.dtype,
            )
            observed_commit = _loaded_model_commit(policy)
            if observed_commit != expected_commit:
                raise RuntimeError(
                    f"loaded model revision for {name!r} is {observed_commit!r}, "
                    f"expected {expected_commit!r}"
                )
            active_adapters = _active_adapters(policy)
            if condition["adapter_path"] is not None and "default" not in active_adapters:
                raise RuntimeError(
                    f"adapter is not active for {name!r}: {active_adapters!r}"
                )
            rows, scoring_hashes = _score_id_candidates(
                policy,
                data.validation_episodes,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        finally:
            _release(policy)
        rows_by_condition[name] = rows
        if common_scoring_hashes is None:
            common_scoring_hashes = scoring_hashes
        elif scoring_hashes != common_scoring_hashes:
            raise RuntimeError(
                f"condition {name!r} changed candidate order, prompts, or token IDs"
            )
        conditions[name] = {
            key: value for key, value in condition.items() if key != "name"
        }
        definition = (
            "base-model P(Yes) over aggregated constrained No/Yes logits at temperature 1"
            if condition["adapter_path"] is None
            else "adapter-enabled P(Yes) over aggregated constrained No/Yes logits at temperature 1"
        )
        conditions[name]["summary"] = summarize_reporter_correlations(
            rows,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            v_rl_definition=definition,
        )
        conditions[name]["candidate_rows"] = rows
        conditions[name]["scoring_audit"] = {
            **scoring_hashes,
            "loaded_model_commit": observed_commit,
            "adapter_enabled": condition["adapter_path"] is not None,
            "active_adapters": active_adapters,
        }

    print("computing paired source-stratified episode bootstrap", flush=True)
    paired = compare_reporter_correlations(
        rows_by_condition,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = {
        "schema_version": 1,
        "status": "completed",
        "scope": "fixed ID validation candidates only",
        "ood_accessed": False,
        "reporter_correlations_used_for_selection": False,
        "protocol": {
            "model": args.model,
            "resolved_model_commit": expected_commit,
            "teacher_tag": args.teacher_tag,
            "split_seed": args.split_seed,
            "workspace_top_k": args.workspace_top_k,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "device": args.device,
            "admission_batch_size": args.batch_size,
            "v_rl": "condition-policy P(Yes) over aggregated constrained No/Yes logits at temperature 1; adapter state is recorded per condition",
            "w_ref": "immutable raw W_rr from the matched frozen reference model",
            "y_utility": "1 iff candidate.label == load_bearing, else 0",
        },
        "split": {
            "manifest_sha256": data.split_manifest["manifest_sha256"],
            "validation_episode_count": len(data.validation_episodes),
            "validation_candidate_count": candidate_count,
            "sources": sorted(sources),
        },
        "conditions": conditions,
        "paired_correlation_differences": paired,
    }
    _write_json_once(args.out, output)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
