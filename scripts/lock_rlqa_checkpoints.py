#!/usr/bin/env python3
"""Freeze the ID-selected RL-QA checkpoints before any OOD access.

Track A's contract is the same shape as the metacognitive campaign's: every
seed must hold an ID-only checkpoint lock, and all locks must exist before
Decoupled or Compositional is opened for any seed.  This command records the
selected step, its tree hash, and the run configuration that produced it, and
it fails closed if a run selected on anything but ID validation, changed the
locked recipe, or already touched an OOD battery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_metacog_alignment_campaign import (  # noqa: E402
    canonical_json_sha256,
    checkpoint_tree_hash,
    sha256_file,
    utc_now,
)

LOCK_SCHEMA = "memory-rl-qa-id-lock/v1"
# One model per campaign, from the launcher's closed allowlist.
EXPECTED_MODEL = os.environ.get("METACOG_EXPECTED_MODEL", "Qwen/Qwen3-8B")
# Everything the H100/A100 Track A specification freezes before formal seeds.
LOCKED_RECIPE = {
    "mode": "rl-qa",
    "lambda_qa": 1.0,
    "lambda_w": 0.0,
    "budget": 2,
    "group_size": 8,
    "grpo_epochs": 2,
    "lora_rank": 32,
    "max_steps": 300,
    "learning_rate": 1e-6,
    "beta": 0.03,
    "split_seed": 0,
    # Frozen by data/results/rlqa_a100/RECIPE_FREEZE.json after Stage B0 showed
    # that 5.0 still satisfies both B0 gates on Qwen3-8B.
    "temperature": 5.0,
}
# Pinned commit per approved model; a run that resolved anything else is refused.
PINNED_REVISIONS = {
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
}
EXPECTED_REVISION = PINNED_REVISIONS.get(EXPECTED_MODEL)
# Any of these appearing in a training run's configuration means the sealed
# conditions were opened before the lock existed.
SEALED_TOKENS = ("battery_v4", "battery_v3", "decoupled", "compositional")


class LockError(RuntimeError):
    pass


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LockError(f"missing or unsafe {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LockError(f"{label} must be a JSON object: {path}")
    return value


def _check_recipe(run_config: Mapping[str, Any], seed: int) -> None:
    if run_config.get("model", "").rstrip("/") != EXPECTED_MODEL:
        raise LockError(f"seed {seed} did not train {EXPECTED_MODEL}")
    if int(run_config.get("seed", -1)) != seed:
        raise LockError(f"seed {seed} run config records seed {run_config.get('seed')!r}")
    for field, expected in LOCKED_RECIPE.items():
        observed = run_config.get(field)
        if observed is None:
            raise LockError(f"seed {seed} run config has no {field}")
        if isinstance(expected, float):
            if abs(float(observed) - expected) > 1e-12:
                raise LockError(f"seed {seed} changed {field}: {observed} != {expected}")
        elif str(observed) != str(expected):
            raise LockError(f"seed {seed} changed {field}: {observed!r} != {expected!r}")
    resolved = run_config.get("resolved_model_commit")
    if EXPECTED_REVISION and resolved is not None and str(resolved) != EXPECTED_REVISION:
        raise LockError(f"seed {seed} resolved a different model commit: {resolved}")
    blob = json.dumps(run_config, sort_keys=True).lower()
    for token in SEALED_TOKENS:
        if token in blob:
            raise LockError(f"seed {seed} training config references sealed data: {token}")


def lock_run(run_dir: Path, seed: int) -> dict[str, Any]:
    run_config = _load_json(run_dir / "run_config.json", f"seed {seed} run config")
    _check_recipe(run_config, seed)
    best = _load_json(run_dir / "best_checkpoint.json", f"seed {seed} best checkpoint")
    raw = Path(str(best.get("path", "")))
    # The trainer records the checkpoint the way it was invoked: absolute, or
    # relative to the repository root (not to the run directory).  Try both
    # readings rather than guessing one.
    candidates = [raw] if raw.is_absolute() else [REPO_ROOT / raw, run_dir / raw]
    checkpoint = next((c for c in candidates if c.is_dir() and not c.is_symlink()), None)
    if checkpoint is None:
        raise LockError(
            f"seed {seed} selected checkpoint is missing or unsafe: "
            + ", ".join(str(c) for c in candidates)
        )
    checkpoint = checkpoint.resolve()
    tree_hash, files = checkpoint_tree_hash(checkpoint)
    summary = _load_json(run_dir / "summary.json", f"seed {seed} summary")
    recorded = str(summary.get("best_checkpoint", "")).rstrip("/")
    if recorded and Path(recorded).resolve() != checkpoint:
        raise LockError(
            f"seed {seed} summary and best_checkpoint.json disagree: {recorded} != {checkpoint}"
        )
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "step": best.get("step"),
        "selection_scope": "id_validation",
        "selection_metric": best.get("metric_name", "id_qa"),
        "selection_value": best.get("metric"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": tree_hash,
        "checkpoint_file_hashes": files,
        "run_config_sha256": sha256_file(run_dir / "run_config.json"),
        "summary_sha256": sha256_file(run_dir / "summary.json"),
        "split_manifest_sha256": sha256_file(run_dir / "split_manifest.json"),
    }


def build_lock(runs: Mapping[int, Path], expected_seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    if sorted(runs) != sorted(expected_seeds):
        raise LockError(
            f"seeds {sorted(expected_seeds)} must be locked together, got {sorted(runs)}"
        )
    seeds = [lock_run(runs[seed], seed) for seed in sorted(runs)]
    splits = {entry["split_manifest_sha256"] for entry in seeds}
    if len(splits) != 1:
        raise LockError("seeds do not share one split manifest; split seed must stay 0")
    del expected_seeds  # consumed above
    payload = {
        "schema_version": LOCK_SCHEMA,
        "created_at_utc": utc_now(),
        "model": EXPECTED_MODEL,
        "locked_recipe": LOCKED_RECIPE,
        "shared_split_manifest_sha256": splits.pop(),
        "seeds": seeds,
        "ood_authorization": {
            "conditions": ["decoupled", "compositional"],
            "attempt_limit": 1,
            "bootstrap_samples": 4000,
        },
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SEED=RUN_DIR",
        help="repeat once per seed, e.g. --run 0=data/results/rlqa_a100/runs/formal_rl-qa_...",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="comma-separated seed set that must be locked together; the 32B "
             "scaling gate authorises seed 0 alone",
    )
    args = parser.parse_args(argv)

    runs: dict[int, Path] = {}
    for item in args.run:
        if "=" not in item:
            parser.error("--run must be SEED=RUN_DIR")
        seed_text, path = item.split("=", 1)
        runs[int(seed_text)] = Path(path).resolve()

    try:
        payload = build_lock(runs, [int(s) for s in args.seeds.split(",") if s.strip()])
    except (LockError, OSError, ValueError) as exc:
        print(f"error: {exc}\nSTOP: no OOD battery was opened.", file=sys.stderr)
        return 1

    if args.out.exists() or args.out.is_symlink():
        print(f"error: refusing to overwrite existing lock: {args.out}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"LOCKED {len(payload['seeds'])} seeds -> {args.out}")
    for entry in payload["seeds"]:
        print(f"  seed {entry['seed']}: step {entry['step']} tree {entry['checkpoint_tree_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
