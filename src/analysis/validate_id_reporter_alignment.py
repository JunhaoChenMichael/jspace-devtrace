"""Strictly recompute a saved fixed-ID reporter-alignment artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory_rl.reporter_correlations import (  # noqa: E402
    compare_reporter_correlations,
    summarize_reporter_correlations,
)


EXPECTED_CONDITIONS = {
    "original",
    "sft-w",
    "rl-qa",
    "hybrid-lw0.5",
    "hybrid-lw0.25",
    "hybrid-lw1.0",
}
EXPECTED_MANIFEST = "1988c8af1fc39884a7bd711f335f91db6c890612e6e0a45917488ba4b44abf0f"
EXPECTED_COMMIT = "a09a35458c702b33eeacc393d103063234e8bc28"


def _strict_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError("top-level artifact must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        artifact = _strict_load(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "fail", "errors": [f"strict JSON parse failed: {exc}"]}

    protocol = artifact.get("protocol", {})
    split = artifact.get("split", {})
    conditions = artifact.get("conditions")
    fixed_checks = {
        "status": artifact.get("status") == "completed",
        "scope": artifact.get("scope") == "fixed ID validation candidates only",
        "ood_accessed": artifact.get("ood_accessed") is False,
        "selection_prohibition": artifact.get("reporter_correlations_used_for_selection") is False,
        "model_commit": protocol.get("resolved_model_commit") == EXPECTED_COMMIT,
        "batch_size": protocol.get("admission_batch_size") == 1,
        "dtype": protocol.get("dtype") == "bfloat16",
        "max_length": protocol.get("max_length") == 2048,
        "manifest": split.get("manifest_sha256") == EXPECTED_MANIFEST,
        "episodes": split.get("validation_episode_count") == 45,
        "candidates": split.get("validation_candidate_count") == 215,
        "condition_set": isinstance(conditions, dict)
        and set(conditions) == EXPECTED_CONDITIONS,
    }
    errors.extend(name for name, passed in fixed_checks.items() if not passed)
    if not isinstance(conditions, dict) or set(conditions) != EXPECTED_CONDITIONS:
        return {
            "status": "fail",
            "errors": errors,
            "details": {"fixed_checks": fixed_checks},
        }

    rows_by_condition = {}
    scoring_hashes = []
    recomputed_conditions = []
    for name, condition in conditions.items():
        rows = condition.get("candidate_rows")
        summary = condition.get("summary")
        audit = condition.get("scoring_audit", {})
        if not isinstance(rows, list) or len(rows) != 215 or not isinstance(summary, dict):
            errors.append(f"{name}: invalid rows/summary")
            continue
        expected_adapter = name != "original"
        if audit.get("adapter_enabled") is not expected_adapter:
            errors.append(f"{name}: adapter_enabled provenance mismatch")
        if audit.get("loaded_model_commit") != EXPECTED_COMMIT:
            errors.append(f"{name}: loaded model commit mismatch")
        scoring_hashes.append(
            tuple(
                audit.get(key)
                for key in (
                    "episode_order_sha256",
                    "candidate_order_sha256",
                    "rendered_prompt_sha256",
                    "prompt_token_ids_sha256",
                )
            )
        )
        definition = summary.get("definitions", {}).get("v_rl")
        try:
            recomputed = summarize_reporter_correlations(
                rows,
                bootstrap_samples=4000,
                bootstrap_seed=0,
                v_rl_definition=definition,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"{name}: recomputation failed: {exc}")
            continue
        if recomputed != summary:
            errors.append(f"{name}: saved summary differs from recomputation")
        else:
            recomputed_conditions.append(name)
        rows_by_condition[name] = rows

        adapter_path = condition.get("adapter_path")
        if expected_adapter:
            weight_path = Path(str(adapter_path)) / "adapter_model.safetensors"
            if not weight_path.is_file():
                errors.append(f"{name}: adapter weights missing")
            elif _sha256(weight_path) != condition.get("adapter_weights_sha256"):
                errors.append(f"{name}: adapter weight hash mismatch")
        elif adapter_path is not None:
            errors.append("original: adapter path must be null")

    if len(set(scoring_hashes)) != 1:
        errors.append("conditions disagree on candidate/prompt/token ordering")
    paired_ok = False
    if len(rows_by_condition) == len(EXPECTED_CONDITIONS):
        try:
            recomputed_paired = compare_reporter_correlations(
                rows_by_condition, bootstrap_samples=4000, bootstrap_seed=0
            )
            paired_ok = recomputed_paired == artifact.get(
                "paired_correlation_differences"
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"paired recomputation failed: {exc}")
        if not paired_ok:
            errors.append("saved paired differences differ from recomputation")

    return {
        "schema_version": 1,
        "artifact": str(path.resolve()),
        "artifact_sha256": _sha256(path),
        "status": "fail" if errors else "pass",
        "errors": errors,
        "details": {
            "fixed_checks": fixed_checks,
            "conditions_recomputed": recomputed_conditions,
            "paired_recomputed": paired_ok,
            "bootstrap_samples": 4000,
            "bootstrap_seed": 0,
            "n_conditions": len(conditions),
            "n_candidate_rows": sum(
                len(condition.get("candidate_rows", []))
                for condition in conditions.values()
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    report = validate(args.artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
