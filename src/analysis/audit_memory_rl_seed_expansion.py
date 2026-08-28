"""Fail-closed audit for formal RL-QA optimization-seed expansion runs.

This audit is deliberately training-only.  It accepts only explicitly named
seed-0 and expansion run directories, never discovers evaluation directories,
and has no OOD input option.  The existing formal run validator supplies the
artifact-level checks; this module adds campaign-specific invariants that are
not part of the generic validator, most importantly byte-identical splits and
strict first-maximum checkpoint selection on ID QA.

Example::

    python src/analysis/audit_memory_rl_seed_expansion.py \
      --seed0-run path/to/formal_rl-qa_..._s0_... \
      --lock path/to/stage_b1_rl_qa_lock.json \
      --run 1=path/to/formal_rl-qa_..._s1_... \
      --run 2=path/to/formal_rl-qa_..._s2_... \
      --out path/to/new_seed_expansion_audit.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence


SRC = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.validate_memory_rl_run import validate_run  # noqa: E402
from memory_rl.training_diagnostics import summarize_selector_training  # noqa: E402


EXPECTED_SEEDS = (1, 2)
EXPECTED_TRAIN_RECORDS = 300
EXPECTED_ROLLOUT_RECORDS = 2400
EXPECTED_VALIDATION_STEPS = (100, 200, 300)
EXPECTED_VALIDATION_EPISODES = 45
EXPECTED_GROUP_SIZE = 8
EXPECTED_MODE = "rl-qa"
EXPECTED_REPORTER_CONFIG = {
    "reporter_correlation_bootstrap_samples": 4000,
    "reporter_correlation_bootstrap_seed": 0,
    "reporter_correlation_diagnostics": True,
    "reporter_correlation_scope": "fixed ID validation candidates",
    "reporter_correlations_used_for_selection": False,
    "reporter_correlation_artifact": "reporter_correlations.jsonl",
}
REPORTER_KEYS = frozenset(EXPECTED_REPORTER_CONFIG)
REQUIRED_RUN_FILES = (
    "run_config.json",
    "split_manifest.json",
    "metrics.jsonl",
    "rollouts.jsonl",
    "summary.json",
    "best_checkpoint.json",
    "dropout_audit.json",
    "reporter_correlations.jsonl",
)

Validator = Callable[..., dict[str, Any]]


def _error(code: str, message: str, path: Path | str | None = None) -> dict[str, str]:
    result = {"code": code, "message": message}
    if path is not None:
        result["path"] = str(path)
    return result


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token!r}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def _nonfinite_locations(value: Any, prefix: str = "$") -> list[str]:
    locations: list[str] = []
    if isinstance(value, bool) or value is None:
        return locations
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    if isinstance(value, Mapping):
        for key, child in value.items():
            locations.extend(_nonfinite_locations(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            locations.extend(_nonfinite_locations(child, f"{prefix}[{index}]"))
    return locations


def _load_json(path: Path, errors: list[dict[str, str]]) -> dict[str, Any] | None:
    try:
        value = _loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(_error("invalid_json", str(exc), path))
        return None
    if not isinstance(value, dict):
        errors.append(_error("invalid_json_type", "top-level JSON must be an object", path))
        return None
    nonfinite = _nonfinite_locations(value)
    if nonfinite:
        errors.append(
            _error(
                "nonfinite_value",
                f"non-finite numeric values at {nonfinite[:5]}",
                path,
            )
        )
        return None
    return value


def _load_jsonl(path: Path, errors: list[dict[str, str]]) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    errors.append(
                        _error("blank_jsonl_line", f"blank line {line_number}", path)
                    )
                    continue
                try:
                    row = _loads_strict(line)
                except (json.JSONDecodeError, ValueError) as exc:
                    errors.append(
                        _error(
                            "invalid_jsonl",
                            f"line {line_number}: {exc}",
                            path,
                        )
                    )
                    continue
                if not isinstance(row, dict):
                    errors.append(
                        _error(
                            "invalid_jsonl_type",
                            f"line {line_number} is not an object",
                            path,
                        )
                    )
                    continue
                nonfinite = _nonfinite_locations(row)
                if nonfinite:
                    errors.append(
                        _error(
                            "nonfinite_value",
                            f"line {line_number}: non-finite values at {nonfinite[:5]}",
                            path,
                        )
                    )
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        errors.append(_error("invalid_jsonl", str(exc), path))
        return None
    if not rows:
        errors.append(_error("empty_jsonl", "JSONL artifact must not be empty", path))
        return None
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_path(raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
        )
    return left == right


def _check_lock_against_seed0(
    lock: Mapping[str, Any],
    seed0_config: Mapping[str, Any],
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not lock:
        return [_error("empty_lock", "lock must contain at least one field")]
    if lock.get("seed") != 0:
        errors.append(_error("seed0_lock_seed", "seed-0 lock must record seed=0"))
    for key, expected in lock.items():
        if key not in seed0_config:
            errors.append(
                _error("seed0_lock_field_missing", f"seed-0 config lacks lock field {key!r}")
            )
        elif seed0_config[key] != expected:
            errors.append(
                _error(
                    "seed0_lock_mismatch",
                    f"seed-0 config field {key!r} differs from lock",
                )
            )
    return errors


def _check_config_parity(
    *,
    seed: int,
    run_dir: Path,
    config: Mapping[str, Any],
    seed0_config: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    errors: list[dict[str, str]] = []
    lock_checks: dict[str, Any] = {}
    for key, locked in lock.items():
        expected = seed if key == "seed" else locked
        present = key in config
        actual = config.get(key)
        passed = present and actual == expected
        lock_checks[key] = {
            "expected": expected,
            "actual": actual,
            "present": present,
            "passed": passed,
        }
        if not passed:
            errors.append(
                _error(
                    "lock_field_mismatch",
                    f"seed {seed} field {key!r}: expected {expected!r}, got {actual!r}",
                    run_dir / "run_config.json",
                )
            )

    for key, expected in seed0_config.items():
        if key in {"seed", "out_dir"}:
            continue
        if key not in config:
            errors.append(
                _error(
                    "config_field_missing",
                    f"seed {seed} config lacks seed-0 field {key!r}",
                    run_dir / "run_config.json",
                )
            )
        elif config[key] != expected:
            errors.append(
                _error(
                    "seed0_config_mismatch",
                    f"seed {seed} config field {key!r} differs from seed 0",
                    run_dir / "run_config.json",
                )
            )

    allowed_keys = set(seed0_config) | REPORTER_KEYS
    extras = sorted(set(config) - allowed_keys)
    if extras:
        errors.append(
            _error(
                "unexpected_config_fields",
                f"seed {seed} has fields absent from seed 0 and not diagnostic-only: {extras}",
                run_dir / "run_config.json",
            )
        )

    if config.get("seed") != seed:
        errors.append(
            _error("optimization_seed_mismatch", f"expected optimization seed {seed}")
        )
    declared_out = _declared_path(config.get("out_dir"))
    if declared_out != run_dir.resolve():
        errors.append(
            _error(
                "out_dir_mismatch",
                f"run_config out_dir resolves to {declared_out}, expected {run_dir.resolve()}",
                run_dir / "run_config.json",
            )
        )

    for key, expected in EXPECTED_REPORTER_CONFIG.items():
        if config.get(key, object()) != expected:
            errors.append(
                _error(
                    "reporter_config_mismatch",
                    f"{key} must be {expected!r}",
                    run_dir / "run_config.json",
                )
            )

    return errors, lock_checks


def _check_constraints(
    run_dir: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    best: Mapping[str, Any],
    dropout: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, bool]]:
    expected = {
        "mode_rl_qa": config.get("mode") == EXPECTED_MODE,
        "formal_model": config.get("model") == "Qwen/Qwen2.5-7B-Instruct",
        "teacher_matches_policy": config.get("teacher_matches_policy_reference") is True,
        "teacher_fixed": config.get("fixed_workspace_teacher") is True,
        "teacher_mismatch_override_false": config.get("teacher_mismatch_override") is False,
        "cli_teacher_mismatch_false": config.get("allow_teacher_mismatch") is False,
        "probe_hidden": config.get("probe_visible_to_policy") is False,
        "recall_adapter_disabled": config.get("frozen_recall_adapter_disabled") is True,
        "workspace_not_optimized": config.get("workspace_reward_used_for_optimization") is False,
        "lambda_qa_one": _same_number(config.get("lambda_qa"), 1.0),
        "lambda_w_zero": _same_number(config.get("lambda_w"), 0.0),
        "constrained_yes_no": config.get("constrained_yes_no_rollouts") is True,
        "lora_dropout_zero": _same_number(config.get("lora_dropout"), 0.0),
        "dropout_postcondition": dropout.get("postcondition_satisfied") is True,
        "dropout_remaining_zero": dropout.get("remaining_nonzero") == [],
        "reporter_not_selection": config.get("reporter_correlations_used_for_selection") is False,
    }
    errors = [
        _error("protocol_constraint_failed", name, run_dir / "run_config.json")
        for name, passed in expected.items()
        if not passed
    ]

    best_path = _declared_path(best.get("path"))
    final_path = _declared_path(summary.get("final_adapter"))
    adapter_paths = {"best": best_path, "final": final_path}
    for label, path in adapter_paths.items():
        if path is None or not _inside(path, run_dir):
            errors.append(
                _error(
                    "adapter_path_outside_run",
                    f"{label} adapter must resolve inside its run directory",
                    str(path),
                )
            )
            continue
        for name in ("training_state.json", "adapter_config.json"):
            if not (path / name).is_file():
                errors.append(
                    _error("adapter_artifact_missing", f"{label} adapter lacks {name}", path)
                )
        if not any(
            (path / name).is_file() and (path / name).stat().st_size > 0
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            errors.append(
                _error("adapter_weights_missing", f"{label} adapter lacks non-empty weights", path)
            )
    expected["adapter_paths_inside_run"] = not any(
        error["code"].startswith("adapter_") for error in errors
    )
    return errors, expected


def _check_training_records(
    run_dir: Path,
    config: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    best: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    errors: list[dict[str, str]] = []
    baselines = [row for row in metrics if row.get("event") == "baseline"]
    train = [row for row in metrics if row.get("event") == "train"]
    validations = [row for row in metrics if row.get("event") == "validation"]
    unknown_events = [row.get("event") for row in metrics if row.get("event") not in {
        "baseline", "train", "validation"
    }]
    train_steps = [row.get("step") for row in train]
    validation_steps = [row.get("step") for row in validations]
    counts = {
        "metric_records": len(metrics),
        "baseline_records": len(baselines),
        "train_records": len(train),
        "validation_records": len(validations),
        "validation_steps": validation_steps,
        "rollout_records": len(rollouts),
        "summary_steps_completed": summary.get("steps_completed"),
    }
    if len(metrics) != 1 + EXPECTED_TRAIN_RECORDS + len(EXPECTED_VALIDATION_STEPS):
        errors.append(_error("metric_record_count", f"expected 304 metric records, got {len(metrics)}"))
    if len(baselines) != 1 or baselines[0].get("step") != 0:
        errors.append(_error("baseline_contract", "expected exactly one step-0 baseline"))
    if len(train) != EXPECTED_TRAIN_RECORDS:
        errors.append(
            _error("train_record_count", f"expected 300 train records, got {len(train)}")
        )
    if train_steps != list(range(1, EXPECTED_TRAIN_RECORDS + 1)):
        errors.append(_error("train_step_sequence", "train steps must be exactly 1..300"))
    if tuple(validation_steps) != EXPECTED_VALIDATION_STEPS:
        errors.append(
            _error(
                "validation_schedule",
                f"validation steps must be {EXPECTED_VALIDATION_STEPS}, got {validation_steps}",
            )
        )
    if any(row.get("episodes") != EXPECTED_VALIDATION_EPISODES for row in validations):
        errors.append(
            _error("validation_episode_count", "every validation must cover all 45 ID episodes")
        )
    if unknown_events:
        errors.append(_error("unknown_metric_events", f"unknown events: {unknown_events[:5]}"))
    if summary.get("steps_completed") != EXPECTED_TRAIN_RECORDS:
        errors.append(_error("steps_completed", "summary must record 300 completed steps"))

    if len(rollouts) != EXPECTED_ROLLOUT_RECORDS:
        errors.append(
            _error(
                "rollout_record_count",
                f"expected 2400 rollouts, got {len(rollouts)}",
                run_dir / "rollouts.jsonl",
            )
        )
    rollout_steps = Counter(row.get("step") for row in rollouts)
    expected_step_counts = {step: EXPECTED_GROUP_SIZE for step in range(1, 301)}
    if dict(rollout_steps) != expected_step_counts:
        errors.append(
            _error(
                "rollout_step_groups",
                "each train step must have exactly eight rollout rows",
                run_dir / "rollouts.jsonl",
            )
        )

    checkpoint: dict[str, Any] = {
        "selection_metric": "ID validation QA accuracy",
        "eligible_steps": list(EXPECTED_VALIDATION_STEPS),
        "tie_break": "strict first maximum (earliest step)",
        "observed_best_step": best.get("step"),
        "observed_best_metric": best.get("metric"),
    }
    if validations and all(
        isinstance(row.get("qa_accuracy"), (int, float))
        and not isinstance(row.get("qa_accuracy"), bool)
        and math.isfinite(float(row["qa_accuracy"]))
        for row in validations
    ):
        maximum = max(float(row["qa_accuracy"]) for row in validations)
        first = next(row for row in validations if float(row["qa_accuracy"]) == maximum)
        expected_step = first["step"]
        checkpoint.update({"expected_best_step": expected_step, "expected_best_metric": maximum})
        if best.get("step") != expected_step or not _same_number(best.get("metric"), maximum):
            errors.append(
                _error(
                    "first_maximum_checkpoint_mismatch",
                    f"expected first maximum at step {expected_step} with QA {maximum}, "
                    f"got step {best.get('step')} metric {best.get('metric')}",
                    run_dir / "best_checkpoint.json",
                )
            )
        if not _same_number(summary.get("best_validation_metric"), maximum):
            errors.append(
                _error("summary_best_metric_mismatch", "summary does not record maximum ID QA")
            )
        checkpoint["passed"] = not any(
            error["code"] in {
                "first_maximum_checkpoint_mismatch", "summary_best_metric_mismatch"
            }
            for error in errors
        )
    else:
        errors.append(_error("validation_qa_invalid", "validation QA values must be finite"))
        checkpoint["passed"] = False

    diagnostic_every = config.get("diagnostic_every")
    diagnostics: dict[str, Any] = {}
    if train and isinstance(diagnostic_every, int) and not isinstance(diagnostic_every, bool):
        try:
            recomputed = summarize_selector_training(train, window_size=diagnostic_every)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(
                _error("diagnostics_recomputation_failed", str(exc), run_dir / "metrics.jsonl")
            )
            return errors, counts, {"checkpoint": checkpoint, "training": diagnostics}
        overall = recomputed["overall"]
        unique_sets = [int(row["number_unique_selected_sets"]) for row in train]
        reward_stds = [float(row["reward_std"]) for row in train]
        gradients = [float(row["grad_norm"]) for row in train]
        kls = [float(row["kl"]) for row in train]
        mixed_count = sum(row.get("mixed_QA_reward_group") is True for row in train)
        variance_count = sum(value > 0.0 for value in reward_stds)
        nonzero_gradients = sum(value > 0.0 for value in gradients)
        diagnostics = {
            "mixed_reward": {
                "groups": mixed_count,
                "fraction": mixed_count / len(train),
                "nonzero_reward_std_groups": variance_count,
                "mean_reward_std": statistics.fmean(reward_stds),
            },
            "diversity": {
                "median_unique_selected_sets": statistics.median(unique_sets),
                "mean_unique_selected_sets": statistics.fmean(unique_sets),
                "min_unique_selected_sets": min(unique_sets),
                "max_unique_selected_sets": max(unique_sets),
                "one_set_groups": sum(value == 1 for value in unique_sets),
            },
            "gradients": {
                "nonzero_groups": nonzero_gradients,
                "fraction": nonzero_gradients / len(train),
                "mean_norm": statistics.fmean(gradients),
                "max_norm": max(gradients),
            },
            "kl": {
                "mean": statistics.fmean(kls),
                "min": min(kls),
                "max": max(kls),
            },
            "selector_training_diagnostics": recomputed,
        }
        if mixed_count < 1 or variance_count < 1:
            errors.append(_error("no_mixed_reward_signal", "no mixed/nonzero-variance QA group"))
        if any(value < 1 or value > EXPECTED_GROUP_SIZE for value in unique_sets):
            errors.append(_error("diversity_out_of_range", "unique-set counts must lie in [1,8]"))
        if max(unique_sets) <= 1:
            errors.append(_error("selection_diversity_collapsed", "all groups have one exact set"))
        if mixed_count != variance_count:
            errors.append(
                _error(
                    "mixed_reward_variance_mismatch",
                    "RL-QA mixed flags must agree with nonzero reward variance",
                )
            )
        if nonzero_gradients < 1:
            errors.append(_error("no_nonzero_gradient", "all recorded gradients are zero"))
        if min(kls) < -1e-6:
            errors.append(_error("negative_kl", "KL must be non-negative within tolerance"))
    else:
        errors.append(_error("diagnostics_unavailable", "cannot recompute selector diagnostics"))

    return errors, counts, {"checkpoint": checkpoint, "training": diagnostics}


def _check_reporter_records(
    run_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    errors: list[dict[str, str]] = []
    observed = [(row.get("event"), row.get("step")) for row in records]
    expected = [("baseline", 0)] + [
        ("validation", step) for step in EXPECTED_VALIDATION_STEPS
    ]
    if observed != expected:
        errors.append(
            _error(
                "reporter_event_schedule",
                f"expected reporter events {expected}, got {observed}",
                run_dir / "reporter_correlations.jsonl",
            )
        )
    if any(row.get("scope") != "fixed_id_validation" for row in records):
        errors.append(
            _error(
                "reporter_scope",
                "all reporter diagnostics must use fixed_id_validation scope",
                run_dir / "reporter_correlations.jsonl",
            )
        )
    return errors, {
        "records": len(records),
        "events": observed,
        "used_for_selection": False,
        "selection_metric": "qa_accuracy",
    }


def _audit_one_run(
    *,
    seed: int,
    run_dir: Path,
    seed0_config: Mapping[str, Any],
    seed0_manifest_bytes: bytes,
    seed0_manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    validator: Validator,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    run_dir = run_dir.expanduser().resolve()
    result: dict[str, Any] = {
        "seed": seed,
        "run_dir": str(run_dir),
        "status": "fail",
        "errors": errors,
        "warnings": warnings,
    }
    if not run_dir.is_dir():
        errors.append(_error("run_dir_missing", "run directory is missing", run_dir))
        return result
    if not run_dir.name.startswith("formal_rl-qa_"):
        errors.append(
            _error("unexpected_run_name", "run directory must be a formal_rl-qa artifact", run_dir)
        )
        return result

    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        errors.append(_error("required_artifacts_missing", f"missing artifacts: {missing}", run_dir))
        return result

    config = _load_json(run_dir / "run_config.json", errors)
    manifest = _load_json(run_dir / "split_manifest.json", errors)
    if config is None or manifest is None:
        return result
    parity_errors, lock_checks = _check_config_parity(
        seed=seed,
        run_dir=run_dir,
        config=config,
        seed0_config=seed0_config,
        lock=lock,
    )
    errors.extend(parity_errors)
    result["lock_fields"] = lock_checks

    try:
        manifest_bytes = (run_dir / "split_manifest.json").read_bytes()
    except OSError as exc:
        errors.append(_error("manifest_read_failed", str(exc), run_dir / "split_manifest.json"))
        return result
    manifest_identical = manifest_bytes == seed0_manifest_bytes
    result["manifest"] = {
        "byte_identical_to_seed0": manifest_identical,
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }
    if not manifest_identical:
        errors.append(
            _error(
                "manifest_not_byte_identical",
                "split_manifest.json differs byte-for-byte from seed 0",
                run_dir / "split_manifest.json",
            )
        )

    # Check every adapter-bearing declaration before the generic validator can
    # resolve or read it.  This keeps a malformed training artifact from making
    # the audit follow an external (including evaluation/OOD) path.
    pre_summary = _load_json(run_dir / "summary.json", errors)
    pre_best = _load_json(run_dir / "best_checkpoint.json", errors)
    if pre_summary is not None and pre_best is not None:
        declared_artifacts = {
            "best_checkpoint.path": pre_best.get("path"),
            "summary.best_checkpoint": pre_summary.get("best_checkpoint"),
            "summary.final_adapter": pre_summary.get("final_adapter"),
        }
        for label, raw_path in declared_artifacts.items():
            resolved = _declared_path(raw_path)
            if resolved is None or not _inside(resolved, run_dir):
                errors.append(
                    _error(
                        "declared_artifact_outside_run",
                        f"{label} must resolve inside the explicit training run",
                        str(resolved),
                    )
                )

    # Do not let the generic validator follow declared paths until local config
    # and split provenance have passed the explicit seed-0 parity precheck.
    if errors:
        result["formal_validator"] = {
            "status": "not_run",
            "reason": "local parity/safety precheck failed",
        }
        return result

    expected_lock = dict(lock)
    expected_lock["seed"] = seed
    formal = validator(
        run_dir,
        profile="formal",
        expected_manifest_sha256=seed0_manifest.get("manifest_sha256"),
        expected_config=expected_lock,
    )
    result["formal_validator"] = formal
    if formal.get("status") != "pass":
        errors.append(
            _error("formal_validator_failed", "generic formal run validator did not pass", run_dir)
        )
        return result
    warnings.extend(formal.get("warnings", []))

    metrics = _load_jsonl(run_dir / "metrics.jsonl", errors)
    rollouts = _load_jsonl(run_dir / "rollouts.jsonl", errors)
    reporter = _load_jsonl(run_dir / "reporter_correlations.jsonl", errors)
    summary = pre_summary
    best = pre_best
    dropout = _load_json(run_dir / "dropout_audit.json", errors)
    if any(value is None for value in (metrics, rollouts, reporter, summary, best, dropout)):
        return result
    assert metrics is not None and rollouts is not None and reporter is not None
    assert summary is not None and best is not None and dropout is not None

    constraint_errors, constraints = _check_constraints(
        run_dir, config, summary, best, dropout
    )
    errors.extend(constraint_errors)
    record_errors, counts, diagnostics = _check_training_records(
        run_dir, config, metrics, rollouts, summary, best
    )
    errors.extend(record_errors)
    reporter_errors, reporter_summary = _check_reporter_records(run_dir, reporter)
    errors.extend(reporter_errors)

    result.update(
        {
            "constraints": constraints,
            "counts": counts,
            "checkpoint_selection": diagnostics["checkpoint"],
            "training_diagnostics": diagnostics["training"],
            "reporter_diagnostics": reporter_summary,
            "artifact_sha256": {
                name: _sha256(run_dir / name) for name in REQUIRED_RUN_FILES
            },
        }
    )
    result["status"] = "pass" if not errors else "fail"
    return result


def audit_seed_expansion(
    *,
    seed0_run: str | Path,
    lock_path: str | Path,
    runs: Mapping[int, str | Path],
    validator: Validator | None = None,
) -> dict[str, Any]:
    """Return a strict-JSON audit without writing files or discovering runs."""
    validator = validator or validate_run
    global_errors: list[dict[str, str]] = []
    seed0_run = Path(seed0_run).expanduser().resolve()
    lock_path = Path(lock_path).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit": "memory_rl_seed_expansion_training",
        "status": "fail",
        "scope": {
            "training_only": True,
            "ood_inputs_supported": False,
            "ood_artifacts_read": False,
            "expected_optimization_seeds": list(EXPECTED_SEEDS),
        },
        "seed0_run": str(seed0_run),
        "lock_path": str(lock_path),
        "errors": global_errors,
        "runs": {},
    }
    if set(runs) != set(EXPECTED_SEEDS):
        global_errors.append(
            _error(
                "seed_set_mismatch",
                f"runs must contain exactly seeds {EXPECTED_SEEDS}, got {sorted(runs)}",
            )
        )
    if not seed0_run.is_dir():
        global_errors.append(_error("seed0_run_missing", "seed-0 run is missing", seed0_run))
    if not lock_path.is_file():
        global_errors.append(_error("lock_missing", "lock file is missing", lock_path))
    if global_errors:
        return report

    seed0_config = _load_json(seed0_run / "run_config.json", global_errors)
    seed0_manifest = _load_json(seed0_run / "split_manifest.json", global_errors)
    lock = _load_json(lock_path, global_errors)
    try:
        seed0_manifest_bytes = (seed0_run / "split_manifest.json").read_bytes()
    except OSError as exc:
        global_errors.append(
            _error("seed0_manifest_read_failed", str(exc), seed0_run / "split_manifest.json")
        )
        seed0_manifest_bytes = b""
    if seed0_config is None or seed0_manifest is None or lock is None or global_errors:
        return report
    global_errors.extend(_check_lock_against_seed0(lock, seed0_config))
    if global_errors:
        return report

    report["seed0_provenance"] = {
        "run_config_sha256": _sha256(seed0_run / "run_config.json"),
        "split_manifest_file_sha256": hashlib.sha256(seed0_manifest_bytes).hexdigest(),
        "split_manifest_sha256": seed0_manifest.get("manifest_sha256"),
        "lock_sha256": _sha256(lock_path),
        "lock_field_count": len(lock),
    }
    for seed in EXPECTED_SEEDS:
        if seed not in runs:
            continue
        try:
            report["runs"][str(seed)] = _audit_one_run(
                seed=seed,
                run_dir=Path(runs[seed]),
                seed0_config=seed0_config,
                seed0_manifest_bytes=seed0_manifest_bytes,
                seed0_manifest=seed0_manifest,
                lock=lock,
                validator=validator,
            )
        except Exception as exc:  # fail closed on unexpected artifact races/schema drift
            report["runs"][str(seed)] = {
                "seed": seed,
                "run_dir": str(Path(runs[seed]).expanduser().resolve()),
                "status": "fail",
                "errors": [
                    _error("unexpected_audit_failure", f"{type(exc).__name__}: {exc}")
                ],
                "warnings": [],
            }
    report["status"] = (
        "pass"
        if not global_errors
        and set(report["runs"]) == {"1", "2"}
        and all(value.get("status") == "pass" for value in report["runs"].values())
        else "fail"
    )
    return report


def write_report_exclusive(path: str | Path, report: Mapping[str, Any]) -> None:
    """Write one strict JSON report and never replace an existing path."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


def _parse_runs(values: Sequence[str]) -> tuple[dict[int, Path], list[dict[str, str]]]:
    runs: dict[int, Path] = {}
    errors: list[dict[str, str]] = []
    for value in values:
        if "=" not in value:
            errors.append(_error("invalid_run_argument", f"expected SEED=PATH, got {value!r}"))
            continue
        raw_seed, raw_path = value.split("=", 1)
        try:
            seed = int(raw_seed)
        except ValueError:
            errors.append(_error("invalid_run_seed", f"invalid seed {raw_seed!r}"))
            continue
        if seed in runs:
            errors.append(_error("duplicate_run_seed", f"seed {seed} was supplied twice"))
            continue
        if not raw_path:
            errors.append(_error("invalid_run_path", f"seed {seed} has an empty path"))
            continue
        runs[seed] = Path(raw_path)
    return runs, errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-run", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"error: refusing to overwrite existing audit report: {args.out}", file=sys.stderr)
        return 2
    runs, argument_errors = _parse_runs(args.run)
    output_path = args.out.expanduser().resolve()
    protected_roots = [args.seed0_run.expanduser().resolve()] + [
        path.expanduser().resolve() for path in runs.values()
    ]
    if any(_inside(output_path, root) for root in protected_roots):
        print(
            "error: audit output must be outside all training run directories",
            file=sys.stderr,
        )
        return 2
    if argument_errors:
        report: dict[str, Any] = {
            "schema_version": 1,
            "audit": "memory_rl_seed_expansion_training",
            "status": "fail",
            "scope": {
                "training_only": True,
                "ood_inputs_supported": False,
                "ood_artifacts_read": False,
            },
            "errors": argument_errors,
            "runs": {},
        }
    else:
        report = audit_seed_expansion(
            seed0_run=args.seed0_run,
            lock_path=args.lock,
            runs=runs,
        )
    try:
        write_report_exclusive(args.out, report)
    except FileExistsError:
        print(f"error: refusing to overwrite existing audit report: {args.out}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    sys.stdout.write(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
