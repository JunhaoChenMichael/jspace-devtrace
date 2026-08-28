"""Validate memory-RL training artifacts without loading a model.

The validator deliberately separates reproducibility/structure failures from
scientific health warnings.  It uses strict JSON parsing, recomputes immutable
input hashes, and cross-checks metrics, rollouts, summaries, and checkpoints.

Example::

    python src/analysis/validate_memory_rl_run.py RUN_DIR --profile smoke

The command prints one strict-JSON report and exits non-zero only when a hard
invariant fails.  Reward trends and collapse heuristics are warnings; campaign
gates remain the responsibility of ``memory_rl_gates.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from memory_rl.data import (  # noqa: E402
    ALLOWED_TRAIN_SOURCES,
    file_sha256,
    verify_split_manifest,
)
from memory_rl.reporter_correlations import (  # noqa: E402
    summarize_reporter_correlations,
    within_episode_utility_auc,
)
from memory_rl.training_diagnostics import summarize_selector_training  # noqa: E402


PROFILES = ("dry-run", "smoke", "formal")
MODES = ("sft-w", "rl-w", "rl-qa", "rl-hybrid")
SELECTOR_MODES = ("rl-qa", "rl-hybrid")
FORMAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_RELEASED_SPLIT = {"train": 175, "validation": 45}
_MISSING = object()


def _issue(code: str, message: str, path: str | None = None) -> dict[str, str]:
    value = {"code": code, "message": message}
    if path is not None:
        value["path"] = path
    return value


@dataclass
class _Audit:
    run_dir: Path
    profile: str
    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    validation_episode_ids: set[str] = field(default_factory=set)

    def error(self, code: str, message: str, path: str | Path | None = None) -> None:
        self.errors.append(_issue(code, message, str(path) if path is not None else None))

    def warn(self, code: str, message: str, path: str | Path | None = None) -> None:
        self.warnings.append(_issue(code, message, str(path) if path is not None else None))

    def report(self, mode: str | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_dir": str(self.run_dir),
            "profile": self.profile,
            "mode": mode,
            "status": "fail" if self.errors else "pass",
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _find_nonfinite(value: Any, prefix: str = "$") -> str | None:
    if isinstance(value, float) and not math.isfinite(value):
        return prefix
    if isinstance(value, Mapping):
        for key, child in value.items():
            found = _find_nonfinite(child, f"{prefix}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_nonfinite(child, f"{prefix}[{index}]")
            if found is not None:
                return found
    return None


def _strict_load_text(text: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = child
        return result

    value = json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=unique_object,
    )
    nonfinite = _find_nonfinite(value)
    if nonfinite is not None:
        raise ValueError(f"non-finite numeric value at {nonfinite}")
    return value


def _load_json(audit: _Audit, path: Path, *, required: bool = True) -> Any | None:
    if not path.is_file():
        if required:
            audit.error("missing_artifact", f"required artifact is missing: {path.name}", path)
        return None
    try:
        value = _strict_load_text(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        audit.error("invalid_json", f"strict JSON parse failed: {exc}", path)
        return None
    if not isinstance(value, dict):
        audit.error("invalid_json_type", "top-level JSON value must be an object", path)
        return None
    return value


def _load_jsonl(audit: _Audit, path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        audit.error("missing_artifact", f"required artifact is missing: {path.name}", path)
        return None
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        audit.error("invalid_jsonl", f"could not read JSONL: {exc}", path)
        return None
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            audit.error("blank_jsonl_line", "JSONL contains a blank line", f"{path}:{line_number}")
            continue
        try:
            value = _strict_load_text(line)
        except (json.JSONDecodeError, ValueError) as exc:
            audit.error(
                "invalid_jsonl",
                f"strict JSON parse failed on line {line_number}: {exc}",
                f"{path}:{line_number}",
            )
            continue
        if not isinstance(value, dict):
            audit.error(
                "invalid_jsonl_type",
                f"line {line_number} must be a JSON object",
                f"{path}:{line_number}",
            )
            continue
        records.append(value)
    if not records:
        audit.error("empty_jsonl", "JSONL contains no object records", path)
    return records


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _close(left: Any, right: Any, *, atol: float = 1e-6) -> bool:
    return _is_finite_number(left) and _is_finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=1e-5, abs_tol=atol
    )


def _require_finite(
    audit: _Audit,
    record: Mapping[str, Any],
    key: str,
    where: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float | None:
    value = record.get(key, _MISSING)
    if not _is_finite_number(value):
        audit.error("nonfinite_metric", f"{key} must be a finite number", where)
        return None
    number = float(value)
    if lower is not None and number < lower - 1e-6:
        audit.error("metric_out_of_range", f"{key}={number} is below {lower}", where)
    if upper is not None and number > upper + 1e-6:
        audit.error("metric_out_of_range", f"{key}={number} is above {upper}", where)
    return number


def _canonical_model(value: Any) -> str | None:
    return value.rstrip("/") if isinstance(value, str) and value else None


def _resolve_declared_path(raw: Any, run_dir: Path) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, run_dir / path]
    candidates.append(run_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _within_run(path: Path, run_dir: Path) -> bool:
    try:
        path.resolve().relative_to(run_dir.resolve())
        return True
    except ValueError:
        return False


def _validate_teacher(audit: _Audit, config: Mapping[str, Any]) -> None:
    model = _canonical_model(config.get("model"))
    teacher = _canonical_model(config.get("workspace_teacher_model"))
    declared_match = config.get("teacher_matches_policy_reference")
    override = config.get("teacher_mismatch_override")
    cli_override = config.get("allow_teacher_mismatch")

    if model is None or teacher is None:
        audit.error("teacher_identity_missing", "model and workspace_teacher_model are required")
        return
    computed_match = model == teacher
    if not isinstance(declared_match, bool) or declared_match != computed_match:
        audit.error(
            "teacher_match_inconsistent",
            "teacher_matches_policy_reference disagrees with declared model IDs",
        )
    if not isinstance(override, bool):
        audit.error("teacher_override_invalid", "teacher_mismatch_override must be boolean")
        override = False
    if isinstance(cli_override, bool) and cli_override != override:
        audit.error(
            "teacher_override_inconsistent",
            "allow_teacher_mismatch and teacher_mismatch_override disagree",
        )
    if not computed_match and not override:
        audit.error("teacher_mismatch_unapproved", "teacher mismatch lacks an explicit override")

    if audit.profile in ("dry-run", "formal") and (not computed_match or override):
        audit.error(
            "teacher_mismatch_forbidden",
            f"{audit.profile} artifacts require a matched teacher and no override",
        )
    elif not computed_match:
        audit.warn(
            "teacher_mismatch_smoke_only",
            "teacher mismatch is acceptable only as non-scientific smoke wiring",
        )

    if config.get("fixed_workspace_teacher") is not True:
        audit.error("teacher_not_fixed", "fixed_workspace_teacher must be true")
    if config.get("constrained_yes_no_rollouts") is not True:
        audit.error("unconstrained_actions", "constrained_yes_no_rollouts must be true")
    if audit.profile == "formal":
        if model != FORMAL_MODEL or teacher != FORMAL_MODEL:
            audit.error(
                "formal_model_mismatch",
                f"formal policy and teacher must both be {FORMAL_MODEL}",
            )


def _validate_config(
    audit: _Audit,
    config: Mapping[str, Any],
    expected_config: Mapping[str, Any] | None,
) -> str | None:
    mode = config.get("mode")
    if mode not in MODES:
        audit.error("invalid_mode", f"mode must be one of {MODES}")
        return None
    dry_run = config.get("dry_run")
    if not isinstance(dry_run, bool) or dry_run != (audit.profile == "dry-run"):
        audit.error("profile_mismatch", "run_config.dry_run disagrees with --profile")
    _validate_teacher(audit, config)

    if mode in SELECTOR_MODES:
        if config.get("probe_visible_to_policy") is not False:
            audit.error(
                "probe_isolation_missing",
                "selector policy must record probe_visible_to_policy=false",
            )
        if config.get("frozen_recall_adapter_disabled") is not True:
            audit.error(
                "recall_not_frozen",
                "selector reward must use an adapter-disabled frozen recall model",
            )
        if mode == "rl-qa" and config.get("workspace_reward_used_for_optimization") is not False:
            audit.error(
                "rl_qa_workspace_reward_leak",
                "RL-QA must not use workspace reward for optimization",
            )
        diagnostic_every = config.get("diagnostic_every")
        if not _is_int(diagnostic_every) or diagnostic_every < 1:
            audit.error("diagnostic_interval_invalid", "selector diagnostic_every must be >= 1")

    correlation_diagnostics = config.get("reporter_correlation_diagnostics")
    if correlation_diagnostics is not None and not isinstance(
        correlation_diagnostics, bool
    ):
        audit.error(
            "reporter_correlation_config_invalid",
            "reporter_correlation_diagnostics must be boolean when present",
        )
    if correlation_diagnostics is True:
        if mode not in SELECTOR_MODES:
            audit.error(
                "reporter_correlation_mode_invalid",
                "reporter correlation diagnostics are defined only for selector modes",
            )
        if config.get("reporter_correlation_scope") != "fixed ID validation candidates":
            audit.error(
                "reporter_correlation_scope_invalid",
                "reporter correlations must be scoped to fixed ID validation candidates",
            )
        if config.get("reporter_correlations_used_for_selection") is not False:
            audit.error(
                "reporter_correlation_selection_leak",
                "reporter correlations must not enter checkpoint or coefficient selection",
            )
        if config.get("reporter_correlation_artifact") != "reporter_correlations.jsonl":
            audit.error(
                "reporter_correlation_artifact_invalid",
                "reporter correlation artifact must be reporter_correlations.jsonl",
            )
        samples = config.get("reporter_correlation_bootstrap_samples")
        seed = config.get("reporter_correlation_bootstrap_seed")
        if not _is_int(samples) or samples < 0:
            audit.error(
                "reporter_correlation_bootstrap_invalid",
                "reporter correlation bootstrap samples must be a non-negative integer",
            )
        if not _is_int(seed):
            audit.error(
                "reporter_correlation_bootstrap_invalid",
                "reporter correlation bootstrap seed must be an integer",
            )

    if config.get("split_seed") != 0:
        audit.error("split_seed_not_locked", "campaign split_seed must be 0")
    fraction = config.get("val_fraction")
    if not _is_finite_number(fraction) or not _close(fraction, 0.2):
        audit.error("validation_fraction_not_locked", "campaign val_fraction must be 0.2")

    if audit.profile in ("dry-run", "formal"):
        for key in ("limit_train_episodes", "limit_validation_episodes"):
            if config.get(key) != 0:
                audit.error("episode_limit_forbidden", f"{key} must be 0 for {audit.profile}")
    if audit.profile == "formal" and config.get("val_eval_episodes") != 0:
        audit.error("formal_validation_limited", "formal val_eval_episodes must be 0")

    if expected_config is not None:
        for key, expected in expected_config.items():
            actual = config.get(key, _MISSING)
            if actual is _MISSING or actual != expected:
                audit.error(
                    "locked_config_mismatch",
                    f"locked config {key!r}: expected {expected!r}, got {actual!r}",
                )
    elif audit.profile == "formal":
        audit.warn(
            "formal_lock_not_supplied",
            "no --expected-config lock was supplied; hyperparameter locking was not checked",
        )
    return str(mode)


def _validate_manifest(
    audit: _Audit,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_manifest_sha256: str | None,
) -> None:
    manifest_path = audit.run_dir / "split_manifest.json"
    if not verify_split_manifest(manifest):
        audit.error("manifest_self_hash_invalid", "split manifest self-hash is invalid", manifest_path)
    digest = manifest.get("manifest_sha256")
    audit.details["manifest_sha256"] = digest
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        audit.error(
            "manifest_mismatch",
            f"manifest SHA differs from expected {expected_manifest_sha256}",
            manifest_path,
        )
    if manifest.get("schema_version") != 1:
        audit.error("manifest_schema_invalid", "split manifest schema_version must be 1")
    if manifest.get("seed") != config.get("split_seed"):
        audit.error("manifest_seed_mismatch", "manifest seed and run_config split_seed differ")
    if not _close(manifest.get("validation_fraction"), config.get("val_fraction")):
        audit.error(
            "manifest_fraction_mismatch",
            "manifest validation_fraction and run_config val_fraction differ",
        )

    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        audit.error("manifest_sources_invalid", "manifest sources must be a non-empty object")
        return
    source_names = set(sources)
    unknown = source_names - set(ALLOWED_TRAIN_SOURCES)
    if unknown:
        audit.error("forbidden_training_source", f"manifest contains forbidden sources: {sorted(unknown)}")
    if audit.profile == "formal" and source_names != set(ALLOWED_TRAIN_SOURCES):
        audit.error(
            "formal_sources_incomplete",
            f"formal sources must be exactly {sorted(ALLOWED_TRAIN_SOURCES)}",
        )

    specs = config.get("train_spec")
    spec_map: dict[str, Mapping[str, Any]] = {}
    if not isinstance(specs, list) or not specs:
        audit.error("train_spec_invalid", "run_config train_spec must be a non-empty list")
    else:
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict) or not isinstance(spec.get("source"), str):
                audit.error("train_spec_invalid", f"train_spec[{index}] must name a source")
                continue
            source = spec["source"]
            if source in spec_map:
                audit.error("duplicate_train_spec", f"train_spec repeats source {source!r}")
            spec_map[source] = spec
        if set(spec_map) != source_names:
            audit.error("train_spec_manifest_mismatch", "train_spec sources differ from manifest sources")

    global_train: set[str] = set()
    global_validation: set[str] = set()
    train_count = 0
    validation_count = 0
    for source, raw_entry in sources.items():
        where = f"{manifest_path}:sources.{source}"
        if not isinstance(raw_entry, dict):
            audit.error("manifest_source_invalid", f"source {source!r} entry must be an object", where)
            continue
        entry = raw_entry
        train_ids = entry.get("train_episode_ids")
        validation_ids = entry.get("validation_episode_ids")
        if not isinstance(train_ids, list) or not all(isinstance(x, str) for x in train_ids):
            audit.error("manifest_ids_invalid", "train_episode_ids must be a string list", where)
            train_ids = []
        if not isinstance(validation_ids, list) or not all(
            isinstance(x, str) for x in validation_ids
        ):
            audit.error("manifest_ids_invalid", "validation_episode_ids must be a string list", where)
            validation_ids = []
        if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
            audit.error("duplicate_split_id", f"source {source!r} contains duplicate episode IDs", where)
        overlap = set(train_ids) & set(validation_ids)
        if overlap:
            audit.error("split_overlap", f"source {source!r} train/validation IDs overlap", where)
        if global_train & set(train_ids) or global_validation & set(validation_ids):
            audit.error("duplicate_split_id", "episode IDs repeat across sources", where)
        global_train.update(train_ids)
        global_validation.update(validation_ids)
        train_count += len(train_ids)
        validation_count += len(validation_ids)

        spec = spec_map.get(source)
        for kind in ("battery", "results"):
            raw_path = entry.get(f"{kind}_path")
            expected_hash = entry.get(f"{kind}_sha256")
            resolved = _resolve_declared_path(raw_path, audit.run_dir)
            if resolved is None or not resolved.is_file():
                audit.error("source_file_missing", f"{source} {kind} file is unavailable", where)
                continue
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                audit.error("source_hash_invalid", f"{source} {kind} SHA-256 is malformed", where)
            else:
                actual_hash = file_sha256(resolved)
                if actual_hash != expected_hash:
                    audit.error(
                        "source_hash_mismatch",
                        f"{source} {kind} SHA-256 no longer matches the manifest",
                        resolved,
                    )
            if spec is not None:
                spec_path = _resolve_declared_path(spec.get(f"{kind}_path"), audit.run_dir)
                if spec_path is None or spec_path.resolve() != resolved.resolve():
                    audit.error(
                        "source_path_mismatch",
                        f"train_spec and manifest disagree on {source} {kind} path",
                        where,
                    )

    if global_train & global_validation:
        audit.error("split_overlap", "global train and validation episode IDs overlap")
    audit.validation_episode_ids = set(global_validation)
    if manifest.get("train_episode_count") != train_count:
        audit.error("split_count_mismatch", "manifest train_episode_count is inconsistent")
    if manifest.get("validation_episode_count") != validation_count:
        audit.error("split_count_mismatch", "manifest validation_episode_count is inconsistent")

    audit.details["split"] = {
        "sources": sorted(source_names),
        "train_episode_count": train_count,
        "validation_episode_count": validation_count,
    }
    if source_names == set(ALLOWED_TRAIN_SOURCES):
        if train_count != EXPECTED_RELEASED_SPLIT["train"]:
            audit.error("released_split_count_mismatch", "released split must contain 175 train episodes")
        if validation_count != EXPECTED_RELEASED_SPLIT["validation"]:
            audit.error(
                "released_split_count_mismatch",
                "released split must contain 45 validation episodes",
            )

    for split_name, manifest_count in (
        ("train", train_count),
        ("validation", validation_count),
    ):
        limit = config.get(f"limit_{split_name}_episodes")
        actual = config.get(f"{split_name}_episode_count")
        if not _is_int(limit) or limit < 0:
            audit.error("episode_limit_invalid", f"limit_{split_name}_episodes must be >= 0")
            continue
        expected = min(limit, manifest_count) if limit else manifest_count
        if actual != expected:
            audit.error(
                "effective_split_count_mismatch",
                f"{split_name}_episode_count should be {expected}, got {actual!r}",
            )


def _validate_dry_artifacts(audit: _Audit, mode: str) -> None:
    forbidden = (
        "metrics.jsonl",
        "rollouts.jsonl",
        "dropout_audit.json",
        "best_checkpoint.json",
        "summary.json",
        "final_adapter",
    )
    for name in forbidden:
        path = audit.run_dir / name
        if path.exists():
            audit.error("unexpected_dry_run_artifact", f"dry-run must not create {name}", path)
    for path in audit.run_dir.glob("best-step-*"):
        audit.error("unexpected_dry_run_artifact", "dry-run created a best checkpoint", path)
    for key in ("resolved_model_name_or_path", "trainable_parameter_count", "total_parameter_count"):
        # Their presence would mean the nominal dry-run crossed the model-loading boundary.
        if key in _strict_optional_config(audit.run_dir / "run_config.json"):
            audit.error("dry_run_loaded_model", f"dry-run config unexpectedly contains {key}")
    audit.details["artifact_contract"] = {"kind": "dry-run", "mode": mode}


def _strict_optional_config(path: Path) -> Mapping[str, Any]:
    """Read an already-validated config for a tiny dry-run presence check."""
    try:
        value = _strict_load_text(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _validate_validation_record(
    audit: _Audit,
    config: Mapping[str, Any],
    record: Mapping[str, Any],
    mode: str,
    where: str,
) -> None:
    if mode in ("sft-w", "rl-w"):
        for key in ("auc", "within_episode_auc", "containment", "yes_rate", "selection_metric"):
            _require_finite(audit, record, key, where, lower=0.0, upper=1.0)
        if _is_finite_number(record.get("auc")) and not _close(
            record.get("selection_metric"), record.get("auc")
        ):
            audit.error("selection_metric_mismatch", "reporter selection_metric must equal auc", where)
    else:
        for key in ("qa_accuracy", "containment", "workspace_set_reward", "selection_metric"):
            _require_finite(audit, record, key, where, lower=0.0, upper=1.0)
        episodes = record.get("episodes")
        if not _is_int(episodes) or episodes < 1:
            audit.error("validation_episode_count_invalid", "selector validation episodes must be >= 1", where)
        if _is_finite_number(record.get("qa_accuracy")) and not _close(
            record.get("selection_metric"), record.get("qa_accuracy")
        ):
            audit.error(
                "selection_metric_mismatch", "selector selection_metric must equal qa_accuracy", where
            )
        if config.get("reporter_correlation_diagnostics") is True:
            for key in (
                "verbal_auc",
                "verbal_within_episode_auc",
                "verbal_yes_rate",
                "combined_reward",
            ):
                _require_finite(audit, record, key, where, lower=0.0, upper=1.0)
            correlations = record.get("reporter_correlations")
            if not isinstance(correlations, dict):
                audit.error(
                    "reporter_correlations_missing",
                    "selector validation must contain reporter_correlations",
                    where,
                )
            lambda_qa = config.get("lambda_qa")
            lambda_w = config.get("lambda_w")
            if (
                _is_finite_number(lambda_qa)
                and _is_finite_number(lambda_w)
                and _is_finite_number(record.get("qa_accuracy"))
                and _is_finite_number(record.get("workspace_set_reward"))
                and float(lambda_qa) + float(lambda_w) > 0
            ):
                expected_combined = (
                    float(lambda_qa) * float(record.get("qa_accuracy", 0.0))
                    + float(lambda_w) * float(record.get("workspace_set_reward", 0.0))
                ) / (float(lambda_qa) + float(lambda_w))
                if not _close(record.get("combined_reward"), expected_combined):
                    audit.error(
                        "combined_reward_mismatch",
                        "ID combined reward disagrees with locked coefficients",
                        where,
                    )


def _validate_train_record(
    audit: _Audit, record: Mapping[str, Any], mode: str, where: str
) -> None:
    _require_finite(audit, record, "loss", where)
    _require_finite(audit, record, "kl", where, lower=-1e-6)
    _require_finite(audit, record, "grad_norm", where, lower=0.0)
    if mode == "sft-w":
        for key in ("reward_mean", "reward_std"):
            if key not in record or record[key] is not None:
                audit.error("invalid_na_metric", f"SFT {key} must be JSON null", where)
        _require_finite(audit, record, "yes_rate", where, lower=0.0, upper=1.0)
    elif mode == "rl-w":
        _require_finite(audit, record, "reward_mean", where, lower=-1.0, upper=1.0)
        _require_finite(audit, record, "reward_std", where, lower=0.0, upper=1.0)
        _require_finite(audit, record, "yes_rate", where, lower=0.0, upper=1.0)
        _require_finite(audit, record, "workspace_percentile", where, lower=0.0, upper=1.0)
        if not isinstance(record.get("candidate_id"), str):
            audit.error("candidate_id_missing", "RL-W train record needs candidate_id", where)
    else:
        _require_finite(audit, record, "reward_mean", where, lower=0.0, upper=1.0)
        _require_finite(audit, record, "reward_std", where, lower=0.0, upper=1.0)
        _require_finite(audit, record, "qa_reward", where, lower=0.0, upper=1.0)
        _require_finite(audit, record, "workspace_reward", where, lower=0.0, upper=1.0)
        if record.get("yes_rate", _MISSING) is not None:
            audit.error("invalid_na_metric", "selector yes_rate must be JSON null", where)
        if not isinstance(record.get("episode_id"), str):
            audit.error("episode_id_missing", "selector train record needs episode_id", where)
        candidate_count = record.get("candidate_count")
        if not _is_int(candidate_count) or candidate_count < 2:
            audit.error("candidate_count_invalid", "selector candidate_count must be >= 2", where)
            candidate_count = 0
        unique_sets = record.get("number_unique_selected_sets")
        if not _is_int(unique_sets) or unique_sets < 1:
            audit.error(
                "selection_diversity_invalid",
                "number_unique_selected_sets must be a positive integer",
                where,
            )
        _require_finite(
            audit, record, "fraction_unique_selected_sets", where, lower=0.0, upper=1.0
        )
        for key in ("mixed_QA_reward_group", "mixed_containment_group"):
            if not isinstance(record.get(key), bool):
                audit.error("selector_group_flag_invalid", f"{key} must be boolean", where)
        _require_finite(audit, record, "containment_rate", where, lower=0.0, upper=1.0)
        _require_finite(audit, record, "policy_set_entropy", where, lower=-1e-6)
        _require_finite(
            audit,
            record,
            "normalized_policy_set_entropy",
            where,
            lower=-1e-6,
            upper=1.0 + 1e-6,
        )
        yes_probabilities = record.get("yes_probabilities")
        if (
            not isinstance(yes_probabilities, list)
            or len(yes_probabilities) != candidate_count
            or not all(
                _is_finite_number(value) and 0.0 <= float(value) <= 1.0
                for value in yes_probabilities
            )
        ):
            audit.error(
                "yes_probability_distribution_invalid",
                "yes_probabilities must align with candidates and lie in [0,1]",
                where,
            )


def _validate_metrics(
    audit: _Audit,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    mode: str,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    path = audit.run_dir / "metrics.jsonl"
    steps_completed = summary.get("steps_completed")
    max_steps = config.get("max_steps")
    if not _is_int(steps_completed) or steps_completed < 1:
        audit.error("steps_completed_invalid", "summary.steps_completed must be >= 1")
        steps_completed = 0
    if not _is_int(max_steps) or max_steps < 1 or steps_completed > max_steps:
        audit.error("max_steps_inconsistent", "steps_completed must not exceed max_steps")

    previous_step = -1
    baselines: list[Mapping[str, Any]] = []
    validations: list[Mapping[str, Any]] = []
    train_by_step: dict[int, Mapping[str, Any]] = {}
    for index, record in enumerate(records, 1):
        where = f"{path}:{index}"
        event = record.get("event")
        step = record.get("step")
        if not _is_int(step) or step < 0:
            audit.error("metric_step_invalid", "metric step must be a non-negative integer", where)
            continue
        if step < previous_step:
            audit.error("metric_step_order", "metric steps must be non-decreasing", where)
        previous_step = step
        if event == "baseline":
            baselines.append(record)
            if step != 0:
                audit.error("baseline_step_invalid", "baseline must be at step 0", where)
            _validate_validation_record(audit, config, record, mode, where)
        elif event == "validation":
            validations.append(record)
            if step < 1 or step > steps_completed:
                audit.error("validation_step_invalid", "validation step is outside training history", where)
            _validate_validation_record(audit, config, record, mode, where)
        elif event == "train":
            if step in train_by_step:
                audit.error("duplicate_train_step", f"duplicate train step {step}", where)
            train_by_step[step] = record
            _validate_train_record(audit, record, mode, where)
        else:
            audit.error("unknown_metric_event", f"unknown metric event {event!r}", where)

    if len(baselines) != 1 or not records or records[0].get("event") != "baseline":
        audit.error("baseline_contract", "metrics must start with exactly one baseline record")
    expected_steps = set(range(1, steps_completed + 1))
    if set(train_by_step) != expected_steps:
        audit.error(
            "train_step_coverage",
            f"train steps must be exactly 1..{steps_completed}",
            path,
        )
    if not validations or validations[-1].get("step") != steps_completed:
        audit.error("final_validation_missing", "last completed step must have validation metrics")

    if baselines and summary.get("baseline_validation") != baselines[0]:
        audit.error("summary_baseline_mismatch", "summary baseline does not match metrics.jsonl")
    if validations and summary.get("last_validation") != validations[-1]:
        audit.error("summary_last_validation_mismatch", "summary last validation does not match metrics.jsonl")

    if validations:
        scores = [record.get("selection_metric") for record in validations]
        if all(_is_finite_number(value) for value in scores):
            maximum = max(float(value) for value in scores)
            if not _close(summary.get("best_validation_metric"), maximum):
                audit.error("best_metric_mismatch", "summary best metric is not the validation maximum")

    finite_gradients = [
        float(record["grad_norm"])
        for record in train_by_step.values()
        if _is_finite_number(record.get("grad_norm"))
    ]
    if finite_gradients and not any(value > 0 for value in finite_gradients):
        audit.warn("zero_gradients", "all recorded gradient norms are zero")
    if mode != "sft-w":
        stds = [
            float(record["reward_std"])
            for record in train_by_step.values()
            if _is_finite_number(record.get("reward_std"))
        ]
        if stds and sum(value < 0.1 for value in stds) / len(stds) > 0.5:
            audit.warn("low_reward_variance", "reward std is below 0.1 in most groups")
    if mode == "rl-w":
        rates = [
            float(record["yes_rate"])
            for record in train_by_step.values()
            if _is_finite_number(record.get("yes_rate"))
        ]
        if rates and sum(value < 0.05 or value > 0.95 for value in rates) / len(rates) > 0.5:
            audit.warn("action_collapse", "RL-W is collapsed toward always-yes or always-no")

    audit.details["metrics"] = {
        "records": len(records),
        "train_records": len(train_by_step),
        "validation_records": len(validations),
        "steps_completed": steps_completed,
    }
    return train_by_step, validations


def _validate_rl_w_rollouts(
    audit: _Audit,
    config: Mapping[str, Any],
    train_by_step: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    group_size = config.get("group_size")
    if not _is_int(group_size) or group_size < 2:
        audit.error("group_size_invalid", "RL group_size must be >= 2")
        return
    by_step: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for line, row in enumerate(rows, 1):
        step = row.get("step")
        if _is_int(step):
            by_step[step].append((line, row))
        else:
            audit.error("rollout_step_invalid", "rollout step must be an integer", f"rollouts.jsonl:{line}")
    if set(by_step) != set(train_by_step):
        audit.error("rollout_step_coverage", "RL-W rollouts must cover every train step exactly once")

    for step, entries in by_step.items():
        if len(entries) != 1:
            audit.error("rollout_group_count", f"RL-W step {step} must have one rollout row")
            continue
        line, row = entries[0]
        where = f"{audit.run_dir / 'rollouts.jsonl'}:{line}"
        metric = train_by_step.get(step)
        actions = row.get("actions")
        rewards = row.get("rewards")
        candidates = row.get("candidate_ids")
        if not isinstance(actions, list) or len(actions) != group_size or not all(
            action in ("No", "Yes") for action in actions
        ):
            audit.error("rollout_actions_invalid", f"actions must contain {group_size} No/Yes values", where)
            actions = []
        if not isinstance(rewards, list) or len(rewards) != group_size or not all(
            _is_finite_number(value) for value in rewards
        ):
            audit.error("rollout_rewards_invalid", f"rewards must contain {group_size} finite values", where)
            rewards = []
        if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], str):
            audit.error("rollout_candidates_invalid", "RL-W candidate_ids must contain one ID", where)
        elif metric is not None and candidates[0] != metric.get("candidate_id"):
            audit.error("rollout_candidate_mismatch", "rollout candidate differs from train metric", where)
        kl = _require_finite(audit, row, "kl", where, lower=-1e-6)
        if metric is None or not actions or not rewards:
            continue
        if kl is not None and not _close(kl, metric.get("kl")):
            audit.error("rollout_aggregate_mismatch", "rollout KL differs from train metric", where)
        if not _close(statistics.fmean(float(x) for x in rewards), metric.get("reward_mean")):
            audit.error("rollout_aggregate_mismatch", "reward mean differs from train metric", where)
        if not _close(statistics.pstdev(float(x) for x in rewards), metric.get("reward_std")):
            audit.error("rollout_aggregate_mismatch", "reward std differs from train metric", where)
        if not _close(sum(action == "Yes" for action in actions) / group_size, metric.get("yes_rate")):
            audit.error("rollout_aggregate_mismatch", "Yes-rate differs from train metric", where)
        objective = config.get("workspace_objective")
        if objective == "rank-continuous" and _is_finite_number(metric.get("workspace_percentile")):
            margin = 2.0 * float(metric["workspace_percentile"]) - 1.0
            expected = [margin if action == "Yes" else -margin for action in actions]
            if any(not _close(actual, wanted) for actual, wanted in zip(rewards, expected)):
                audit.error("workspace_reward_mismatch", "RL-W rewards do not match actions/percentile", where)
        elif objective == "top-k":
            if any(not (_close(value, -1.0) or _close(value, 1.0)) for value in rewards):
                audit.error("workspace_reward_mismatch", "top-k rewards must be -1 or 1", where)


def _percentile_ranks(values: Sequence[float]) -> list[float]:
    if len(values) == 1:
        return [0.5]
    denominator = len(values) - 1
    return [
        (sum(other < value for other in values) + 0.5 * (sum(other == value for other in values) - 1))
        / denominator
        for value in values
    ]


def _selector_workspace_reward(
    scores: Sequence[float], selected_indices: Sequence[int], contrastive: bool
) -> float:
    ranks = _percentile_ranks(scores)
    selected = [ranks[index] for index in selected_indices]
    selected_mean = statistics.fmean(selected)
    if not contrastive:
        return selected_mean
    unselected = [value for index, value in enumerate(ranks) if index not in selected_indices]
    return 0.5 * (1.0 + selected_mean - statistics.fmean(unselected))


def _validate_selector_rollouts(
    audit: _Audit,
    config: Mapping[str, Any],
    mode: str,
    train_by_step: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    group_size = config.get("group_size")
    budget = config.get("budget")
    if not _is_int(group_size) or group_size < 2 or not _is_int(budget) or budget < 1:
        audit.error("selector_shape_invalid", "selector group_size>=2 and budget>=1 are required")
        return
    by_step: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for line, row in enumerate(rows, 1):
        step = row.get("step")
        if _is_int(step):
            by_step[step].append((line, row))
        else:
            audit.error("rollout_step_invalid", "rollout step must be an integer", f"rollouts.jsonl:{line}")
    if set(by_step) != set(train_by_step):
        audit.error("rollout_step_coverage", "selector rollouts must cover every train step")

    for step, entries in by_step.items():
        metric = train_by_step.get(step)
        if len(entries) != group_size:
            audit.error("rollout_group_count", f"selector step {step} must have {group_size} rows")
        group_indices = [row.get("group_index") for _, row in entries]
        if group_indices != list(range(group_size)):
            audit.error("rollout_group_indices", f"selector step {step} group_index must be 0..G-1")
        step_rewards: list[float] = []
        step_qa: list[float] = []
        step_workspace: list[float] = []
        step_kls: list[float] = []
        common_episode: str | None = None
        common_candidates: list[str] | None = None
        for line, row in entries:
            where = f"{audit.run_dir / 'rollouts.jsonl'}:{line}"
            episode = row.get("episode_id")
            if not isinstance(episode, str):
                audit.error("episode_id_missing", "selector rollout needs episode_id", where)
            elif common_episode is None:
                common_episode = episode
            elif episode != common_episode:
                audit.error("rollout_episode_mismatch", "all rows in a group must share an episode", where)

            candidates = row.get("candidate_ids")
            if not isinstance(candidates, list) or not candidates or not all(
                isinstance(value, str) for value in candidates
            ) or len(set(candidates)) != len(candidates):
                audit.error("rollout_candidates_invalid", "candidate_ids must be a unique string list", where)
                continue
            if common_candidates is None:
                common_candidates = candidates
            elif candidates != common_candidates:
                audit.error("rollout_candidate_mismatch", "candidate order changes within a group", where)
            if budget >= len(candidates):
                audit.error("selector_budget_invalid", "selector budget must be less than candidate count", where)

            selected = row.get("selected_set")
            if not isinstance(selected, list) or len(selected) != budget or not all(
                isinstance(value, str) for value in selected
            ) or len(set(selected)) != len(selected) or not set(selected).issubset(candidates):
                audit.error("selected_set_invalid", f"selected_set must be {budget} unique candidate IDs", where)
                continue
            selected_indices = [candidates.index(value) for value in selected]
            if selected_indices != sorted(selected_indices):
                audit.error("selected_set_not_canonical", "selected_set must follow canonical candidate order", where)
            concepts = row.get("selected_concepts")
            if not isinstance(concepts, list) or len(concepts) != budget or not all(
                isinstance(value, str) for value in concepts
            ):
                audit.error("selected_concepts_invalid", "selected_concepts must match the exact budget", where)

            probability = _require_finite(
                audit, row, "selection_probability", where, lower=0.0, upper=1.0
            )
            log_probability = _require_finite(audit, row, "selection_log_probability", where)
            if probability is not None and probability <= 0:
                audit.error("selection_probability_invalid", "selection probability must be > 0", where)
            if log_probability is not None and log_probability > 1e-6:
                audit.error("selection_log_probability_invalid", "selection log-probability must be <= 0", where)
            if probability is not None and log_probability is not None and not _close(
                probability, math.exp(log_probability)
            ):
                audit.error("selection_probability_mismatch", "probability != exp(log_probability)", where)

            for key in ("QA_correct", "oracle_QA_correct", "full_context_correct", "contains_load_bearing"):
                if not isinstance(row.get(key), bool):
                    audit.error("rollout_boolean_invalid", f"{key} must be boolean", where)
            qa_reward = _require_finite(audit, row, "QA_reward", where, lower=0.0, upper=1.0)
            workspace_reward = _require_finite(
                audit, row, "workspace_reward", where, lower=0.0, upper=1.0
            )
            reward = _require_finite(audit, row, "reward", where, lower=0.0, upper=1.0)
            kl = _require_finite(audit, row, "KL", where, lower=-1e-6)
            if qa_reward is not None and isinstance(row.get("QA_correct"), bool) and not _close(
                qa_reward, float(row["QA_correct"])
            ):
                audit.error("qa_reward_mismatch", "QA_reward must equal QA_correct", where)

            scores = row.get("workspace_scores")
            if not isinstance(scores, list) or len(scores) != len(candidates) or not all(
                _is_finite_number(value) for value in scores
            ):
                audit.error("workspace_scores_invalid", "workspace_scores must align with candidates", where)
            elif workspace_reward is not None:
                expected_workspace = _selector_workspace_reward(
                    [float(value) for value in scores],
                    selected_indices,
                    config.get("workspace_set_reward") == "contrastive",
                )
                if not _close(workspace_reward, expected_workspace):
                    audit.error("workspace_reward_mismatch", "workspace set reward is inconsistent", where)
            verbal = row.get("verbal_scores")
            if not isinstance(verbal, list) or len(verbal) != len(candidates) or not all(
                value is None or _is_finite_number(value) for value in verbal
            ):
                audit.error("verbal_scores_invalid", "verbal_scores must align with candidates", where)

            if reward is not None and qa_reward is not None and workspace_reward is not None:
                if mode == "rl-qa":
                    expected_reward = qa_reward
                else:
                    lambda_qa = config.get("lambda_qa")
                    lambda_w = config.get("lambda_w")
                    if not _is_finite_number(lambda_qa) or not _is_finite_number(lambda_w) or (
                        float(lambda_qa) + float(lambda_w) <= 0
                    ):
                        audit.error("hybrid_coefficients_invalid", "hybrid coefficients must be finite and positive in sum")
                        expected_reward = reward
                    else:
                        expected_reward = (
                            float(lambda_qa) * qa_reward + float(lambda_w) * workspace_reward
                        ) / (float(lambda_qa) + float(lambda_w))
                if not _close(reward, expected_reward):
                    audit.error("selector_reward_mismatch", "selector total reward is inconsistent", where)

            contains = row.get("contains_load_bearing")
            correct = row.get("QA_correct")
            expected_failure = (
                "selection" if contains is False
                else ("recall_or_composition" if correct is False else "none")
            )
            if isinstance(contains, bool) and isinstance(correct, bool) and row.get("failure_type") != expected_failure:
                audit.error("failure_type_mismatch", "failure_type disagrees with containment/QA", where)
            if not isinstance(row.get("answer"), str):
                audit.error("answer_invalid", "selector answer must be a string", where)

            if reward is not None:
                step_rewards.append(reward)
            if qa_reward is not None:
                step_qa.append(qa_reward)
            if workspace_reward is not None:
                step_workspace.append(workspace_reward)
            if kl is not None:
                step_kls.append(kl)

        if metric is None or len(entries) != group_size:
            continue
        aggregates = (
            (step_rewards, "reward_mean", statistics.fmean),
            (step_rewards, "reward_std", statistics.pstdev),
            (step_qa, "qa_reward", statistics.fmean),
            (step_workspace, "workspace_reward", statistics.fmean),
            (step_kls, "kl", statistics.fmean),
        )
        for values, key, reducer in aggregates:
            if len(values) == group_size and not _close(reducer(values), metric.get(key)):
                audit.error("rollout_aggregate_mismatch", f"selector {key} differs from train metric")
        if common_episode is not None and common_episode != metric.get("episode_id"):
            audit.error("rollout_episode_mismatch", "rollout episode differs from train metric")
        selected_sets = [tuple(row.get("selected_set", [])) for _, row in entries]
        unique_count = len(set(selected_sets))
        if metric.get("candidate_count") != len(common_candidates or []):
            audit.error("rollout_aggregate_mismatch", "selector candidate_count differs from rollout")
        if metric.get("number_unique_selected_sets") != unique_count:
            audit.error("rollout_aggregate_mismatch", "selector unique-set count differs from rollout")
        if not _close(metric.get("fraction_unique_selected_sets"), unique_count / group_size):
            audit.error("rollout_aggregate_mismatch", "selector unique-set fraction differs from rollout")
        if len(step_qa) == group_size:
            mixed_qa = len(set(step_qa)) > 1
            if metric.get("mixed_QA_reward_group") is not mixed_qa:
                audit.error("rollout_aggregate_mismatch", "selector mixed-QA flag differs from rollout")
        contain_values = [row.get("contains_load_bearing") for _, row in entries]
        if all(isinstance(value, bool) for value in contain_values):
            containment_rate = sum(contain_values) / group_size
            if not _close(metric.get("containment_rate"), containment_rate):
                audit.error("rollout_aggregate_mismatch", "selector containment rate differs from rollout")
            if metric.get("mixed_containment_group") is not (len(set(contain_values)) > 1):
                audit.error("rollout_aggregate_mismatch", "selector mixed-containment flag differs from rollout")


def _validate_reporter_correlation_artifact(
    audit: _Audit,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    metric_records: Sequence[Mapping[str, Any]],
) -> None:
    """Recompute ID reporter diagnostics from their candidate-level rows."""

    if config.get("reporter_correlation_diagnostics") is not True:
        return
    path = audit.run_dir / "reporter_correlations.jsonl"
    records = _load_jsonl(audit, path)
    if records is None:
        return
    expected_metrics = [
        row
        for row in metric_records
        if row.get("event") in ("baseline", "validation")
    ]
    if len(records) != len(expected_metrics):
        audit.error(
            "reporter_correlation_record_count",
            "reporter correlation records must match baseline/validation events",
            path,
        )
    samples = config.get("reporter_correlation_bootstrap_samples")
    seed = config.get("reporter_correlation_bootstrap_seed")
    if not _is_int(samples) or samples < 0 or not _is_int(seed):
        return
    candidate_counts: list[int] = []
    for index, (record, metric) in enumerate(zip(records, expected_metrics), 1):
        where = f"{path}:{index}"
        if record.get("schema_version") != 1:
            audit.error(
                "reporter_correlation_schema_invalid",
                "reporter correlation schema_version must be 1",
                where,
            )
        if record.get("scope") != "fixed_id_validation":
            audit.error(
                "reporter_correlation_scope_invalid",
                "reporter correlation artifact must use fixed_id_validation scope",
                where,
            )
        if record.get("event") != metric.get("event") or record.get("step") != metric.get("step"):
            audit.error(
                "reporter_correlation_event_mismatch",
                "reporter correlation event/step differs from metrics.jsonl",
                where,
            )
        rows = record.get("rows")
        if not isinstance(rows, list) or not rows:
            audit.error(
                "reporter_correlation_rows_invalid",
                "reporter correlation rows must be a non-empty list",
                where,
            )
            continue
        episode_ids = {
            row.get("episode_id")
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("episode_id"), str)
        }
        if not episode_ids.issubset(audit.validation_episode_ids):
            audit.error(
                "reporter_correlation_split_leak",
                "reporter correlations contain a non-validation episode",
                where,
            )
        if audit.profile == "formal" and episode_ids != audit.validation_episode_ids:
            audit.error(
                "reporter_correlation_validation_incomplete",
                "formal reporter correlations must cover all fixed validation episodes",
                where,
            )
        try:
            expected_summary = summarize_reporter_correlations(
                rows,
                bootstrap_samples=samples,
                bootstrap_seed=seed,
            )
        except (TypeError, ValueError) as exc:
            audit.error("reporter_correlation_rows_invalid", str(exc), where)
            continue
        observed_summary = record.get("summary")
        if observed_summary != expected_summary:
            audit.error(
                "reporter_correlation_summary_mismatch",
                "reporter correlation summary does not recompute from candidate rows",
                where,
            )
        if metric.get("reporter_correlations") != observed_summary:
            audit.error(
                "reporter_correlation_metric_mismatch",
                "metrics.jsonl and reporter correlation artifact summaries differ",
                where,
            )
        if not _close(metric.get("verbal_auc"), expected_summary.get("utility_auc")):
            audit.error(
                "reporter_verbal_auc_mismatch",
                "verbal_auc does not recompute from reporter rows",
                where,
            )
        within_auc = within_episode_utility_auc(rows)
        if not _close(metric.get("verbal_within_episode_auc"), within_auc):
            audit.error(
                "reporter_within_auc_mismatch",
                "verbal_within_episode_auc does not recompute from reporter rows",
                where,
            )
        if not _close(metric.get("verbal_yes_rate"), expected_summary.get("yes_rate")):
            audit.error(
                "reporter_yes_rate_mismatch",
                "verbal_yes_rate does not recompute from reporter rows",
                where,
            )
        if metric.get("episodes") != expected_summary.get("n_episodes"):
            audit.error(
                "reporter_episode_count_mismatch",
                "selector validation episode count differs from reporter rows",
                where,
            )
        candidate_counts.append(len(rows))

    declared = _resolve_declared_path(summary.get("reporter_correlation_artifact"), audit.run_dir)
    if declared is None or declared.resolve() != path.resolve():
        audit.error(
            "reporter_correlation_summary_path_mismatch",
            "summary reporter correlation artifact path is inconsistent",
            path,
        )
    audit.details["reporter_correlations"] = {
        "records": len(records),
        "candidate_counts": candidate_counts,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def _validate_dropout(
    audit: _Audit,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    mode: str,
) -> None:
    path = audit.run_dir / "dropout_audit.json"
    if mode == "sft-w":
        if path.exists():
            audit.error("unexpected_dropout_audit", "SFT must not claim an RL dropout audit", path)
        if summary.get("dropout_audit") is not None:
            audit.error("summary_dropout_mismatch", "SFT summary dropout_audit must be null")
        return

    value = _load_json(audit, path)
    if value is None:
        return
    count = value.get("dropout_modules_zeroed")
    fields = value.get("config_fields_zeroed")
    if not _is_int(count) or count < 0:
        audit.error("dropout_audit_invalid", "dropout_modules_zeroed must be a non-negative integer", path)
    if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
        audit.error("dropout_audit_invalid", "config_fields_zeroed must be a string list", path)
    if summary.get("dropout_audit") != value:
        audit.error("summary_dropout_mismatch", "summary dropout audit differs from dropout_audit.json")
    detailed_keys = {
        "dropout_modules_found",
        "dropout_modules_modified",
        "config_fields_found",
        "config_fields_modified",
        "remaining_nonzero",
        "postcondition_satisfied",
    }
    is_detailed = detailed_keys.issubset(value)
    if not is_detailed:
        audit.warn(
            "legacy_dropout_audit_no_postscan",
            "legacy dropout audit records mutations but not a postcondition scan",
            path,
        )
        if audit.profile == "formal":
            audit.error(
                "formal_dropout_postscan_missing",
                "formal RL artifacts require the detailed dropout postcondition schema",
                path,
            )
    else:
        residual = value.get("remaining_nonzero")
        if residual != [] or value.get("postcondition_satisfied") is not True:
            audit.error(
                "dropout_remaining_nonzero",
                "dropout post-scan must report remaining_nonzero=[] and success=true",
                path,
            )
        for prefix in ("dropout_modules", "config_fields"):
            found = value.get(f"{prefix}_found")
            modified = value.get(f"{prefix}_modified")
            if not isinstance(found, list) or not all(isinstance(row, dict) for row in found):
                audit.error("dropout_audit_invalid", f"{prefix}_found must be an object list", path)
                continue
            if not isinstance(modified, list) or not all(isinstance(name, str) for name in modified):
                audit.error("dropout_audit_invalid", f"{prefix}_modified must be a string list", path)
                continue
            names = [row.get("name") for row in found]
            if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
                audit.error("dropout_audit_invalid", f"{prefix}_found names must be unique strings", path)
            expected_modified = [
                row.get("name") for row in found if row.get("modified") is True
            ]
            if modified != expected_modified:
                audit.error("dropout_audit_invalid", f"{prefix}_modified disagrees with found rows", path)
            for row in found:
                if row.get("modified") not in (True, False) or not _close(row.get("after"), 0.0):
                    audit.error(
                        "dropout_audit_invalid",
                        f"{prefix}_found rows must have boolean modified and after=0",
                        path,
                    )
                    break
        if isinstance(value.get("dropout_modules_modified"), list) and count != len(
            value["dropout_modules_modified"]
        ):
            audit.error("dropout_audit_invalid", "legacy module count disagrees with detailed audit", path)
        if isinstance(value.get("config_fields_modified"), list) and fields != value.get(
            "config_fields_modified"
        ):
            audit.error("dropout_audit_invalid", "legacy config field list disagrees with detailed audit", path)
    lora_dropout = config.get("lora_dropout")
    if audit.profile == "formal" and (not _is_finite_number(lora_dropout) or float(lora_dropout) != 0.0):
        audit.error("formal_dropout_nonzero", "formal lora_dropout must be 0")


def _validate_adapter_directory(
    audit: _Audit,
    raw_path: Any,
    *,
    label: str,
    expected_step: int,
    expected_metric: float | None = None,
) -> Path | None:
    path = _resolve_declared_path(raw_path, audit.run_dir)
    if path is None or not path.is_dir():
        audit.error("checkpoint_missing", f"{label} directory is missing", str(raw_path))
        return None
    if not _within_run(path, audit.run_dir):
        audit.error("checkpoint_outside_run", f"{label} must be contained in the run directory", path)
    state = _load_json(audit, path / "training_state.json")
    if state is not None:
        if state.get("step") != expected_step:
            audit.error("checkpoint_step_mismatch", f"{label} training_state step is inconsistent", path)
        if expected_metric is not None and not _close(state.get("metric"), expected_metric):
            audit.error("checkpoint_metric_mismatch", f"{label} training_state metric is inconsistent", path)
    if not (path / "adapter_config.json").is_file():
        audit.error("adapter_config_missing", f"{label} lacks adapter_config.json", path)
    if not any(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        audit.error("adapter_weights_missing", f"{label} lacks adapter weights", path)
    if not (path / "tokenizer_config.json").is_file():
        audit.warn("tokenizer_artifact_missing", f"{label} lacks tokenizer_config.json", path)
    return path


def _validate_checkpoints(
    audit: _Audit,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    validations: Sequence[Mapping[str, Any]],
) -> None:
    best_file = _load_json(audit, audit.run_dir / "best_checkpoint.json")
    if best_file is None:
        return
    best_step = best_file.get("step")
    best_metric = best_file.get("metric")
    if not _is_int(best_step) or best_step < 1:
        audit.error("best_checkpoint_invalid", "best checkpoint step must be >= 1")
        return
    if not _is_finite_number(best_metric):
        audit.error("best_checkpoint_invalid", "best checkpoint metric must be finite")
        return
    if best_file.get("path") != summary.get("best_checkpoint"):
        audit.error("best_checkpoint_summary_mismatch", "best checkpoint path differs from summary")
    if not _close(best_metric, summary.get("best_validation_metric")):
        audit.error("best_checkpoint_summary_mismatch", "best checkpoint metric differs from summary")
    if not any(
        row.get("step") == best_step and _close(row.get("selection_metric"), best_metric)
        for row in validations
    ):
        audit.error("best_checkpoint_validation_mismatch", "best checkpoint is not backed by validation")
    best_path = _validate_adapter_directory(
        audit,
        best_file.get("path"),
        label="best checkpoint",
        expected_step=best_step,
        expected_metric=float(best_metric),
    )
    if best_path is not None and best_path.name != f"best-step-{best_step}":
        audit.error("best_checkpoint_name_mismatch", "best checkpoint directory name encodes the wrong step")

    steps_completed = summary.get("steps_completed")
    if _is_int(steps_completed):
        _validate_adapter_directory(
            audit,
            summary.get("final_adapter"),
            label="final adapter",
            expected_step=steps_completed,
        )

    if audit.profile == "formal":
        resolved = _canonical_model(config.get("resolved_model_name_or_path"))
        if resolved != FORMAL_MODEL:
            audit.error("resolved_model_mismatch", "resolved formal model differs from declared checkpoint")
        commit = config.get("resolved_model_commit")
        if not isinstance(commit, str) or not commit:
            audit.warn("resolved_commit_missing", "formal artifact lacks a resolved model commit")
        trainable = config.get("trainable_parameter_count")
        total = config.get("total_parameter_count")
        if not _is_int(trainable) or not _is_int(total) or trainable <= 0 or total <= trainable:
            audit.error("parameter_counts_invalid", "formal parameter counts must satisfy 0 < trainable < total")
        versions = config.get("software_versions")
        required = ("python", "torch", "transformers", "peft", "numpy", "scikit-learn")
        if not isinstance(versions, dict) or any(not versions.get(key) for key in required):
            audit.error("software_versions_incomplete", "formal software version provenance is incomplete")


def _validate_non_dry(
    audit: _Audit, config: Mapping[str, Any], mode: str
) -> None:
    summary = _load_json(audit, audit.run_dir / "summary.json")
    records = _load_jsonl(audit, audit.run_dir / "metrics.jsonl")
    if summary is None or records is None:
        return
    if summary.get("mode") != mode:
        audit.error("summary_mode_mismatch", "summary mode differs from run_config")
    if summary.get("workspace_teacher_model") != config.get("workspace_teacher_model"):
        audit.error("summary_teacher_mismatch", "summary teacher differs from run_config")
    if summary.get("teacher_matches_policy_reference") != config.get(
        "teacher_matches_policy_reference"
    ):
        audit.error("summary_teacher_match_mismatch", "summary teacher-match flag differs")
    if summary.get("effective_advantage_normalization") != config.get(
        "effective_advantage_normalization"
    ):
        audit.error("summary_advantage_mismatch", "summary advantage mode differs from run_config")
    _require_finite(audit, summary, "elapsed_seconds", "summary.json", lower=0.0)
    diagnostics = summary.get("diagnostics")
    if not isinstance(diagnostics, list) or not all(isinstance(value, str) for value in diagnostics):
        audit.error("summary_diagnostics_invalid", "summary diagnostics must be a string list")

    train_by_step, validations = _validate_metrics(audit, config, summary, mode, records)
    _validate_reporter_correlation_artifact(audit, config, summary, records)
    if config.get("reporter_correlation_diagnostics") is True:
        best_validation = summary.get("best_validation")
        if not isinstance(best_validation, dict) or not any(
            row == best_validation for row in validations
        ):
            audit.error(
                "best_validation_missing",
                "summary.best_validation must reproduce one validation record",
            )
        elif not _close(
            best_validation.get("selection_metric"),
            summary.get("best_validation_metric"),
        ):
            audit.error(
                "best_validation_metric_mismatch",
                "summary.best_validation does not match the selected metric",
            )
    observed_selector_diagnostics = summary.get("selector_training_diagnostics")
    if mode in SELECTOR_MODES:
        try:
            expected_selector_diagnostics = summarize_selector_training(
                list(train_by_step.values()),
                window_size=int(config.get("diagnostic_every")),
            )
        except (TypeError, ValueError) as exc:
            audit.error("selector_diagnostics_invalid", str(exc))
        else:
            if observed_selector_diagnostics != expected_selector_diagnostics:
                audit.error(
                    "selector_diagnostics_mismatch",
                    "summary selector diagnostics do not recompute from metrics.jsonl",
                )
            audit.details["selector_training_diagnostics"] = expected_selector_diagnostics
    elif observed_selector_diagnostics is not None:
        audit.error(
            "unexpected_selector_diagnostics",
            "reporter modes must store selector_training_diagnostics as null",
        )
    _validate_dropout(audit, config, summary, mode)
    _validate_checkpoints(audit, config, summary, validations)

    rollout_path = audit.run_dir / "rollouts.jsonl"
    if mode == "sft-w":
        if rollout_path.exists():
            audit.warn("unexpected_sft_rollouts", "SFT does not define a rollout artifact")
    else:
        rollouts = _load_jsonl(audit, rollout_path)
        if rollouts is not None:
            if mode == "rl-w":
                _validate_rl_w_rollouts(audit, config, train_by_step, rollouts)
            else:
                _validate_selector_rollouts(audit, config, mode, train_by_step, rollouts)
            audit.details["rollouts"] = {"records": len(rollouts)}


def validate_run(
    run_dir: str | Path,
    *,
    profile: str,
    expected_manifest_sha256: str | None = None,
    expected_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a strict-JSON-serializable validation report for one run."""
    root = Path(run_dir).expanduser().resolve()
    audit = _Audit(root, profile)
    mode: str | None = None
    if profile not in PROFILES:
        audit.error("invalid_profile", f"profile must be one of {PROFILES}")
        return audit.report()
    if not root.is_dir():
        audit.error("run_dir_missing", "run directory does not exist", root)
        return audit.report()

    config = _load_json(audit, root / "run_config.json")
    manifest = _load_json(audit, root / "split_manifest.json")
    if config is not None:
        mode = _validate_config(audit, config, expected_config)
    if config is not None and manifest is not None:
        _validate_manifest(audit, config, manifest, expected_manifest_sha256)
    if config is not None and mode is not None:
        if profile == "dry-run":
            _validate_dry_artifacts(audit, mode)
        else:
            _validate_non_dry(audit, config, mode)
    return audit.report(mode)


def _load_expected_config(path: Path) -> Mapping[str, Any]:
    value = _strict_load_text(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected config must be a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument(
        "--expected-manifest-sha256",
        help="require equality with a previously accepted split manifest",
    )
    parser.add_argument(
        "--expected-config",
        type=Path,
        help="JSON object containing locked run_config key/value pairs",
    )
    parser.add_argument("--out", type=Path, help="optionally write the JSON report")
    args = parser.parse_args(argv)

    try:
        expected_config = (
            _load_expected_config(args.expected_config) if args.expected_config else None
        )
        report = validate_run(
            args.run_dir,
            profile=args.profile,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_config=expected_config,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "run_dir": str(Path(args.run_dir).expanduser().resolve()),
            "profile": args.profile,
            "mode": None,
            "status": "fail",
            "errors": [_issue("expected_config_invalid", str(exc), args.expected_config)],
            "warnings": [],
            "details": {},
        }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
