"""Read-only, fail-closed reproducibility audit for shared unified-OOD conditions.

The seed-expansion evaluator contains additional adapters, so comparing the two
JSON files byte-for-byte is neither possible nor desirable.  This audit instead
compares the five conditions that must be invariant across the sealed seed-0 and
seed-expansion evaluations, while independently validating the evaluator's raw
per-item/per-episode schema and ordering contract.

Only explicitly named files are read.  The sole write is a new report opened in
exclusive-create mode; input files are never modified and directories are never
created by this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
AUDIT_NAME = "unified_ood_shared_condition_reproducibility"
SHARED_CONDITIONS = (
    "original",
    "sft-w-s0-k2",
    "rl-qa-s0-k2",
    "workspace",
    "oracle",
)
SHARED_ADAPTERS = ("sft-w-s0-k2", "rl-qa-s0-k2")
NO_HARM_CONDITIONS = ("original", *SHARED_ADAPTERS)
EXPECTED_BUDGETS = (2,)
DEFAULT_ABS_TOL = 1e-8
DEFAULT_REL_TOL = 1e-7
MAX_ISSUE_EXAMPLES = 100


@dataclass(frozen=True)
class SourceContract:
    source: str
    n_episodes: int
    n_items: int
    results_path: str
    battery_path: str
    old_skip_no_harm: bool | None
    new_skip_no_harm: bool | None


SOURCE_CONTRACTS: dict[str, SourceContract] = {
    # The sealed Decoupled sensitivity raw deliberately skipped the expensive
    # adapter-enabled no-harm pass.  A new raw cannot recreate data that the old
    # file never contained, so that endpoint is explicitly N/A for this pair.
    "decoupled": SourceContract(
        "decoupled",
        68,
        335,
        "data/results/results_v4f_7B-Instruct.json",
        "data/benchmarks/battery_v4_final.json",
        True,
        None,
    ),
    "compositional": SourceContract(
        "compositional",
        52,
        261,
        "data/results/results_v3f_7B-Instruct.json",
        "data/benchmarks/battery_v3d.json",
        None,
        None,
    ),
}


FORMAL_CONFIG = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "rating_json": {},
    "embedding_model": None,
    "budgets": [2],
    "workspace_top_k": 2,
    "device": "cuda",
    "dtype": "bfloat16",
    "max_length": 2048,
    "max_new_tokens": 64,
    "admission_batch_size": 16,
    "qa_batch_size": 1,
    "bootstrap_samples": 4000,
    "bootstrap_seed": 0,
    "skip_qa": False,
    "original_verbal_source": "precomputed_v_ref",
    "policy_input_fields": ["context", "candidate.concept"],
    "probe_visible_to_policy": False,
    "recall_model": "adapter-disabled base checkpoint",
}

ITEM_REQUIRED_KEYS = frozenset(
    {
        "uid",
        "episode_uid",
        "source",
        "source_episode",
        "candidate_index",
        "concept",
        "label",
        "scores",
    }
)
ITEM_OPTIONAL_KEYS = frozenset({"model_log_odds", "role", "provenance"})
EPISODE_REQUIRED_KEYS = frozenset(
    {
        "uid",
        "source",
        "source_episode",
        "probe_question",
        "gold_answer",
        "policies",
        "refs",
    }
)
EPISODE_OPTIONAL_KEYS = frozenset({"no_harm_full_context"})
POLICY_KEYS = frozenset({"within_episode_auc", "selections"})
SELECTION_KEYS = frozenset(
    {
        "selected_indices",
        "selected_candidate_uids",
        "selected_concepts",
        "contains_load_bearing",
        "contains_all_load_bearing",
        "selected_load_bearing",
        "total_load_bearing",
        "load_bearing_recall",
        "qa",
    }
)
QA_KEYS = frozenset({"selected_concepts", "answer", "correct"})
NO_HARM_KEYS = frozenset({"answer", "correct"})
ORACLE_REF_KEYS = frozenset(
    {"selected_indices", "selected_concepts", "answer", "correct"}
)
REFERENCE_KEYS = frozenset({"oracle@2", "full_context", "no_memory"})


class IssueCollector:
    """Count every issue while bounding report size."""

    def __init__(self, *, limit: int = MAX_ISSUE_EXAMPLES) -> None:
        self.total = 0
        self.by_code: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []
        self.limit = limit

    def add(
        self,
        code: str,
        path: str,
        message: str,
        *,
        old: Any = None,
        new: Any = None,
        values_present: bool = False,
    ) -> None:
        self.total += 1
        self.by_code[code] += 1
        if len(self.examples) >= self.limit:
            return
        row: dict[str, Any] = {"code": code, "path": path, "message": message}
        if values_present:
            row["old"] = _preview(old)
            row["new"] = _preview(new)
        self.examples.append(row)

    def report(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.total == 0 else "fail",
            "issue_count": self.total,
            "issue_counts_by_code": dict(sorted(self.by_code.items())),
            "issue_examples": self.examples,
            "issue_examples_truncated": self.total > len(self.examples),
        }


class ComparisonCounter:
    def __init__(self) -> None:
        self.records: Counter[str] = Counter()
        self.scalar_values: Counter[str] = Counter()

    def record(self, category: str, count: int = 1) -> None:
        self.records[category] += count

    def scalar(self, category: str) -> None:
        self.scalar_values[category] += 1

    def report(self) -> dict[str, Any]:
        categories = sorted(set(self.records) | set(self.scalar_values))
        return {
            category: {
                "records": self.records.get(category, 0),
                "scalar_values": self.scalar_values.get(category, 0),
            }
            for category in categories
        }


def _preview(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 300 else value[:297] + "..."
    try:
        rendered = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        rendered = repr(value)
    return rendered if len(rendered) <= 500 else rendered[:497] + "..."


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token!r}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite_locations(value: Any, prefix: str = "$") -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    locations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            locations.extend(_nonfinite_locations(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            locations.extend(_nonfinite_locations(child, f"{prefix}[{index}]"))
    return locations


def _load_input(
    path: Path, side: str, issues: IssueCollector
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        issues.add("input_read_error", side, f"cannot read {path}: {exc}")
        return None, None
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        issues.add("invalid_json", side, f"invalid strict JSON in {path}: {exc}")
        return None, digest
    if not isinstance(value, dict):
        issues.add("invalid_json_type", side, "top-level JSON must be an object")
        return None, digest
    nonfinite = _nonfinite_locations(value)
    if nonfinite:
        issues.add(
            "nonfinite_json_value",
            side,
            f"non-finite numeric values at {nonfinite[:5]}",
        )
        return None, digest
    return value, digest


def _is_exact(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    return type(actual) is type(expected) and actual == expected


def _require_exact(
    actual: Any,
    expected: Any,
    path: str,
    issues: IssueCollector,
    code: str = "schema_value_mismatch",
) -> bool:
    passed = _is_exact(actual, expected)
    if not passed:
        issues.add(
            code,
            path,
            "value does not satisfy the sealed schema/protocol",
            old=expected,
            new=actual,
            values_present=True,
        )
    return passed


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _mapping(
    value: Any, path: str, issues: IssueCollector
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        issues.add("schema_type", path, "must be an object")
        return None
    return value


def _array(value: Any, path: str, issues: IssueCollector) -> list[Any] | None:
    if not isinstance(value, list):
        issues.add("schema_type", path, "must be an array")
        return None
    return value


def _check_key_set(
    value: Mapping[str, Any],
    expected: frozenset[str],
    path: str,
    issues: IssueCollector,
    *,
    allow: frozenset[str] = frozenset(),
) -> bool:
    keys = set(value)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected - allow)
    if missing:
        issues.add("schema_missing_keys", path, f"missing keys: {missing}")
    if unknown:
        issues.add("schema_unknown_keys", path, f"unknown keys: {unknown}")
    return not missing and not unknown


def _condition_subsequence(order: Any) -> list[str] | None:
    if not isinstance(order, list) or not all(isinstance(value, str) for value in order):
        return None
    return [value for value in order if value in SHARED_CONDITIONS]


def _validate_config(
    payload: Mapping[str, Any],
    contract: SourceContract,
    side: str,
    issues: IssueCollector,
) -> Mapping[str, Any] | None:
    _require_exact(payload.get("schema_version"), SCHEMA_VERSION, f"{side}.schema_version", issues)
    config = _mapping(payload.get("config"), f"{side}.config", issues)
    if config is None:
        return None
    for key, expected in FORMAL_CONFIG.items():
        if key not in config:
            issues.add("config_missing_field", f"{side}.config", f"missing {key!r}")
        else:
            _require_exact(
                config[key],
                expected,
                f"{side}.config.{key}",
                issues,
                "formal_config_mismatch",
            )

    skip_no_harm = config.get("skip_no_harm")
    if not isinstance(skip_no_harm, bool):
        issues.add(
            "formal_config_mismatch",
            f"{side}.config.skip_no_harm",
            "must be boolean",
        )
    else:
        expected_skip = (
            contract.old_skip_no_harm
            if side == "old"
            else contract.new_skip_no_harm
        )
        if expected_skip is not None:
            _require_exact(
                skip_no_harm,
                expected_skip,
                f"{side}.config.skip_no_harm",
                issues,
                "formal_config_mismatch",
            )
        no_harm_batch_size = config.get("no_harm_batch_size")
        if (
            isinstance(no_harm_batch_size, bool)
            or not isinstance(no_harm_batch_size, int)
            or no_harm_batch_size <= 0
        ):
            issues.add(
                "formal_config_mismatch",
                f"{side}.config.no_harm_batch_size",
                "must be a positive integer",
            )
        elif not skip_no_harm:
            _require_exact(
                no_harm_batch_size,
                1,
                f"{side}.config.no_harm_batch_size",
                issues,
                "formal_config_mismatch",
            )

    specs = _array(config.get("specs"), f"{side}.config.specs", issues)
    if specs is not None:
        if len(specs) != 1:
            issues.add(
                "source_isolation_violation",
                f"{side}.config.specs",
                f"expected exactly one source spec, got {len(specs)}",
            )
        elif isinstance(specs[0], Mapping):
            expected_spec = {
                "name": contract.source,
                "source": contract.source,
                "results_path": contract.results_path,
                "battery_path": contract.battery_path,
            }
            _require_exact(
                dict(specs[0]),
                expected_spec,
                f"{side}.config.specs[0]",
                issues,
                "source_spec_mismatch",
            )
        else:
            issues.add("schema_type", f"{side}.config.specs[0]", "must be an object")

    adapters = _mapping(config.get("adapters"), f"{side}.config.adapters", issues)
    if adapters is not None:
        for condition in SHARED_ADAPTERS:
            value = adapters.get(condition)
            if not isinstance(value, str) or not value:
                issues.add(
                    "shared_adapter_missing",
                    f"{side}.config.adapters.{condition}",
                    "shared adapter path must be a non-empty string",
                )

    condition_order = payload.get("condition_order")
    shared_order = _condition_subsequence(condition_order)
    if shared_order is None:
        issues.add("schema_type", f"{side}.condition_order", "must be an array of strings")
    else:
        if len(condition_order) != len(set(condition_order)):
            issues.add(
                "duplicate_condition",
                f"{side}.condition_order",
                "condition names must be unique",
            )
        _require_exact(
            shared_order,
            list(SHARED_CONDITIONS),
            f"{side}.condition_order[shared]",
            issues,
            "shared_condition_order_mismatch",
        )

    for key in ("refs", "mcnemar"):
        section = _mapping(payload.get(key), f"{side}.{key}", issues)
        if section is not None:
            _require_exact(
                section.get("skipped"),
                False,
                f"{side}.{key}.skipped",
                issues,
                "evaluation_was_skipped",
            )
    no_harm = _mapping(payload.get("no_harm"), f"{side}.no_harm", issues)
    if no_harm is not None and isinstance(skip_no_harm, bool):
        _require_exact(
            no_harm.get("skipped"),
            skip_no_harm,
            f"{side}.no_harm.skipped",
            issues,
            "no_harm_config_artifact_mismatch",
        )
    return config


def _validate_item_rows(
    payload: Mapping[str, Any],
    contract: SourceContract,
    side: str,
    config: Mapping[str, Any] | None,
    issues: IssueCollector,
) -> tuple[list[Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    raw_items = _array(payload.get("per_item"), f"{side}.per_item", issues)
    if raw_items is None:
        return [], {}
    if len(raw_items) != contract.n_items:
        issues.add(
            "item_count_mismatch",
            f"{side}.per_item",
            f"expected {contract.n_items} items, got {len(raw_items)}",
        )

    rows: list[Mapping[str, Any]] = []
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    seen_uids: set[str] = set()
    expected_episode = 0
    expected_candidate = 0
    expected_log_odds = set(SHARED_ADAPTERS)
    if config is not None and config.get("original_verbal_source") == "model_forward":
        expected_log_odds.add("original")

    for row_index, raw_row in enumerate(raw_items):
        path = f"{side}.per_item[{row_index}]"
        row = _mapping(raw_row, path, issues)
        if row is None:
            continue
        rows.append(row)
        _check_key_set(row, ITEM_REQUIRED_KEYS, path, issues, allow=ITEM_OPTIONAL_KEYS)
        source_episode = row.get("source_episode")
        candidate_index = row.get("candidate_index")
        _require_exact(row.get("source"), contract.source, f"{path}.source", issues, "source_mismatch")
        if not isinstance(source_episode, int) or isinstance(source_episode, bool):
            issues.add("schema_type", f"{path}.source_episode", "must be an integer")
        if not isinstance(candidate_index, int) or isinstance(candidate_index, bool):
            issues.add("schema_type", f"{path}.candidate_index", "must be an integer")
        if isinstance(source_episode, int) and not isinstance(source_episode, bool) and isinstance(
            candidate_index, int
        ) and not isinstance(candidate_index, bool):
            if source_episode == expected_episode and candidate_index == expected_candidate:
                expected_candidate += 1
            elif source_episode == expected_episode + 1 and candidate_index == 0:
                expected_episode += 1
                expected_candidate = 1
            else:
                issues.add(
                    "candidate_order_mismatch",
                    path,
                    "items must be ordered by contiguous source_episode then candidate_index",
                )
                expected_episode = source_episode
                expected_candidate = candidate_index + 1
            expected_episode_uid = f"{contract.source}:episode:{source_episode:06d}"
            expected_uid = f"{expected_episode_uid}:candidate:{candidate_index:03d}"
            _require_exact(
                row.get("episode_uid"),
                expected_episode_uid,
                f"{path}.episode_uid",
                issues,
                "episode_uid_mismatch",
            )
            _require_exact(
                row.get("uid"),
                expected_uid,
                f"{path}.uid",
                issues,
                "candidate_uid_mismatch",
            )
        uid = row.get("uid")
        if not isinstance(uid, str) or not uid:
            issues.add("schema_type", f"{path}.uid", "must be a non-empty string")
        elif uid in seen_uids:
            issues.add("duplicate_candidate_uid", f"{path}.uid", f"duplicate UID {uid!r}")
        else:
            seen_uids.add(uid)
        for key in ("episode_uid", "concept", "label"):
            if not isinstance(row.get(key), str) or not row.get(key):
                issues.add("schema_type", f"{path}.{key}", "must be a non-empty string")

        scores = _mapping(row.get("scores"), f"{path}.scores", issues)
        if scores is not None:
            missing = [condition for condition in SHARED_CONDITIONS if condition not in scores]
            if missing:
                issues.add(
                    "shared_scores_missing",
                    f"{path}.scores",
                    f"missing shared conditions: {missing}",
                )
            for condition in SHARED_CONDITIONS:
                if condition in scores and not _finite_number(scores[condition]):
                    issues.add(
                        "invalid_admission_score",
                        f"{path}.scores.{condition}",
                        "must be finite numeric",
                    )

        log_odds = row.get("model_log_odds")
        log_mapping = _mapping(log_odds, f"{path}.model_log_odds", issues)
        if log_mapping is not None:
            shared_present = {key for key in log_mapping if key in SHARED_CONDITIONS}
            if shared_present != expected_log_odds:
                issues.add(
                    "model_log_odds_conditions",
                    f"{path}.model_log_odds",
                    f"expected shared model-log-odds keys {sorted(expected_log_odds)}, "
                    f"got {sorted(shared_present)}",
                )
            for condition in expected_log_odds:
                if condition in log_mapping and not _finite_number(log_mapping[condition]):
                    issues.add(
                        "invalid_model_log_odds",
                        f"{path}.model_log_odds.{condition}",
                        "must be finite numeric",
                    )

        episode_uid = row.get("episode_uid")
        if isinstance(episode_uid, str):
            by_episode.setdefault(episode_uid, []).append(row)

    if raw_items and expected_episode != contract.n_episodes - 1:
        issues.add(
            "item_episode_coverage",
            f"{side}.per_item",
            f"last source_episode is {expected_episode}; expected {contract.n_episodes - 1}",
        )
    return rows, by_episode


def _validate_answer(
    value: Any,
    path: str,
    issues: IssueCollector,
    *,
    expected_keys: frozenset[str],
) -> Mapping[str, Any] | None:
    detail = _mapping(value, path, issues)
    if detail is None:
        return None
    _check_key_set(detail, expected_keys, path, issues)
    if not isinstance(detail.get("answer"), str):
        issues.add("schema_type", f"{path}.answer", "must be a string")
    if not isinstance(detail.get("correct"), bool):
        issues.add("schema_type", f"{path}.correct", "must be boolean")
    return detail


def _validate_selection(
    selection: Any,
    *,
    condition: str,
    items: Sequence[Mapping[str, Any]],
    path: str,
    issues: IssueCollector,
) -> None:
    record = _mapping(selection, path, issues)
    if record is None:
        return
    _check_key_set(record, SELECTION_KEYS, path, issues)
    scores: list[float] = []
    score_complete = True
    for item in items:
        score_map = item.get("scores")
        score = score_map.get(condition) if isinstance(score_map, Mapping) else None
        if not _finite_number(score):
            score_complete = False
            break
        scores.append(float(score))
    expected_indices = (
        sorted(range(len(items)), key=lambda index: (-scores[index], index))[: min(2, len(items))]
        if score_complete
        else None
    )
    selected_indices = record.get("selected_indices")
    if not isinstance(selected_indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in selected_indices
    ):
        issues.add("schema_type", f"{path}.selected_indices", "must be an integer array")
        valid_indices: list[int] | None = None
    else:
        valid_indices = selected_indices
        if expected_indices is not None:
            _require_exact(
                selected_indices,
                expected_indices,
                f"{path}.selected_indices",
                issues,
                "selection_not_reproducible_from_scores",
            )
        if any(index < 0 or index >= len(items) for index in selected_indices):
            issues.add("selection_index_out_of_range", f"{path}.selected_indices", "index out of range")

    if valid_indices is not None and all(0 <= index < len(items) for index in valid_indices):
        expected_uids = [items[index].get("uid") for index in valid_indices]
        expected_concepts = [items[index].get("concept") for index in valid_indices]
        # FrozenRecall.evaluate_sets canonicalizes non-oracle prompt memory by
        # sorted candidate index, whereas selection_record preserves score-rank
        # order.  Oracle QA comes from references(), which preserves its oracle
        # set order.  These are intentional distinct fields in the raw schema.
        expected_qa_concepts = (
            expected_concepts
            if condition == "oracle"
            else [items[index].get("concept") for index in sorted(set(valid_indices))]
        )
        positives = [
            index for index, item in enumerate(items) if item.get("label") == "load_bearing"
        ]
        chosen = set(valid_indices)
        selected_positive = [index for index in positives if index in chosen]
        expected_fields = {
            "selected_candidate_uids": expected_uids,
            "selected_concepts": expected_concepts,
            "contains_load_bearing": bool(selected_positive),
            "contains_all_load_bearing": bool(positives)
            and len(selected_positive) == len(positives),
            "selected_load_bearing": len(selected_positive),
            "total_load_bearing": len(positives),
            "load_bearing_recall": len(selected_positive) / len(positives) if positives else None,
        }
        for key, expected in expected_fields.items():
            _require_exact(
                record.get(key),
                expected,
                f"{path}.{key}",
                issues,
                "selection_record_inconsistent",
            )
        qa = _validate_answer(
            record.get("qa"), f"{path}.qa", issues, expected_keys=QA_KEYS
        )
        if qa is not None:
            _require_exact(
                qa.get("selected_concepts"),
                expected_qa_concepts,
                f"{path}.qa.selected_concepts",
                issues,
                "qa_selection_inconsistent",
            )


def _validate_episode_rows(
    payload: Mapping[str, Any],
    contract: SourceContract,
    side: str,
    config: Mapping[str, Any] | None,
    by_episode: Mapping[str, list[Mapping[str, Any]]],
    issues: IssueCollector,
) -> list[Mapping[str, Any]]:
    raw_episodes = _array(payload.get("per_episode"), f"{side}.per_episode", issues)
    if raw_episodes is None:
        return []
    if len(raw_episodes) != contract.n_episodes:
        issues.add(
            "episode_count_mismatch",
            f"{side}.per_episode",
            f"expected {contract.n_episodes} episodes, got {len(raw_episodes)}",
        )
    rows: list[Mapping[str, Any]] = []
    seen_uids: set[str] = set()
    for index, raw_episode in enumerate(raw_episodes):
        path = f"{side}.per_episode[{index}]"
        episode = _mapping(raw_episode, path, issues)
        if episode is None:
            continue
        rows.append(episode)
        _check_key_set(
            episode,
            EPISODE_REQUIRED_KEYS,
            path,
            issues,
            allow=EPISODE_OPTIONAL_KEYS,
        )
        expected_uid = f"{contract.source}:episode:{index:06d}"
        _require_exact(episode.get("uid"), expected_uid, f"{path}.uid", issues, "episode_order_mismatch")
        _require_exact(
            episode.get("source_episode"), index, f"{path}.source_episode", issues, "episode_order_mismatch"
        )
        _require_exact(episode.get("source"), contract.source, f"{path}.source", issues, "source_mismatch")
        uid = episode.get("uid")
        if isinstance(uid, str):
            if uid in seen_uids:
                issues.add("duplicate_episode_uid", f"{path}.uid", f"duplicate UID {uid!r}")
            seen_uids.add(uid)
        for key in ("probe_question", "gold_answer"):
            if not isinstance(episode.get(key), str):
                issues.add("schema_type", f"{path}.{key}", "must be a string")
        items = by_episode.get(expected_uid, [])
        if not items:
            issues.add("episode_without_items", path, "episode has no matching per-item rows")

        policies = _mapping(episode.get("policies"), f"{path}.policies", issues)
        if policies is not None:
            shared_order = [key for key in policies if key in SHARED_CONDITIONS]
            _require_exact(
                shared_order,
                list(SHARED_CONDITIONS),
                f"{path}.policies[shared]",
                issues,
                "shared_policy_order_mismatch",
            )
            for condition in SHARED_CONDITIONS:
                policy_path = f"{path}.policies.{condition}"
                policy = _mapping(policies.get(condition), policy_path, issues)
                if policy is None:
                    continue
                _check_key_set(policy, POLICY_KEYS, policy_path, issues)
                auc = policy.get("within_episode_auc")
                if auc is not None and not _finite_number(auc):
                    issues.add("invalid_within_episode_auc", f"{policy_path}.within_episode_auc", "must be null or finite numeric")
                selections = _mapping(policy.get("selections"), f"{policy_path}.selections", issues)
                if selections is not None:
                    _require_exact(
                        list(selections),
                        ["2"],
                        f"{policy_path}.selections.keys",
                        issues,
                        "selection_budget_mismatch",
                    )
                    _validate_selection(
                        selections.get("2"),
                        condition=condition,
                        items=items,
                        path=f"{policy_path}.selections.2",
                        issues=issues,
                    )

        refs = _mapping(episode.get("refs"), f"{path}.refs", issues)
        if refs is not None:
            _check_key_set(refs, REFERENCE_KEYS, f"{path}.refs", issues)
            oracle_ref = _validate_answer(
                refs.get("oracle@2"),
                f"{path}.refs.oracle@2",
                issues,
                expected_keys=ORACLE_REF_KEYS,
            )
            _validate_answer(
                refs.get("full_context"),
                f"{path}.refs.full_context",
                issues,
                expected_keys=NO_HARM_KEYS,
            )
            _validate_answer(
                refs.get("no_memory"),
                f"{path}.refs.no_memory",
                issues,
                expected_keys=NO_HARM_KEYS,
            )
            if oracle_ref is not None and policies is not None:
                oracle_policy = policies.get("oracle")
                oracle_selection = (
                    oracle_policy.get("selections", {}).get("2")
                    if isinstance(oracle_policy, Mapping)
                    else None
                )
                if isinstance(oracle_selection, Mapping):
                    for key in ("selected_indices", "selected_concepts"):
                        _require_exact(
                            oracle_ref.get(key),
                            oracle_selection.get(key),
                            f"{path}.refs.oracle@2.{key}",
                            issues,
                            "oracle_reference_inconsistent",
                        )
                    qa = oracle_selection.get("qa")
                    if isinstance(qa, Mapping):
                        for key in ("answer", "correct"):
                            _require_exact(
                                oracle_ref.get(key),
                                qa.get(key),
                                f"{path}.refs.oracle@2.{key}",
                                issues,
                                "oracle_reference_inconsistent",
                            )

        skip_no_harm = config.get("skip_no_harm") if config is not None else None
        no_harm_value = episode.get("no_harm_full_context")
        if skip_no_harm is True:
            if "no_harm_full_context" in episode:
                issues.add(
                    "skipped_no_harm_has_episode_data",
                    f"{path}.no_harm_full_context",
                    "field must be absent when config.skip_no_harm=true",
                )
            no_harm = None
        elif skip_no_harm is False:
            if "no_harm_full_context" not in episode:
                issues.add(
                    "no_harm_episode_data_missing",
                    f"{path}.no_harm_full_context",
                    "field is required when config.skip_no_harm=false",
                )
                no_harm = None
            else:
                no_harm = _mapping(
                    no_harm_value, f"{path}.no_harm_full_context", issues
                )
        else:
            no_harm = None
        if no_harm is not None:
            shared_order = [key for key in no_harm if key in NO_HARM_CONDITIONS]
            _require_exact(
                shared_order,
                list(NO_HARM_CONDITIONS),
                f"{path}.no_harm_full_context[shared]",
                issues,
                "shared_no_harm_order_mismatch",
            )
            for condition in NO_HARM_CONDITIONS:
                _validate_answer(
                    no_harm.get(condition),
                    f"{path}.no_harm_full_context.{condition}",
                    issues,
                    expected_keys=NO_HARM_KEYS,
                )

    extra_item_episodes = sorted(set(by_episode) - seen_uids)
    if extra_item_episodes:
        issues.add(
            "items_without_episode",
            f"{side}.per_item",
            f"item rows refer to unknown episodes: {extra_item_episodes[:5]}",
        )
    return rows


def _validate_payload(
    payload: Mapping[str, Any],
    contract: SourceContract,
    side: str,
    issues: IssueCollector,
) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    config = _validate_config(payload, contract, side, issues)
    items, by_episode = _validate_item_rows(payload, contract, side, config, issues)
    episodes = _validate_episode_rows(
        payload, contract, side, config, by_episode, issues
    )
    return config, items, episodes


def _numbers_close(old: Any, new: Any, *, abs_tol: float, rel_tol: float) -> bool:
    if not _finite_number(old) or not _finite_number(new):
        return False
    return math.isclose(float(old), float(new), abs_tol=abs_tol, rel_tol=rel_tol)


def _compare_semantic(
    old: Any,
    new: Any,
    *,
    path: str,
    category: str,
    issues: IssueCollector,
    counts: ComparisonCounter,
    abs_tol: float,
    rel_tol: float,
) -> None:
    if isinstance(old, bool) or isinstance(new, bool):
        counts.scalar(category)
        if type(old) is not bool or type(new) is not bool or old is not new:
            issues.add("semantic_mismatch", path, "boolean values differ", old=old, new=new, values_present=True)
        return
    if old is None or new is None:
        counts.scalar(category)
        if old is not None or new is not None:
            issues.add("semantic_mismatch", path, "null/value presence differs", old=old, new=new, values_present=True)
        return
    if isinstance(old, Mapping) or isinstance(new, Mapping):
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            issues.add("semantic_type_mismatch", path, "object/scalar types differ", old=old, new=new, values_present=True)
            return
        old_keys, new_keys = set(old), set(new)
        if old_keys != new_keys:
            issues.add(
                "semantic_key_mismatch",
                path,
                f"object keys differ; old-only={sorted(old_keys-new_keys)}, new-only={sorted(new_keys-old_keys)}",
            )
        for key in sorted(old_keys & new_keys):
            _compare_semantic(
                old[key],
                new[key],
                path=f"{path}.{key}",
                category=category,
                issues=issues,
                counts=counts,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        return
    if isinstance(old, list) or isinstance(new, list):
        if not isinstance(old, list) or not isinstance(new, list):
            issues.add("semantic_type_mismatch", path, "array/scalar types differ", old=old, new=new, values_present=True)
            return
        if len(old) != len(new):
            issues.add(
                "semantic_length_mismatch",
                path,
                f"array lengths differ: old={len(old)}, new={len(new)}",
            )
        for index, (old_value, new_value) in enumerate(zip(old, new)):
            _compare_semantic(
                old_value,
                new_value,
                path=f"{path}[{index}]",
                category=category,
                issues=issues,
                counts=counts,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        return
    if _finite_number(old) or _finite_number(new):
        counts.scalar(category)
        if isinstance(old, int) and isinstance(new, int):
            equal = old == new
        else:
            equal = _numbers_close(old, new, abs_tol=abs_tol, rel_tol=rel_tol)
        if not equal:
            issues.add("numeric_mismatch", path, "numeric values differ beyond tolerance", old=old, new=new, values_present=True)
        return
    counts.scalar(category)
    if type(old) is not type(new) or old != new:
        issues.add("semantic_mismatch", path, "values differ", old=old, new=new, values_present=True)


def _compare_payloads(
    old_payload: Mapping[str, Any],
    new_payload: Mapping[str, Any],
    old_config: Mapping[str, Any],
    new_config: Mapping[str, Any],
    old_items: Sequence[Mapping[str, Any]],
    new_items: Sequence[Mapping[str, Any]],
    old_episodes: Sequence[Mapping[str, Any]],
    new_episodes: Sequence[Mapping[str, Any]],
    *,
    compare_adapter_enabled_full_context: bool,
    abs_tol: float,
    rel_tol: float,
    issues: IssueCollector,
) -> ComparisonCounter:
    counts = ComparisonCounter()

    old_config_shared = dict(old_config)
    new_config_shared = dict(new_config)
    old_adapters = old_config_shared.pop("adapters", {})
    new_adapters = new_config_shared.pop("adapters", {})
    old_config_shared["adapters"] = {
        key: old_adapters.get(key) for key in SHARED_ADAPTERS
    }
    new_config_shared["adapters"] = {
        key: new_adapters.get(key) for key in SHARED_ADAPTERS
    }
    if not compare_adapter_enabled_full_context:
        # Decoupled old deliberately skipped this endpoint.  These knobs cannot
        # affect admission scoring, selection, or frozen-recall QA and therefore
        # are outside the reproducibility claim for that source pair.
        for key in ("skip_no_harm", "no_harm_batch_size"):
            old_config_shared.pop(key, None)
            new_config_shared.pop(key, None)
    counts.record("protocol", 1)
    _compare_semantic(
        old_config_shared,
        new_config_shared,
        path="config[shared]",
        category="protocol",
        issues=issues,
        counts=counts,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    counts.record("condition_order", 1)
    _compare_semantic(
        _condition_subsequence(old_payload.get("condition_order")),
        _condition_subsequence(new_payload.get("condition_order")),
        path="condition_order[shared]",
        category="condition_order",
        issues=issues,
        counts=counts,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

    if len(old_items) != len(new_items):
        issues.add("item_count_difference", "per_item", f"old={len(old_items)}, new={len(new_items)}")
    for index, (old_row, new_row) in enumerate(zip(old_items, new_items)):
        metadata_keys = sorted((set(old_row) | set(new_row)) - {"scores", "model_log_odds"})
        old_metadata = {key: old_row.get(key) for key in metadata_keys}
        new_metadata = {key: new_row.get(key) for key in metadata_keys}
        counts.record("item_identity_and_metadata")
        _compare_semantic(
            old_metadata,
            new_metadata,
            path=f"per_item[{index}].metadata",
            category="item_identity_and_metadata",
            issues=issues,
            counts=counts,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        old_scores = old_row.get("scores", {})
        new_scores = new_row.get("scores", {})
        for condition in SHARED_CONDITIONS:
            counts.record("admission_scores")
            _compare_semantic(
                old_scores.get(condition),
                new_scores.get(condition),
                path=f"per_item[{index}].scores.{condition}",
                category="admission_scores",
                issues=issues,
                counts=counts,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        old_log_odds = old_row.get("model_log_odds", {})
        new_log_odds = new_row.get("model_log_odds", {})
        expected_log_odds = list(SHARED_ADAPTERS)
        if old_config.get("original_verbal_source") == "model_forward":
            expected_log_odds.insert(0, "original")
        for condition in expected_log_odds:
            counts.record("model_log_odds")
            _compare_semantic(
                old_log_odds.get(condition),
                new_log_odds.get(condition),
                path=f"per_item[{index}].model_log_odds.{condition}",
                category="model_log_odds",
                issues=issues,
                counts=counts,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )

    if len(old_episodes) != len(new_episodes):
        issues.add("episode_count_difference", "per_episode", f"old={len(old_episodes)}, new={len(new_episodes)}")
    for index, (old_episode, new_episode) in enumerate(zip(old_episodes, new_episodes)):
        excluded = {"policies", "refs", "no_harm_full_context"}
        metadata_keys = sorted((set(old_episode) | set(new_episode)) - excluded)
        counts.record("episode_identity_and_metadata")
        _compare_semantic(
            {key: old_episode.get(key) for key in metadata_keys},
            {key: new_episode.get(key) for key in metadata_keys},
            path=f"per_episode[{index}].metadata",
            category="episode_identity_and_metadata",
            issues=issues,
            counts=counts,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        old_policies = old_episode.get("policies", {})
        new_policies = new_episode.get("policies", {})
        for condition in SHARED_CONDITIONS:
            old_policy = old_policies.get(condition, {})
            new_policy = new_policies.get(condition, {})
            counts.record("within_episode_auc")
            _compare_semantic(
                old_policy.get("within_episode_auc"),
                new_policy.get("within_episode_auc"),
                path=f"per_episode[{index}].policies.{condition}.within_episode_auc",
                category="within_episode_auc",
                issues=issues,
                counts=counts,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
            old_selection = old_policy.get("selections", {}).get("2", {})
            new_selection = new_policy.get("selections", {}).get("2", {})
            counts.record("selections")
            _compare_semantic(
                {key: value for key, value in old_selection.items() if key != "qa"},
                {key: value for key, value in new_selection.items() if key != "qa"},
                path=f"per_episode[{index}].policies.{condition}.selections.2",
                category="selections",
                issues=issues,
                counts=counts,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
            counts.record("selection_qa")
            _compare_semantic(
                old_selection.get("qa"),
                new_selection.get("qa"),
                path=f"per_episode[{index}].policies.{condition}.selections.2.qa",
                category="selection_qa",
                issues=issues,
                counts=counts,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )

        old_refs = old_episode.get("refs", {})
        new_refs = new_episode.get("refs", {})
        counts.record("oracle_references")
        _compare_semantic(
            old_refs.get("oracle@2"),
            new_refs.get("oracle@2"),
            path=f"per_episode[{index}].refs.oracle@2",
            category="oracle_references",
            issues=issues,
            counts=counts,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        counts.record("full_context_references")
        _compare_semantic(
            old_refs.get("full_context"),
            new_refs.get("full_context"),
            path=f"per_episode[{index}].refs.full_context",
            category="full_context_references",
            issues=issues,
            counts=counts,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        counts.record("no_memory_references")
        _compare_semantic(
            old_refs.get("no_memory"),
            new_refs.get("no_memory"),
            path=f"per_episode[{index}].refs.no_memory",
            category="no_memory_references",
            issues=issues,
            counts=counts,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )

        if compare_adapter_enabled_full_context:
            old_no_harm = old_episode.get("no_harm_full_context", {})
            new_no_harm = new_episode.get("no_harm_full_context", {})
            for condition in NO_HARM_CONDITIONS:
                counts.record("adapter_enabled_full_context")
                _compare_semantic(
                    old_no_harm.get(condition),
                    new_no_harm.get(condition),
                    path=f"per_episode[{index}].no_harm_full_context.{condition}",
                    category="adapter_enabled_full_context",
                    issues=issues,
                    counts=counts,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                )
    return counts


def _audit_source(
    contract: SourceContract,
    old_path: Path,
    new_path: Path,
    *,
    abs_tol: float,
    rel_tol: float,
) -> dict[str, Any]:
    old_issues = IssueCollector()
    new_issues = IssueCollector()
    comparison_issues = IssueCollector()
    old_payload, old_hash = _load_input(old_path, "old", old_issues)
    new_payload, new_hash = _load_input(new_path, "new", new_issues)

    old_config: Mapping[str, Any] | None = None
    new_config: Mapping[str, Any] | None = None
    old_items: list[Mapping[str, Any]] = []
    new_items: list[Mapping[str, Any]] = []
    old_episodes: list[Mapping[str, Any]] = []
    new_episodes: list[Mapping[str, Any]] = []
    if old_payload is not None:
        old_config, old_items, old_episodes = _validate_payload(
            old_payload, contract, "old", old_issues
        )
    if new_payload is not None:
        new_config, new_items, new_episodes = _validate_payload(
            new_payload, contract, "new", new_issues
        )

    comparison_counts = ComparisonCounter()
    comparison_executed = False
    old_skip_no_harm = (
        old_config.get("skip_no_harm") if old_config is not None else None
    )
    new_skip_no_harm = (
        new_config.get("skip_no_harm") if new_config is not None else None
    )
    compare_adapter_enabled_full_context = (
        old_skip_no_harm is False and new_skip_no_harm is False
    )
    if (
        old_payload is not None
        and new_payload is not None
        and old_config is not None
        and new_config is not None
        and old_issues.total == 0
        and new_issues.total == 0
    ):
        comparison_executed = True
        comparison_counts = _compare_payloads(
            old_payload,
            new_payload,
            old_config,
            new_config,
            old_items,
            new_items,
            old_episodes,
            new_episodes,
            compare_adapter_enabled_full_context=compare_adapter_enabled_full_context,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            issues=comparison_issues,
        )
    status = (
        "pass"
        if old_issues.total == new_issues.total == comparison_issues.total == 0
        and comparison_executed
        else "fail"
    )
    return {
        "status": status,
        "contract": {
            "source": contract.source,
            "n_episodes": contract.n_episodes,
            "n_items": contract.n_items,
            "budgets": list(EXPECTED_BUDGETS),
            "expected_old_skip_no_harm": contract.old_skip_no_harm,
            "expected_new_skip_no_harm": contract.new_skip_no_harm,
        },
        "old": {
            "path": str(old_path),
            "sha256": old_hash,
            "observed_counts": {
                "episodes": len(old_episodes) if old_payload is not None else None,
                "items": len(old_items) if old_payload is not None else None,
            },
            "validation": old_issues.report(),
        },
        "new": {
            "path": str(new_path),
            "sha256": new_hash,
            "observed_counts": {
                "episodes": len(new_episodes) if new_payload is not None else None,
                "items": len(new_items) if new_payload is not None else None,
            },
            "validation": new_issues.report(),
        },
        "comparison": {
            **comparison_issues.report(),
            "status": comparison_issues.report()["status"] if comparison_executed else "not-run",
            "executed": comparison_executed,
            "counts": comparison_counts.report(),
        },
        "full_context_scope": {
            "adapter_disabled_base_reference": {
                "status": "compared",
                "field": "refs.full_context",
                "note": "common frozen-base QA reference, not a per-condition no-harm endpoint",
            },
            "condition_specific_adapter_enabled": (
                {
                    "status": "compared",
                    "conditions": list(NO_HARM_CONDITIONS),
                    "field": "no_harm_full_context",
                }
                if compare_adapter_enabled_full_context
                else {
                    "status": "not-applicable",
                    "conditions": list(NO_HARM_CONDITIONS),
                    "reason": (
                        "at least one raw has config.skip_no_harm=true and contains "
                        "no comparable per-condition records"
                    ),
                    "old_skip_no_harm": old_skip_no_harm,
                    "new_skip_no_harm": new_skip_no_harm,
                }
            ),
            "structurally_not_applicable_conditions": ["workspace", "oracle"],
        },
    }


def audit_shared_condition_reproducibility(
    pairs: Mapping[str, tuple[Path, Path]],
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    contracts: Mapping[str, SourceContract] = SOURCE_CONTRACTS,
) -> dict[str, Any]:
    """Audit old/new source-isolated raws without modifying any input."""
    if not math.isfinite(abs_tol) or abs_tol < 0:
        raise ValueError("abs_tol must be finite and non-negative")
    if not math.isfinite(rel_tol) or rel_tol < 0:
        raise ValueError("rel_tol must be finite and non-negative")
    expected_sources = set(contracts)
    supplied_sources = set(pairs)
    global_issues = IssueCollector()
    if supplied_sources != expected_sources:
        global_issues.add(
            "source_set_mismatch",
            "pairs",
            f"expected sources {sorted(expected_sources)}, got {sorted(supplied_sources)}",
        )
    source_reports: dict[str, Any] = {}
    for source in sorted(expected_sources & supplied_sources):
        old_path, new_path = pairs[source]
        source_reports[source] = _audit_source(
            contracts[source],
            Path(old_path),
            Path(new_path),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    passed = global_issues.total == 0 and len(source_reports) == len(contracts) and all(
        report["status"] == "pass" for report in source_reports.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": AUDIT_NAME,
        "status": "pass" if passed else "fail",
        "scope": {
            "input_mode": "explicit-paths-only",
            "inputs_read_only": True,
            "directory_discovery": False,
            "shared_conditions": list(SHARED_CONDITIONS),
            "shared_adapter_conditions": list(SHARED_ADAPTERS),
            "sources": sorted(contracts),
            "nonshared_conditions_ignored": True,
        },
        "tolerance": {"absolute": abs_tol, "relative": rel_tol},
        "global_validation": global_issues.report(),
        "sources": source_reports,
    }


def write_report_exclusive(path: Path, report: Mapping[str, Any]) -> None:
    """Write exactly one new JSON file; never create parents or overwrite."""
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {path.parent}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoupled-old", required=True, type=Path)
    parser.add_argument("--decoupled-new", required=True, type=Path)
    parser.add_argument("--compositional-old", required=True, type=Path)
    parser.add_argument("--compositional-new", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL)
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.out
    # This preflight occurs before any input access; open("x") below closes the
    # race between this check and the eventual write.
    if output.exists():
        print(f"refusing to overwrite existing output: {output}", file=sys.stderr)
        return 2
    inputs = {
        args.decoupled_old,
        args.decoupled_new,
        args.compositional_old,
        args.compositional_new,
    }
    if output.resolve(strict=False) in {path.resolve(strict=False) for path in inputs}:
        print("output path must not equal an input path", file=sys.stderr)
        return 2
    if not output.parent.is_dir():
        print(f"output parent directory does not exist: {output.parent}", file=sys.stderr)
        return 2
    try:
        report = audit_shared_condition_reproducibility(
            {
                "decoupled": (args.decoupled_old, args.decoupled_new),
                "compositional": (args.compositional_old, args.compositional_new),
            },
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
        )
        write_report_exclusive(output, report)
    except (ValueError, FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"audit failed before report completion: {exc}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
