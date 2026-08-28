"""Auditable correlations between an admission reporter and fixed targets.

The reporting unit is a candidate, but uncertainty is clustered by episode so
that candidates sharing one context are never treated as independent.  This
module intentionally contains no model code: callers provide adapter-enabled
``P(Yes)`` values together with immutable candidate metadata.
"""

from __future__ import annotations

from collections import Counter
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np


TARGETS = ("w_ref", "y_utility")
METHODS = ("pearson", "spearman")


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-indexed average ranks, including exact-tie handling."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _correlation(
    left: Sequence[float], right: Sequence[float], method: str
) -> tuple[float | None, str]:
    if method not in METHODS:
        raise ValueError(f"unknown correlation method: {method}")
    if len(left) != len(right):
        raise ValueError("correlation inputs must have equal length")
    if len(left) < 2:
        return None, "insufficient_data"
    x = [float(value) for value in left]
    y = [float(value) for value in right]
    if not all(math.isfinite(value) for value in (*x, *y)):
        raise ValueError("correlation inputs must be finite")
    if max(x) == min(x):
        return None, "constant_v_rl"
    if max(y) == min(y):
        return None, "constant_target"
    if method == "spearman":
        x = _average_ranks(x)
        y = _average_ranks(y)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x)
        * sum((b - y_mean) ** 2 for b in y)
    )
    if denominator == 0.0:
        return None, "degenerate"
    # Floating-point roundoff can otherwise produce values infinitesimally
    # outside the mathematical [-1, 1] range.
    return max(-1.0, min(1.0, numerator / denominator)), "ok"


def _normalized_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for index, raw in enumerate(rows):
        where = f"reporter row {index}"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where} must be an object")
        episode_id = raw.get("episode_id")
        candidate_id = raw.get("candidate_id")
        source = raw.get("source")
        if not all(isinstance(value, str) and value for value in (episode_id, candidate_id, source)):
            raise ValueError(f"{where} needs non-empty episode_id, candidate_id, and source")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate reporter candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        numbers: dict[str, float] = {}
        for key in ("v_rl", "w_ref"):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{where}.{key} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{where}.{key} must be finite")
            numbers[key] = number
        if not 0.0 <= numbers["v_rl"] <= 1.0:
            raise ValueError(f"{where}.v_rl must lie in [0, 1]")
        utility = raw.get("y_utility")
        if utility not in (0, 1, False, True):
            raise ValueError(f"{where}.y_utility must be binary")
        normalized.append(
            {
                "episode_id": episode_id,
                "candidate_id": candidate_id,
                "source": source,
                "v_rl": numbers["v_rl"],
                "w_ref": numbers["w_ref"],
                "y_utility": int(utility),
            }
        )
    if not normalized:
        raise ValueError("reporter correlation rows must not be empty")
    episode_sources: dict[str, str] = {}
    for row in normalized:
        previous = episode_sources.setdefault(row["episode_id"], row["source"])
        if previous != row["source"]:
            raise ValueError("one episode_id cannot belong to multiple sources")
    return normalized


def _clusters(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
    return list(by_episode.values())


def _point_estimands(
    rows: Sequence[Mapping[str, Any]], target: str, method: str
) -> dict[str, dict[str, Any]]:
    pooled_value, pooled_status = _correlation(
        [float(row["v_rl"]) for row in rows],
        [float(row[target]) for row in rows],
        method,
    )
    within_values: list[float] = []
    excluded = Counter()
    for cluster in _clusters(rows):
        value, status = _correlation(
            [float(row["v_rl"]) for row in cluster],
            [float(row[target]) for row in cluster],
            method,
        )
        if value is None:
            excluded[status] += 1
        else:
            within_values.append(value)
    return {
        "pooled_candidates": {
            "estimate": pooled_value,
            "status": pooled_status,
            "n_candidates": len(rows),
        },
        "equal_episode_mean": {
            "estimate": (
                sum(within_values) / len(within_values) if within_values else None
            ),
            "status": "ok" if within_values else "no_valid_episodes",
            "n_valid_episodes": len(within_values),
            "n_excluded_episodes": sum(excluded.values()),
            "excluded_episode_reasons": dict(sorted(excluded.items())),
        },
    }


def _with_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    point: dict[str, dict[str, dict[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError("bootstrap samples must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    clusters = _clusters(rows)
    cluster_sources = [str(cluster[0]["source"]) for cluster in clusters]
    source_indices = {
        source: [
            index
            for index, observed_source in enumerate(cluster_sources)
            if observed_source == source
        ]
        for source in sorted(set(cluster_sources))
    }
    within_by_target_method: dict[tuple[str, str], list[float | None]] = {}
    for target in TARGETS:
        for method in METHODS:
            values = []
            for cluster in clusters:
                value, _ = _correlation(
                    [float(row["v_rl"]) for row in cluster],
                    [float(row[target]) for row in cluster],
                    method,
                )
                values.append(value)
            within_by_target_method[(target, method)] = values

    distributions: dict[tuple[str, str, str], list[float]] = {
        (target, method, estimand): []
        for target in TARGETS
        for method in METHODS
        for estimand in ("pooled_candidates", "equal_episode_mean")
    }
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        draw: list[int] = []
        for source in sorted(source_indices):
            indices = source_indices[source]
            draw.extend(
                indices[int(local_index)]
                for local_index in rng.integers(0, len(indices), size=len(indices))
            )
        sampled_rows = [row for index in draw for row in clusters[int(index)]]
        for target in TARGETS:
            for method in METHODS:
                pooled, _ = _correlation(
                    [float(row["v_rl"]) for row in sampled_rows],
                    [float(row[target]) for row in sampled_rows],
                    method,
                )
                if pooled is not None:
                    distributions[(target, method, "pooled_candidates")].append(pooled)
                within = [
                    within_by_target_method[(target, method)][int(index)]
                    for index in draw
                ]
                finite_within = [value for value in within if value is not None]
                if finite_within:
                    distributions[(target, method, "equal_episode_mean")].append(
                        sum(finite_within) / len(finite_within)
                    )

    for target in TARGETS:
        for method in METHODS:
            for estimand in ("pooled_candidates", "equal_episode_mean"):
                values = distributions[(target, method, estimand)]
                record = point[target][method][estimand]
                record["ci_95"] = (
                    np.percentile(values, [2.5, 97.5]).tolist()
                    if values
                    else [None, None]
                )
                record["bootstrap_samples_effective"] = len(values)
                record["probability_gt_zero"] = (
                    sum(value > 0.0 for value in values) / len(values)
                    if values
                    else None
                )
    return point


def _point_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        target: {
            method: _point_estimands(rows, target, method)
            for method in METHODS
        }
        for target in TARGETS
    }


def binary_utility_auc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Tie-aware candidate AUC of V_RL against the binary utility label."""

    positives = [float(row["v_rl"]) for row in rows if int(row["y_utility"]) == 1]
    negatives = [float(row["v_rl"]) for row in rows if int(row["y_utility"]) == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        (positive > negative) + 0.5 * (positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def within_episode_utility_auc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Equal-episode mean of tie-aware utility AUCs."""

    values = [binary_utility_auc(cluster) for cluster in _clusters(rows)]
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def summarize_reporter_correlations(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 4000,
    bootstrap_seed: int = 0,
    v_rl_definition: str = (
        "adapter-enabled P(Yes) over aggregated constrained No/Yes logits at temperature 1"
    ),
) -> dict[str, Any]:
    """Summarize reporter correlations with episode-cluster uncertainty."""

    if not isinstance(v_rl_definition, str) or not v_rl_definition:
        raise ValueError("v_rl definition must be a non-empty string")

    clean = _normalized_rows(rows)
    clusters = _clusters(clean)
    overall = _with_bootstrap(
        clean,
        _point_summary(clean),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    by_source = {}
    for source in sorted({str(row["source"]) for row in clean}):
        subset = [row for row in clean if row["source"] == source]
        by_source[source] = {
            "n_episodes": len(_clusters(subset)),
            "n_candidates": len(subset),
            "utility_auc": binary_utility_auc(subset),
            "targets": _point_summary(subset),
        }
    return {
        "schema_version": 1,
        "definitions": {
            "v_rl": v_rl_definition,
            "w_ref": "immutable raw W_rr from the matched frozen reference model",
            "y_utility": "1 iff candidate.label == load_bearing, else 0",
            "correlation_unit": "candidate",
            "uncertainty_unit": "episode",
        },
        "bootstrap": {
            "method": "episode_cluster_percentile",
            "samples_requested": bootstrap_samples,
            "seed": bootstrap_seed,
            "rng": "numpy.random.default_rng / PCG64",
            "confidence": 0.95,
            "stratified_by_source": True,
        },
        "n_episodes": len(clusters),
        "n_candidates": len(clean),
        "utility_auc": binary_utility_auc(clean),
        "yes_rate": sum(float(row["v_rl"]) >= 0.5 for row in clean) / len(clean),
        "targets": overall,
        "by_source": by_source,
    }


def _align_conditions(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Validate and align several reporters on one immutable candidate panel."""

    if not isinstance(condition_rows, Mapping) or len(condition_rows) < 2:
        raise ValueError("paired reporter comparison requires at least two conditions")
    names = list(condition_rows)
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("reporter condition names must be non-empty strings")

    normalized = {
        name: _normalized_rows(rows) for name, rows in condition_rows.items()
    }
    reference_name = names[0]
    reference = normalized[reference_name]
    candidate_order = [row["candidate_id"] for row in reference]
    immutable_keys = ("episode_id", "source", "w_ref", "y_utility")
    aligned = {reference_name: reference}
    for name in names[1:]:
        by_candidate = {row["candidate_id"]: row for row in normalized[name]}
        if set(by_candidate) != set(candidate_order):
            missing = sorted(set(candidate_order) - set(by_candidate))
            extra = sorted(set(by_candidate) - set(candidate_order))
            raise ValueError(
                f"condition {name!r} has a different candidate panel: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        rows = [by_candidate[candidate_id] for candidate_id in candidate_order]
        for reference_row, row in zip(reference, rows):
            for key in immutable_keys:
                if row[key] != reference_row[key]:
                    raise ValueError(
                        f"condition {name!r} changes immutable {key} for "
                        f"candidate {row['candidate_id']!r}"
                    )
        aligned[name] = rows
    return names, aligned


def compare_reporter_correlations(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_samples: int = 4000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Return paired correlation deltas using shared episode bootstrap draws.

    Conditions must cover the same candidate IDs and agree exactly on episode,
    source, ``W_ref``, and utility labels.  A draw is retained for a given
    estimand only when both conditions produce a defined correlation.
    """

    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 0
    ):
        raise ValueError("bootstrap samples must be a non-negative integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap seed must be an integer")

    names, aligned = _align_conditions(condition_rows)
    reference = aligned[names[0]]
    episode_order = list(dict.fromkeys(row["episode_id"] for row in reference))
    source_by_episode = {
        row["episode_id"]: row["source"] for row in reference
    }
    source_indices = {
        source: [
            index
            for index, episode_id in enumerate(episode_order)
            if source_by_episode[episode_id] == source
        ]
        for source in sorted(set(source_by_episode.values()))
    }
    rows_by_condition_episode = {
        name: {
            episode_id: [
                row for row in rows if row["episode_id"] == episode_id
            ]
            for episode_id in episode_order
        }
        for name, rows in aligned.items()
    }

    point = {name: _point_summary(rows) for name, rows in aligned.items()}
    pair_names = list(itertools.combinations(names, 2))
    estimands = ("pooled_candidates", "equal_episode_mean")
    distributions = {
        (left, right, target, method, estimand): []
        for left, right in pair_names
        for target in TARGETS
        for method in METHODS
        for estimand in estimands
    }

    rng = np.random.default_rng(bootstrap_seed)
    for _ in range(bootstrap_samples):
        draw: list[int] = []
        for source in sorted(source_indices):
            indices = source_indices[source]
            draw.extend(
                indices[int(local_index)]
                for local_index in rng.integers(0, len(indices), size=len(indices))
            )
        sampled = {}
        for name in names:
            sampled_rows = [
                row
                for index in draw
                for row in rows_by_condition_episode[name][episode_order[index]]
            ]
            sampled[name] = _point_summary(sampled_rows)
        for left, right in pair_names:
            for target in TARGETS:
                for method in METHODS:
                    for estimand in estimands:
                        left_value = sampled[left][target][method][estimand]["estimate"]
                        right_value = sampled[right][target][method][estimand]["estimate"]
                        if left_value is not None and right_value is not None:
                            distributions[
                                (left, right, target, method, estimand)
                            ].append(left_value - right_value)

    pairs = []
    for left, right in pair_names:
        targets = {}
        for target in TARGETS:
            targets[target] = {}
            for method in METHODS:
                targets[target][method] = {}
                for estimand in estimands:
                    left_value = point[left][target][method][estimand]["estimate"]
                    right_value = point[right][target][method][estimand]["estimate"]
                    values = distributions[(left, right, target, method, estimand)]
                    targets[target][method][estimand] = {
                        "estimate": (
                            left_value - right_value
                            if left_value is not None and right_value is not None
                            else None
                        ),
                        "status": (
                            "ok"
                            if left_value is not None and right_value is not None
                            else "undefined_condition_correlation"
                        ),
                        "ci_95": (
                            np.percentile(values, [2.5, 97.5]).tolist()
                            if values
                            else [None, None]
                        ),
                        "bootstrap_samples_effective": len(values),
                        "probability_gt_zero": (
                            sum(value > 0.0 for value in values) / len(values)
                            if values
                            else None
                        ),
                    }
        pairs.append(
            {
                "a": left,
                "b": right,
                "direction": "a_minus_b",
                "targets": targets,
            }
        )

    return {
        "schema_version": 1,
        "conditions": names,
        "n_episodes": len(episode_order),
        "n_candidates": len(reference),
        "bootstrap": {
            "method": "source_stratified_episode_cluster_percentile",
            "samples_requested": bootstrap_samples,
            "seed": bootstrap_seed,
            "rng": "numpy.random.default_rng / PCG64",
            "confidence": 0.95,
            "shared_draws_across_conditions": True,
        },
        "pairs": pairs,
    }


__all__ = [
    "binary_utility_auc",
    "compare_reporter_correlations",
    "summarize_reporter_correlations",
    "within_episode_utility_auc",
]
