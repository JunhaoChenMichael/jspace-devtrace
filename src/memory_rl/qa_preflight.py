"""Pure statistics and gate logic for the Stage-B0 QA reward preflight."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _fraction(count: int, total: int) -> float:
    if total <= 0:
        raise ValueError("a proportion requires at least one observation")
    return count / total


def summarize_group(samples: Sequence[Mapping[str, object]]) -> dict:
    """Summarize one episode's fixed-size group of sampled memory sets."""

    rows = list(samples)
    if not rows:
        raise ValueError("a QA preflight group cannot be empty")
    episode_ids = {row.get("episode_id") for row in rows}
    if len(episode_ids) != 1 or None in episode_ids:
        raise ValueError("all samples in a group must share one episode_id")

    canonical_sets: list[tuple[str, ...]] = []
    rewards: list[float] = []
    containment: list[bool] = []
    for index, row in enumerate(rows):
        selected = row.get("selected_set")
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"samples[{index}].selected_set must be a non-empty list")
        if not all(isinstance(value, str) and value for value in selected):
            raise ValueError(f"samples[{index}].selected_set contains an invalid ID")
        if len(set(selected)) != len(selected):
            raise ValueError(f"samples[{index}].selected_set contains duplicates")
        canonical_sets.append(tuple(sorted(selected)))

        reward = _finite_number(row.get("QA_reward"), f"samples[{index}].QA_reward")
        if reward not in (0.0, 1.0):
            raise ValueError("QA_reward must be binary")
        correct = row.get("QA_correct")
        if not isinstance(correct, bool) or reward != float(correct):
            raise ValueError("QA_reward must equal float(QA_correct)")
        rewards.append(reward)

        contains = row.get("contains_load_bearing")
        if not isinstance(contains, bool):
            raise ValueError("contains_load_bearing must be boolean")
        containment.append(contains)

    mean_reward = statistics.fmean(rewards)
    reward_std = math.sqrt(
        statistics.fmean((reward - mean_reward) ** 2 for reward in rewards)
    )
    unique_sets = len(set(canonical_sets))
    return {
        "episode_id": next(iter(episode_ids)),
        "group_size": len(rows),
        "number_unique_selected_sets": unique_sets,
        "fraction_unique_selected_sets": unique_sets / len(rows),
        "mean_QA_reward": mean_reward,
        "QA_reward_std": reward_std,
        "mixed_QA_reward_group": len(set(rewards)) > 1,
        "mixed_containment_group": len(set(containment)) > 1,
    }


def classify_gate_b0(
    mixed_qa_reward_fraction: float,
    median_unique_selected_sets: float,
) -> dict:
    """Apply the continuation plan's GREEN/AMBER/RED B0 decision rule.

    The plan leaves one edge case implicit: mixed reward can exceed 40% while
    selection diversity remains below four sets.  That case is conservatively
    AMBER because it fails the conjunction required for GREEN but is not sparse
    enough for RED.
    """

    mixed = _finite_number(mixed_qa_reward_fraction, "mixed QA reward fraction")
    median_unique = _finite_number(
        median_unique_selected_sets, "median unique selected sets"
    )
    if not 0.0 <= mixed <= 1.0 or median_unique < 0.0:
        raise ValueError("gate inputs are outside their valid ranges")

    if mixed >= 0.40 and median_unique >= 4.0:
        status = "GREEN"
        next_action = "run RL-QA, then run Hybrid"
    elif mixed < 0.20:
        status = "RED"
        next_action = (
            "binary QA reward is too sparse for the primary experiment; "
            "optionally run a short RL-QA diagnostic, then proceed to Hybrid"
        )
    else:
        status = "AMBER"
        next_action = (
            "run RL-QA as a diagnostic and run Hybrid immediately afterward; "
            "do not spend a large QA-only hyperparameter budget"
        )

    reasons = []
    if mixed < 0.40:
        reasons.append("mixed QA reward groups are below the 40% GREEN threshold")
    if median_unique < 4.0:
        reasons.append("median unique selected sets are below the GREEN threshold of 4")
    if not reasons:
        reasons.append("both GREEN thresholds are satisfied")
    return {
        "status": status,
        "observed": {
            "mixed_QA_reward_groups_fraction": mixed,
            "median_unique_selected_sets": median_unique,
        },
        "thresholds": {
            "green_mixed_QA_reward_groups_fraction_min": 0.40,
            "green_median_unique_selected_sets_min": 4.0,
            "red_mixed_QA_reward_groups_fraction_max_exclusive": 0.20,
        },
        "reasons": reasons,
        "next_action": next_action,
    }


def summarize_preflight(
    samples: Sequence[Mapping[str, object]],
    groups: Sequence[Mapping[str, object]],
    references: Sequence[Mapping[str, object]],
) -> dict:
    """Build the complete B0 aggregate report from sample/group/reference rows."""

    sample_rows = list(samples)
    group_rows = list(groups)
    reference_rows = list(references)
    if not sample_rows or not group_rows or not reference_rows:
        raise ValueError("samples, groups, and references must all be non-empty")

    unique_counts = [
        _finite_number(row.get("number_unique_selected_sets"), "unique set count")
        for row in group_rows
    ]
    reward_stds = [
        _finite_number(row.get("QA_reward_std"), "QA reward std")
        for row in group_rows
    ]
    mixed_reward_count = sum(
        row.get("mixed_QA_reward_group") is True for row in group_rows
    )
    mixed_containment_count = sum(
        row.get("mixed_containment_group") is True for row in group_rows
    )
    diverse_count = sum(value >= 2 for value in unique_counts)
    positive_std_count = sum(value > 0.0 for value in reward_stds)
    group_count = len(group_rows)
    median_unique = float(statistics.median(unique_counts))
    mixed_fraction = _fraction(mixed_reward_count, group_count)

    retained = [row for row in sample_rows if row.get("contains_load_bearing") is True]
    not_retained = [
        row for row in sample_rows if row.get("contains_load_bearing") is False
    ]

    def conditional(rows: Sequence[Mapping[str, object]]) -> tuple[float | None, int, int]:
        correct = sum(row.get("QA_correct") is True for row in rows)
        return (correct / len(rows) if rows else None, len(rows), correct)

    p_retained, retained_n, retained_correct = conditional(retained)
    p_not, not_n, not_correct = conditional(not_retained)
    relationship_difference = (
        p_retained - p_not if p_retained is not None and p_not is not None else None
    )

    def reference_accuracy(key: str) -> tuple[float, int]:
        values = [row.get(key) for row in reference_rows]
        if not all(isinstance(value, bool) for value in values):
            raise ValueError(f"reference field {key} must be boolean")
        return sum(bool(value) for value in values) / len(values), len(values)

    oracle_accuracy, reference_n = reference_accuracy("oracle_QA_correct")
    full_accuracy, _ = reference_accuracy("full_context_QA_correct")
    no_memory_accuracy, _ = reference_accuracy("no_memory_QA_correct")
    qa_rewards = [
        _finite_number(row.get("QA_reward"), "sample QA reward")
        for row in sample_rows
    ]

    gate = classify_gate_b0(mixed_fraction, median_unique)
    return {
        "schema_version": 1,
        "counts": {
            "episodes": group_count,
            "groups": group_count,
            "samples": len(sample_rows),
            "references": reference_n,
        },
        "selection_diversity": {
            "groups_with_at_least_2_unique_sets": diverse_count,
            "groups_with_at_least_2_unique_sets_fraction": _fraction(
                diverse_count, group_count
            ),
            "groups_with_at_least_2_unique_sets_percent": 100.0
            * _fraction(diverse_count, group_count),
            "median_unique_selected_sets": median_unique,
            "mean_unique_selected_sets": statistics.fmean(unique_counts),
        },
        "reward_diversity": {
            "mixed_QA_reward_groups": mixed_reward_count,
            "mixed_QA_reward_groups_fraction": mixed_fraction,
            "mixed_QA_reward_groups_percent": 100.0 * mixed_fraction,
            "groups_with_QA_reward_std_gt_zero": positive_std_count,
            "groups_with_QA_reward_std_gt_zero_fraction": _fraction(
                positive_std_count, group_count
            ),
            "groups_with_QA_reward_std_gt_zero_percent": 100.0
            * _fraction(positive_std_count, group_count),
            "mean_within_group_QA_reward_std": statistics.fmean(reward_stds),
            "mean_QA_reward": statistics.fmean(qa_rewards),
        },
        "containment_diversity": {
            "mixed_containment_groups": mixed_containment_count,
            "mixed_containment_groups_fraction": _fraction(
                mixed_containment_count, group_count
            ),
            "mixed_containment_groups_percent": 100.0
            * _fraction(mixed_containment_count, group_count),
        },
        "containment_QA_relationship": {
            "retained_samples": retained_n,
            "retained_correct": retained_correct,
            "not_retained_samples": not_n,
            "not_retained_correct": not_correct,
            "P_QA_correct_given_load_bearing_retained": p_retained,
            "P_QA_correct_given_load_bearing_not_retained": p_not,
            "difference": relationship_difference,
        },
        "references": {
            "exploitable_episode_definition": (
                "oracle-selected memory produces a correct downstream answer"
            ),
            "exploitable_episodes": sum(
                row.get("oracle_QA_correct") is True for row in reference_rows
            ),
            "exploitable_fraction": oracle_accuracy,
            "oracle_QA_accuracy": oracle_accuracy,
            "full_context_QA_accuracy": full_accuracy,
            "no_memory_QA_accuracy": no_memory_accuracy,
        },
        "gate_b0": gate,
    }


def select_temperature(
    calibration_rows: Sequence[Mapping[str, object]],
    *,
    min_median_unique_sets: float = 4.0,
) -> tuple[float, str]:
    """Choose the lowest calibrated temperature meeting the diversity target."""

    rows = list(calibration_rows)
    if not rows:
        raise ValueError("temperature calibration cannot be empty")
    target = _finite_number(min_median_unique_sets, "minimum median unique sets")
    if target < 1.0:
        raise ValueError("minimum median unique sets must be >= 1")
    parsed = []
    for row in rows:
        temperature = _finite_number(row.get("temperature"), "temperature")
        median_unique = _finite_number(
            row.get("median_unique_selected_sets"), "median unique selected sets"
        )
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        parsed.append((temperature, median_unique))
    if len({temperature for temperature, _ in parsed}) != len(parsed):
        raise ValueError("temperature candidates must be unique")
    parsed.sort()
    for temperature, median_unique in parsed:
        if median_unique >= target:
            return temperature, "lowest candidate meeting the median diversity target"
    return parsed[-1][0], "no candidate met the target; selected the highest candidate"


__all__ = [
    "classify_gate_b0",
    "select_temperature",
    "summarize_group",
    "summarize_preflight",
]
