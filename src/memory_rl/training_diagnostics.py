"""Strict, recomputable diagnostics for exact-budget selector RL training."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"selector diagnostic field {key!r} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"selector diagnostic field {key!r} must be finite")
    return value


def summarize_selector_window(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate a non-empty consecutive window of selector train records."""

    if not rows:
        raise ValueError("selector diagnostic window cannot be empty")
    steps = []
    unique_counts = []
    rewards = []
    reward_stds = []
    containment = []
    entropies = []
    normalized_entropies = []
    kls = []
    gradients = []
    yes_probabilities = []
    mixed_qa = 0
    mixed_containment = 0

    for row in rows:
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError("selector diagnostic step must be a positive integer")
        steps.append(step)
        unique = _number(row, "number_unique_selected_sets")
        if unique < 1 or not unique.is_integer():
            raise ValueError("number_unique_selected_sets must be a positive integer")
        unique_counts.append(int(unique))
        rewards.append(_number(row, "reward_mean"))
        reward_stds.append(_number(row, "reward_std"))
        containment.append(_number(row, "containment_rate"))
        entropies.append(_number(row, "policy_set_entropy"))
        normalized_entropies.append(_number(row, "normalized_policy_set_entropy"))
        kls.append(_number(row, "kl"))
        gradients.append(_number(row, "grad_norm"))
        if not isinstance(row.get("mixed_QA_reward_group"), bool):
            raise ValueError("mixed_QA_reward_group must be boolean")
        if not isinstance(row.get("mixed_containment_group"), bool):
            raise ValueError("mixed_containment_group must be boolean")
        mixed_qa += int(row["mixed_QA_reward_group"])
        mixed_containment += int(row["mixed_containment_group"])
        raw_yes = row.get("yes_probabilities")
        if not isinstance(raw_yes, list) or not raw_yes:
            raise ValueError("yes_probabilities must be a non-empty list")
        yes_probabilities.extend(
            _number({"value": value}, "value") for value in raw_yes
        )

    if steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError("selector diagnostic rows must be consecutive")
    group_count = len(rows)
    return {
        "start_step": steps[0],
        "end_step": steps[-1],
        "groups": group_count,
        "mean_training_reward": statistics.fmean(rewards),
        "mean_reward_std": statistics.fmean(reward_stds),
        "mixed_QA_reward_groups": mixed_qa,
        "mixed_QA_reward_groups_fraction": mixed_qa / group_count,
        "groups_with_reward_std_gt_zero": sum(value > 0.0 for value in reward_stds),
        "groups_with_reward_std_gt_zero_fraction": sum(
            value > 0.0 for value in reward_stds
        )
        / group_count,
        "median_unique_selected_sets": float(statistics.median(unique_counts)),
        "mean_unique_selected_sets": statistics.fmean(unique_counts),
        "identical_set_groups": sum(value == 1 for value in unique_counts),
        "identical_set_groups_fraction": sum(value == 1 for value in unique_counts)
        / group_count,
        "mean_containment_rate": statistics.fmean(containment),
        "mixed_containment_groups": mixed_containment,
        "mixed_containment_groups_fraction": mixed_containment / group_count,
        "mean_policy_set_entropy": statistics.fmean(entropies),
        "mean_normalized_policy_set_entropy": statistics.fmean(normalized_entropies),
        "yes_probability_count": len(yes_probabilities),
        "yes_probability_mean": statistics.fmean(yes_probabilities),
        "yes_probability_std": statistics.pstdev(yes_probabilities),
        "yes_probability_min": min(yes_probabilities),
        "yes_probability_max": max(yes_probabilities),
        "mean_KL": statistics.fmean(kls),
        "max_KL": max(kls),
        "mean_gradient_norm": statistics.fmean(gradients),
        "nonzero_gradient_groups_fraction": sum(value > 0.0 for value in gradients)
        / group_count,
    }


def summarize_selector_training(
    rows: Sequence[Mapping[str, Any]], *, window_size: int = 25
) -> dict[str, Any]:
    """Return whole-run and fixed-window diagnostics for selector RL."""

    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if not rows:
        raise ValueError("selector training diagnostics require at least one row")
    ordered = sorted(rows, key=lambda row: row.get("step", -1))
    overall = summarize_selector_window(ordered)
    windows = [
        summarize_selector_window(ordered[start : start + window_size])
        for start in range(0, len(ordered), window_size)
    ]
    return {"window_size": window_size, "overall": overall, "windows": windows}


__all__ = ["summarize_selector_training", "summarize_selector_window"]
