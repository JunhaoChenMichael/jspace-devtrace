from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_rl.training_diagnostics import (  # noqa: E402
    summarize_selector_training,
    summarize_selector_window,
)


def _row(step: int, *, unique: int, mixed: bool, containment: float) -> dict:
    return {
        "step": step,
        "reward_mean": 0.5,
        "reward_std": 0.5 if mixed else 0.0,
        "number_unique_selected_sets": unique,
        "mixed_QA_reward_group": mixed,
        "containment_rate": containment,
        "mixed_containment_group": 0.0 < containment < 1.0,
        "policy_set_entropy": 1.2,
        "normalized_policy_set_entropy": 0.8,
        "yes_probabilities": [0.2, 0.5, 0.8],
        "kl": 0.01 * step,
        "grad_norm": 0.3,
    }


def test_selector_diagnostics_report_reward_diversity_and_fixed_windows():
    rows = [
        _row(1, unique=3, mixed=True, containment=0.5),
        _row(2, unique=1, mixed=False, containment=1.0),
        _row(3, unique=2, mixed=True, containment=0.0),
    ]
    result = summarize_selector_training(rows, window_size=2)
    overall = result["overall"]
    assert overall["mixed_QA_reward_groups_fraction"] == pytest.approx(2 / 3)
    assert overall["median_unique_selected_sets"] == 2
    assert overall["identical_set_groups_fraction"] == pytest.approx(1 / 3)
    assert overall["mean_containment_rate"] == pytest.approx(0.5)
    assert overall["yes_probability_count"] == 9
    assert [(row["start_step"], row["end_step"]) for row in result["windows"]] == [
        (1, 2),
        (3, 3),
    ]


def test_selector_diagnostic_rejects_gaps_and_bad_probabilities():
    with pytest.raises(ValueError, match="consecutive"):
        summarize_selector_window(
            [_row(1, unique=2, mixed=True, containment=0.5), _row(3, unique=2, mixed=True, containment=0.5)]
        )
    bad = _row(1, unique=2, mixed=True, containment=0.5)
    bad["yes_probabilities"] = []
    with pytest.raises(ValueError, match="yes_probabilities"):
        summarize_selector_window([bad])
