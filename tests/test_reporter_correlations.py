from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_rl.reporter_correlations import (
    compare_reporter_correlations,
    summarize_reporter_correlations,
)


def _rows():
    return [
        {
            "episode_id": "explicit:episode:0",
            "candidate_id": "explicit:episode:0:candidate:0",
            "source": "explicit",
            "v_rl": 0.1,
            "w_ref": 0.1,
            "y_utility": 0,
        },
        {
            "episode_id": "explicit:episode:0",
            "candidate_id": "explicit:episode:0:candidate:1",
            "source": "explicit",
            "v_rl": 0.9,
            "w_ref": 0.9,
            "y_utility": 1,
        },
        {
            "episode_id": "evoked:episode:1",
            "candidate_id": "evoked:episode:1:candidate:0",
            "source": "evoked",
            "v_rl": 0.2,
            "w_ref": 0.2,
            "y_utility": 0,
        },
        {
            "episode_id": "evoked:episode:1",
            "candidate_id": "evoked:episode:1:candidate:1",
            "source": "evoked",
            "v_rl": 0.8,
            "w_ref": 0.8,
            "y_utility": 1,
        },
    ]


def test_reporter_correlations_cluster_bootstrap_and_source_breakdown():
    result = summarize_reporter_correlations(
        _rows(), bootstrap_samples=100, bootstrap_seed=7
    )

    assert result["n_episodes"] == 2
    assert result["n_candidates"] == 4
    assert result["utility_auc"] == 1.0
    assert result["yes_rate"] == 0.5
    assert set(result["by_source"]) == {"explicit", "evoked"}
    for method in ("pearson", "spearman"):
        workspace = result["targets"]["w_ref"][method]["pooled_candidates"]
        assert workspace["estimate"] == pytest.approx(1.0)
        assert workspace["ci_95"] == pytest.approx([1.0, 1.0])
        for target in ("w_ref", "y_utility"):
            pooled = result["targets"][target][method]["pooled_candidates"]
            within = result["targets"][target][method]["equal_episode_mean"]
            assert pooled["estimate"] > 0.8
            assert pooled["ci_95"][0] > 0.0
            assert pooled["bootstrap_samples_effective"] == 100
            assert within["estimate"] == pytest.approx(1.0)
            assert within["n_valid_episodes"] == 2
            assert within["n_excluded_episodes"] == 0

    assert result == summarize_reporter_correlations(
        _rows(), bootstrap_samples=100, bootstrap_seed=7
    )


def test_reporter_correlations_record_constant_scores_as_null_not_zero():
    rows = _rows()
    for row in rows:
        row["v_rl"] = 0.5
    result = summarize_reporter_correlations(rows, bootstrap_samples=20)

    assert result["utility_auc"] == 0.5
    for target in ("w_ref", "y_utility"):
        for method in ("pearson", "spearman"):
            pooled = result["targets"][target][method]["pooled_candidates"]
            assert pooled["estimate"] is None
            assert pooled["status"] == "constant_v_rl"
            assert pooled["ci_95"] == [None, None]
            assert pooled["bootstrap_samples_effective"] == 0


def test_reporter_correlations_reject_malformed_or_duplicate_rows():
    duplicate = _rows()
    duplicate[1]["candidate_id"] = duplicate[0]["candidate_id"]
    with pytest.raises(ValueError, match="duplicate reporter candidate_id"):
        summarize_reporter_correlations(duplicate, bootstrap_samples=0)

    invalid = copy.deepcopy(_rows())
    invalid[0]["v_rl"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        summarize_reporter_correlations(invalid, bootstrap_samples=0)

    with pytest.raises(ValueError, match="non-negative integer"):
        summarize_reporter_correlations(_rows(), bootstrap_samples=-1)

    with pytest.raises(ValueError, match="definition"):
        summarize_reporter_correlations(
            _rows(), bootstrap_samples=0, v_rl_definition=""
        )


def test_reporter_correlations_allow_explicit_base_policy_provenance():
    definition = "base-model constrained P(Yes) at temperature 1"
    result = summarize_reporter_correlations(
        _rows(), bootstrap_samples=0, v_rl_definition=definition
    )
    assert result["definitions"]["v_rl"] == definition


def test_paired_reporter_correlations_use_shared_stratified_draws():
    aligned = _rows()
    inverted = copy.deepcopy(aligned)
    for row in inverted:
        row["v_rl"] = 1.0 - row["v_rl"]

    result = compare_reporter_correlations(
        {"aligned": aligned, "inverted": inverted},
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert result["conditions"] == ["aligned", "inverted"]
    assert result["n_episodes"] == 2
    assert result["n_candidates"] == 4
    assert result["bootstrap"]["shared_draws_across_conditions"] is True
    pair = result["pairs"][0]
    assert pair["direction"] == "a_minus_b"
    aligned_point = summarize_reporter_correlations(aligned, bootstrap_samples=0)
    inverted_point = summarize_reporter_correlations(inverted, bootstrap_samples=0)
    for target in ("w_ref", "y_utility"):
        for method in ("pearson", "spearman"):
            for estimand in ("pooled_candidates", "equal_episode_mean"):
                delta = pair["targets"][target][method][estimand]
                expected = (
                    aligned_point["targets"][target][method][estimand]["estimate"]
                    - inverted_point["targets"][target][method][estimand]["estimate"]
                )
                assert delta["estimate"] == pytest.approx(expected)
                assert delta["ci_95"] == pytest.approx([expected, expected])
                assert delta["bootstrap_samples_effective"] == 100

    assert result == compare_reporter_correlations(
        {"aligned": aligned, "inverted": inverted},
        bootstrap_samples=100,
        bootstrap_seed=7,
    )


def test_paired_reporter_correlations_reject_mismatched_candidate_panel_or_targets():
    missing = copy.deepcopy(_rows()[:-1])
    with pytest.raises(ValueError, match="different candidate panel"):
        compare_reporter_correlations(
            {"reference": _rows(), "missing": missing}, bootstrap_samples=0
        )

    changed = copy.deepcopy(_rows())
    changed[0]["w_ref"] = 0.25
    with pytest.raises(ValueError, match="changes immutable w_ref"):
        compare_reporter_correlations(
            {"reference": _rows(), "changed": changed}, bootstrap_samples=0
        )
