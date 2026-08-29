from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "scripts" / "report_metacog_three_seed.py"
SPEC = importlib.util.spec_from_file_location("report_metacog_three_seed", MODULE)
assert SPEC and SPEC.loader
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def _rows(scale: float, episodes: int = 8) -> list[dict]:
    rows = []
    for episode in range(episodes):
        for index, label in enumerate(("load_bearing", "distractor", "filler")):
            rows.append(
                {
                    "episode": episode,
                    "candidate_index": index,
                    "concept": f"c{episode}{index}",
                    "label": label,
                    # load_bearing scores rise with `scale`; the others do not.
                    "V": (0.9 * scale if label == "load_bearing" else 0.1 + 0.01 * index),
                    "W_rr": 0.5 + 0.01 * index,
                }
            )
    return rows


def _condition(scale_before: float, scale_after: float, qa_drop: int = 0) -> dict:
    before, after = _rows(scale_before), _rows(scale_after)
    episodes = sorted({row["episode"] for row in before})
    return {
        "verbal": {
            "before": {"pooled_auc": 0.34, "within_episode_auc": 0.3},
            "after": {"pooled_auc": 0.66, "within_episode_auc": 0.6},
            "delta_pooled_auc": 0.32,
            "delta_within_episode_auc": 0.3,
            "paired_episode_bootstrap": {
                "after_minus_before": {"ci_95": [0.24, 0.39], "probability_gt_zero": 1.0}
            },
        },
        "workspace": {
            "before": {"pooled_auc": 0.65, "within_episode_auc": 0.6},
            "after": {"pooled_auc": 0.651, "within_episode_auc": 0.6},
            "delta_pooled_auc": 0.001,
            "delta_within_episode_auc": 0.0,
            "paired_episode_bootstrap": {
                "after_minus_before": {"ci_95": [-0.01, 0.01], "probability_gt_zero": 0.5}
            },
        },
        "full_context_qa": {
            "per_episode": {
                "before": [{"episode": e, "correct": True} for e in episodes],
                "after": [
                    {"episode": e, "correct": e >= qa_drop} for e in episodes
                ],
            }
        },
        "candidate_identity_sha256": "identity",
        "per_item": {"before": before, "after": after},
    }


def _payload(seed: int, *, scale_after: float = 1.0, qa_drop: int = 0) -> dict:
    return {
        "stage": "M1_OOD",
        "attempt_id": f"{seed}" * 32,
        "model": "Qwen/Qwen3-8B",
        "model_revision": "b" * 40,
        "tokenizer_revision": "b" * 40,
        "checkpoint_tree_sha256": f"{seed}" * 64,
        "conditions": {
            "decoupled": _condition(0.2, scale_after, qa_drop),
            "compositional": _condition(0.2, scale_after, qa_drop),
        },
    }


def test_three_green_seeds_replicate_with_shared_draws() -> None:
    summary = reporter.build({s: _payload(s) for s in (0, 1, 2)}, draws=64)
    assert summary["replication_decision"] == "GREEN"
    assert summary["strong_green_seeds"] == [0, 1, 2]
    shared = summary["shared_draw_mean_delta_v_decoupled"]
    assert shared["shared_draws_across_seeds"] is True
    assert shared["draws_effective"] <= 64
    assert shared["ci_95"][0] <= shared["estimate"] <= shared["ci_95"][1]
    assert summary["aggregate_decoupled"]["delta_v"]["sample_sd"] == pytest.approx(0.0)


def test_one_failing_seed_blocks_the_replication() -> None:
    payloads = {s: _payload(s) for s in (0, 1, 2)}
    # Seed 2 loses more than two points of full-context QA.
    payloads[2] = _payload(2, qa_drop=3)
    summary = reporter.build(payloads, draws=32)
    assert summary["per_seed"][2]["checks"]["qa_drop_at_most_2pp"] is False
    assert summary["per_seed"][0]["gate"] == "GREEN"
    assert summary["replication_decision"] == "NOT_GREEN"


def test_duplicate_checkpoints_and_partial_seed_sets_are_refused() -> None:
    with pytest.raises(reporter.ReportError, match="seeds 0, 1 and 2"):
        reporter.build({0: _payload(0), 1: _payload(1)}, draws=8)

    payloads = {s: _payload(s) for s in (0, 1, 2)}
    payloads[1]["checkpoint_tree_sha256"] = payloads[0]["checkpoint_tree_sha256"]
    with pytest.raises(reporter.ReportError, match="same checkpoint"):
        reporter.build(payloads, draws=8)

    payloads = {s: _payload(s) for s in (0, 1, 2)}
    payloads[1]["conditions"]["decoupled"]["candidate_identity_sha256"] = "other"
    with pytest.raises(reporter.ReportError, match="different decoupled items"):
        reporter.build(payloads, draws=8)


def test_render_reports_every_seed_and_the_shared_interval() -> None:
    summary = reporter.build({s: _payload(s) for s in (0, 1, 2)}, draws=32)
    text = reporter.render(
        summary, gpu="NVIDIA A100-SXM4-80GB", sources={s: f"seed{s}/ood/result.json" for s in (0, 1, 2)}
    )
    assert "three-seed NVIDIA A100-SXM4-80GB replication" in text
    assert "shared episode-cluster draws" in text
    for seed in (0, 1, 2):
        assert f"seed{seed}/ood/result.json" in text
