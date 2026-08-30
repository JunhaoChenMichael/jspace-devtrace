from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "scripts" / "report_rlqa_three_seed.py"
SPEC = importlib.util.spec_from_file_location("report_rlqa_three_seed", MODULE)
assert SPEC and SPEC.loader
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)

EPISODES = 20


def _payload(*, seed_qa=(0.85, 0.90, 0.85), original_qa=0.35, auc_delta=0.06, qa_drop_pp=0.0):
    conditions = {
        "original": {
            "qa": {"2": {"accuracy": original_qa, "n_episodes": EPISODES}},
            "classification": {"pooled_auc": 0.50},
        }
    }
    paired = []
    for index, seed in enumerate(reporter.SEEDS):
        name = f"rl-qa-s{seed}"
        conditions[name] = {
            "qa": {"2": {"accuracy": seed_qa[index], "n_episodes": EPISODES}},
            "classification": {"pooled_auc": 0.50 + auc_delta},
        }
        paired.append(
            {
                "a": name,
                "b": "original",
                "direction": "a_minus_b",
                "pooled_auc_difference": {
                    "estimate": auc_delta,
                    "ci_95": [auc_delta - 0.02, auc_delta + 0.02],
                    "bootstrap_samples_effective": 4000,
                },
            }
        )
    per_episode = []
    for index in range(EPISODES):
        policies = {
            "original": {"selections": {"2": {"qa": {"correct": index < round(original_qa * EPISODES)}}}}
        }
        for pos, seed in enumerate(reporter.SEEDS):
            policies[f"rl-qa-s{seed}"] = {
                "selections": {"2": {"qa": {"correct": index < round(seed_qa[pos] * EPISODES)}}}
            }
        per_episode.append({"uid": f"e{index}", "source": "decoupled", "policies": policies})
    return {
        "metrics": {"by_spec": {"decoupled": {"conditions": conditions, "paired_auc_differences": paired}}},
        "no_harm": {
            "skipped": False,
            "summary": {
                "by_spec": {
                    "decoupled": {
                        "conditions": {
                            "original": {"accuracy": 0.10, "n_episodes": EPISODES},
                            **{
                                f"rl-qa-s{seed}": {
                                    "accuracy": 0.10 - qa_drop_pp / 100.0,
                                    "n_episodes": EPISODES,
                                }
                                for seed in reporter.SEEDS
                            },
                        },
                        "comparisons": [
                            {
                                "base": "original",
                                "adapter": f"rl-qa-s{seed}",
                                "adapter_minus_base_accuracy": -qa_drop_pp / 100.0,
                                "exact_mcnemar_p_value": 1.0,
                            }
                            for seed in reporter.SEEDS
                        ],
                    }
                }
            },
        },
        "mcnemar": {
            "by_spec": {
                "decoupled": {
                    "2": [
                        {"a": "original", "b": f"rl-qa-s{seed}", "a_only": 0, "b_only": 6,
                         "discordant": 6, "p_value": 0.01, "n": EPISODES}
                        for seed in reporter.SEEDS
                    ]
                }
            }
        },
        "per_episode": per_episode,
    }


def test_all_criteria_pass_when_every_seed_gains() -> None:
    summary = reporter.build(_payload(), draws=200)
    assert summary["decision"] == "PASS"
    assert summary["success_criteria"]["primary_mean_effect_ci_excludes_zero"] is True
    assert summary["shared_draw_mean_qa_effect"]["estimate_pp"] > 0
    assert summary["aggregate"]["qa_delta_pp"]["mean"] > 0
    mcnemar = summary["per_seed"][0]["exact_mcnemar"]
    assert mcnemar["p_value"] == 0.01
    # the evaluator orients the pair as original-vs-adapter; the reporter must
    # relabel the discordant counts rather than report them backwards
    assert mcnemar["adapter_only_correct"] == 6 and mcnemar["original_only_correct"] == 0


def test_one_negative_seed_fails_the_primary_criterion() -> None:
    summary = reporter.build(_payload(seed_qa=(0.85, 0.20, 0.85)), draws=200)
    assert summary["per_seed"][1]["checks"]["decoupled_qa_delta_positive"] is False
    assert summary["success_criteria"]["primary_every_seed_qa_delta_positive"] is False
    assert summary["decision"] == "NOT_PASS"


def test_no_harm_thresholds_are_enforced() -> None:
    summary = reporter.build(_payload(qa_drop_pp=3.0), draws=100)
    assert summary["success_criteria"]["no_harm_full_context_qa"] is False
    assert summary["decision"] == "NOT_PASS"

    workspace = {seed: {"w_rr_before": 0.65, "w_rr_after": 0.60, "w_rr_drop": 0.05} for seed in reporter.SEEDS}
    summary = reporter.build(_payload(), draws=100, workspace=workspace)
    assert summary["success_criteria"]["no_harm_workspace_w_rr"] is False


def test_flipped_paired_difference_orientation_is_normalised() -> None:
    payload = _payload()
    scope = payload["metrics"]["by_spec"]["decoupled"]
    scope["paired_auc_differences"] = [
        {
            "a": "original",
            "b": "rl-qa-s0",
            "pooled_auc_difference": {"estimate": -0.06, "ci_95": [-0.08, -0.04]},
        },
        *scope["paired_auc_differences"][1:],
    ]
    summary = reporter.build(payload, draws=50)
    assert summary["per_seed"][0]["admission_auc_delta"] == pytest.approx(0.06)
    assert summary["per_seed"][0]["admission_auc_delta_ci_95"] == pytest.approx([0.04, 0.08])


def test_render_lists_every_criterion_and_seed() -> None:
    summary = reporter.build(_payload(), draws=50)
    text = reporter.render(summary, gpu="NVIDIA A100-SXM4-80GB", recipe=None, lock=None)
    assert "three-seed NVIDIA A100-SXM4-80GB replication" in text
    assert "shared episode-cluster draws" in text
    assert text.count("| 0 |") >= 1 and text.count("| 1 |") >= 1 and text.count("| 2 |") >= 1
    assert "RL-W, Hybrid" in text
