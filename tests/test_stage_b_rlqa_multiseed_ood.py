from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import stage_b_rlqa_multiseed_ood as stage_b


SFT_PATH = "test/sft/best-step-300"
RL_PATHS = {
    f"rl-qa-s{seed}-k2": (
        "test/formal_rl-qa_Qwen2.5-7B-Instruct_rank_continuous_"
        f"split0_s{seed}_beta0p03_k2_lq1_lw0/best-step-{100 + seed}"
    )
    for seed in stage_b.SEEDS
}
PROTOCOL = stage_b.Protocol(
    model="test/model",
    conditions=stage_b.CONDITIONS,
    seeds=stage_b.SEEDS,
    budget=2,
    workspace_top_k=2,
    dtype="bfloat16",
    admission_batch_size=16,
    qa_batch_size=1,
    no_harm_batch_size=1,
    max_length=2048,
    max_new_tokens=64,
    bootstrap_samples=32,
    bootstrap_seed=0,
    sources=(
        stage_b.SourceContract(
            "decoupled", "test/dec.json", "test/dec-battery.json", 4, 12, 2
        ),
        stage_b.SourceContract(
            "compositional", "test/comp.json", "test/comp-battery.json", 3, 9
        ),
    ),
    sft_adapter=SFT_PATH,
)


def _lock() -> dict:
    return {
        "schema_version": 1,
        "lock_status": "locked",
        "authorization": {"pre_ood": True},
        "ood_state_at_lock": {
            "results_inspected": False,
            "completed_artifact_exists": False,
        },
        "analysis_contract": {
            "analysis": "stage_b_rlqa_three_seed_ood_k2",
            "method": "rl-qa",
            "model": PROTOCOL.model,
            "seeds": [0, 1, 2],
            "split_seed": 0,
            "budget": 2,
            "condition_order": list(stage_b.CONDITIONS),
            "admission_batch_size": 16,
            "qa_batch_size": 1,
            "no_harm_batch_size": 1,
            "bootstrap_samples": 32,
            "bootstrap_seed": 0,
            "adapters": {stage_b.SFT: SFT_PATH, **RL_PATHS},
        },
    }


def _score(condition: str, candidate: int, episode: int) -> float:
    templates = {
        "original": (0.30, 0.60, 0.20),
        stage_b.SFT: (0.62, 0.45, 0.15),
        stage_b.RLQA[0]: (0.70, 0.40, 0.10),
        stage_b.RLQA[1]: (0.76, 0.35, 0.12),
        stage_b.RLQA[2]: (0.82, 0.30, 0.08),
        "workspace": (0.90, 0.20, 0.10),
        "oracle": (1.0, 0.0, 0.0),
    }
    return templates[condition][candidate] + (
        0.001 * episode if condition != "oracle" else 0.0
    )


def _selection(condition: str, episode: int) -> list[int]:
    patterns = {
        "original": ([0, 1], [1, 2], [0, 2], [1, 2]),
        stage_b.SFT: ([0, 2], [0, 1], [1, 2], [0, 1]),
        stage_b.RLQA[0]: ([0, 1], [0, 2], [1, 2], [0, 1]),
        stage_b.RLQA[1]: ([0, 1], [0, 2], [0, 1], [1, 2]),
        stage_b.RLQA[2]: ([0, 1], [0, 2], [0, 1], [0, 2]),
        "workspace": ([0, 1],) * 4,
        "oracle": ([0, 1],) * 4,
    }
    return list(patterns[condition][episode % 4])


def _qa(condition: str, episode: int, exploitable: bool) -> bool:
    if condition == "oracle":
        return exploitable
    patterns = {
        "original": (False, False, True, False),
        stage_b.SFT: (False, True, True, False),
        stage_b.RLQA[0]: (True, False, True, False),
        stage_b.RLQA[1]: (True, True, False, True),
        stage_b.RLQA[2]: (True, True, True, True),
        "workspace": (True, False, True, False),
    }
    return patterns[condition][episode % 4]


def _full_context(condition: str, episode: int) -> bool:
    patterns = {
        "original": (True, True, False, True),
        stage_b.SFT: (True, False, False, True),
        stage_b.RLQA[0]: (True, True, False, True),
        stage_b.RLQA[1]: (True, True, True, False),
        stage_b.RLQA[2]: (True, True, True, True),
    }
    return patterns[condition][episode % 4]


def _exact_for_evaluator(base: list[bool], adapter: list[bool]) -> dict:
    base_only = sum(a and not b for a, b in zip(base, adapter))
    adapter_only = sum(b and not a for a, b in zip(base, adapter))
    discordant = base_only + adapter_only
    if discordant:
        tail = min(base_only, adapter_only)
        p_value = min(
            1.0,
            2 * sum(math.comb(discordant, i) for i in range(tail + 1)) / 2**discordant,
        )
    else:
        p_value = 1.0
    return {
        "base_only_correct": base_only,
        "adapter_only_correct": adapter_only,
        "discordant": discordant,
        "exact_mcnemar_p_value": p_value,
        "n": len(base),
    }


def _fixture(contract: stage_b.SourceContract) -> dict:
    items: list[dict] = []
    by_episode: list[list[dict]] = []
    episodes: list[dict] = []
    vectors = {
        condition: {"qa": [], "containment": []} for condition in stage_b.CONDITIONS
    }
    noharm = {condition: [] for condition in stage_b.MODEL_CONDITIONS}
    for episode_index in range(contract.n_episodes):
        episode_uid = f"{contract.source}:episode:{episode_index:06d}"
        episode_items = []
        for candidate in range(3):
            row = {
                "uid": f"{episode_uid}:candidate:{candidate:03d}",
                "episode_uid": episode_uid,
                "source": contract.source,
                "source_episode": episode_index,
                "candidate_index": candidate,
                "concept": f"concept-{episode_index}-{candidate}",
                "label": "load_bearing"
                if candidate == 0
                else ("distractor" if candidate == 1 else "filler"),
                "scores": {
                    condition: _score(condition, candidate, episode_index)
                    for condition in stage_b.CONDITIONS
                },
            }
            items.append(row)
            episode_items.append(row)
        by_episode.append(episode_items)
        exploitable = episode_index < (contract.exploitable_episodes or 2)
        policies = {}
        for condition in stage_b.CONDITIONS:
            selected = _selection(condition, episode_index)
            containment = 0 in selected
            correct = _qa(condition, episode_index, exploitable)
            vectors[condition]["qa"].append(correct)
            vectors[condition]["containment"].append(containment)
            policies[condition] = {
                "within_episode_auc": 0.5,
                "selections": {
                    "2": {
                        "selected_indices": selected,
                        "selected_candidate_uids": [
                            episode_items[i]["uid"] for i in selected
                        ],
                        "selected_concepts": [
                            episode_items[i]["concept"] for i in selected
                        ],
                        "contains_load_bearing": containment,
                        "qa": {"answer": "fixture", "correct": correct},
                    }
                },
            }
        full = {}
        for condition in stage_b.MODEL_CONDITIONS:
            correct = _full_context(condition, episode_index)
            noharm[condition].append(correct)
            full[condition] = {"answer": "full context fixture", "correct": correct}
        episodes.append(
            {
                "uid": episode_uid,
                "source": contract.source,
                "source_episode": episode_index,
                "policies": policies,
                "refs": {"oracle@2": {"answer": "fixture", "correct": exploitable}},
                "no_harm_full_context": full,
            }
        )

    labels = [row["label"] == "load_bearing" for row in items]
    aucs = {
        condition: stage_b._auc(labels, [row["scores"][condition] for row in items])
        for condition in stage_b.CONDITIONS
    }
    conditions = {}
    for condition in stage_b.CONDITIONS:
        conditions[condition] = {
            "classification": {
                "pooled_auc": aucs[condition],
                "within_episode_auc": 0.5,
                "n_items": contract.n_items,
                "n_episodes": contract.n_episodes,
                "n_within_episode_auc": contract.n_episodes,
                "pooled_auc_bootstrap": {
                    "estimate": aucs[condition],
                    "ci_95": [
                        max(0.0, aucs[condition] - 0.1),
                        min(1.0, aucs[condition] + 0.1),
                    ],
                    "bootstrap_samples_effective": PROTOCOL.bootstrap_samples,
                },
            },
            "selection": {
                "2": {
                    "top_k_containment": sum(vectors[condition]["containment"])
                    / contract.n_episodes,
                    "n_episodes": contract.n_episodes,
                }
            },
            "qa": {
                "2": {
                    "accuracy": sum(vectors[condition]["qa"]) / contract.n_episodes,
                    "n_episodes": contract.n_episodes,
                }
            },
        }
    draws = stage_b._draws(contract.n_episodes, PROTOCOL)
    paired = []
    for left, right in itertools.combinations(stage_b.CONDITIONS, 2):
        distribution = [
            stage_b._episode_auc(by_episode, left, draw)
            - stage_b._episode_auc(by_episode, right, draw)
            for draw in draws
        ]
        paired.append(
            {
                "a": left,
                "b": right,
                "direction": "a_minus_b",
                "pooled_auc_difference": {
                    **stage_b._interval(aucs[left] - aucs[right], distribution),
                    "probability_gt_zero": sum(value > 0 for value in distribution)
                    / len(distribution),
                },
                "within_episode_auc_difference": {
                    "estimate": 0.0,
                    "ci_95": [0.0, 0.0],
                    "bootstrap_samples_effective": PROTOCOL.bootstrap_samples,
                    "probability_gt_zero": 0.0,
                },
            }
        )
    metric_scope = {
        "n_episodes": contract.n_episodes,
        "n_items": contract.n_items,
        "conditions": conditions,
        "bootstrap": {
            "method": "episode_cluster_percentile",
            "confidence": 0.95,
            "samples_requested": PROTOCOL.bootstrap_samples,
            "seed": 0,
            "skipped": False,
        },
        "paired_auc_differences": paired,
    }
    noharm_conditions = {
        condition: {
            "accuracy": sum(values) / contract.n_episodes,
            "n_episodes": contract.n_episodes,
        }
        for condition, values in noharm.items()
    }
    noharm_comparisons = []
    for adapter in stage_b.MODEL_CONDITIONS[1:]:
        exact = _exact_for_evaluator(noharm["original"], noharm[adapter])
        noharm_comparisons.append(
            {
                "base": "original",
                "adapter": adapter,
                "adapter_minus_base_accuracy": noharm_conditions[adapter]["accuracy"]
                - noharm_conditions["original"]["accuracy"],
                **exact,
            }
        )
    noharm_scope = {"conditions": noharm_conditions, "comparisons": noharm_comparisons}
    return {
        "schema_version": 1,
        "config": {
            "model": PROTOCOL.model,
            "specs": [
                {
                    "name": contract.source,
                    "source": contract.source,
                    "results_path": contract.results_path,
                    "battery_path": contract.battery_path,
                }
            ],
            "adapters": {stage_b.SFT: SFT_PATH, **RL_PATHS},
            "rating_json": {},
            "embedding_model": None,
            "budgets": [2],
            "workspace_top_k": 2,
            "dtype": "bfloat16",
            "max_length": 2048,
            "max_new_tokens": 64,
            "admission_batch_size": 16,
            "qa_batch_size": 1,
            "no_harm_batch_size": 1,
            "bootstrap_samples": PROTOCOL.bootstrap_samples,
            "bootstrap_seed": 0,
            "skip_qa": False,
            "skip_no_harm": False,
            "original_verbal_source": "precomputed_v_ref",
            "policy_input_fields": ["context", "candidate.concept"],
            "probe_visible_to_policy": False,
            "recall_model": "adapter-disabled base checkpoint",
        },
        "condition_order": list(stage_b.CONDITIONS),
        "metrics": {
            "by_spec": {contract.source: metric_scope},
            "aggregate": copy.deepcopy(metric_scope),
        },
        "refs": {"skipped": False},
        "mcnemar": {"skipped": False},
        "no_harm": {
            "skipped": False,
            "mode": "adapter_enabled_full_context_qa",
            "separate_from": "adapter_disabled_frozen_base_selection_recall",
            "summary": {
                "by_spec": {contract.source: noharm_scope},
                "aggregate": copy.deepcopy(noharm_scope),
            },
        },
        "per_item": items,
        "per_episode": episodes,
    }


def _payloads() -> dict[str, dict]:
    return {contract.source: _fixture(contract) for contract in PROTOCOL.sources}


def test_analysis_reports_three_seed_raw_paired_and_shared_draw_statistics() -> None:
    payloads = _payloads()
    report = stage_b.build_analysis(
        payloads["decoupled"], payloads["compositional"], _lock(), protocol=PROTOCOL
    )
    dec = report["sources"]["decoupled"]
    assert list(dec["seeds"]) == ["0", "1", "2"]
    summary = dec["aggregate"]["rl_qa"]["qa"]
    assert summary["n_seeds"] == 3
    assert summary["ddof"] == 1
    assert summary["sample_std"] == pytest.approx(
        statistics.stdev(summary["individual"].values())
    )
    paired = dec["aggregate"]["paired"]["mean_rl_qa_minus_original"]["qa"]
    assert paired["direction"] == "mean_rl_qa_minus_original"
    assert paired["shared_episode_bootstrap"]["bootstrap_samples_effective"] == 32
    assert paired["pooled_mcnemar"]["status"] == "not-applicable"
    seed_pair = dec["seeds"]["2"]["paired"]["rl-qa-s2-k2_minus_original"]
    assert seed_pair["qa"]["direction"] == "rl-qa-s2-k2_minus_original"
    assert seed_pair["auc"]["direction"] == "rl-qa-s2-k2_minus_original"
    assert seed_pair["qa"]["exact_mcnemar"]["n_episodes"] == 4
    assert (
        seed_pair["auc"]["paired_episode_bootstrap"]["bootstrap_samples_effective"]
        == 32
    )
    assert dec["exploitable_subset"]["n_episodes"] == 2
    assert dec["exploitable_subset"]["definition"] == "refs['oracle@2'].correct == true"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p["decoupled"]["condition_order"].remove(stage_b.RLQA[1]),
        lambda p: p["decoupled"]["config"].__setitem__("qa_batch_size", 8),
        lambda p: p["decoupled"]["config"].__setitem__("qa_batch_size", True),
        lambda p: p["decoupled"]["config"]["adapters"].__setitem__(
            stage_b.RLQA[2], "wrong"
        ),
        lambda p: p["decoupled"]["per_item"][1].__setitem__(
            "uid", p["decoupled"]["per_item"][0]["uid"]
        ),
        lambda p: p["decoupled"]["per_item"][0].__setitem__("candidate_index", False),
        lambda p: p["decoupled"]["per_episode"][0]["policies"].pop(stage_b.RLQA[1]),
        lambda p: p["decoupled"]["metrics"]["by_spec"]["decoupled"]["conditions"][
            stage_b.RLQA[1]
        ]["qa"]["2"].__setitem__("accuracy", None),
        lambda p: p["decoupled"]["no_harm"].__setitem__("skipped", True),
        lambda p: p["decoupled"]["no_harm"]["summary"]["by_spec"]["decoupled"][
            "comparisons"
        ].__setitem__(0, None),
        lambda p: p["decoupled"]["metrics"]["by_spec"]["decoupled"][
            "paired_auc_differences"
        ][1]["pooled_auc_difference"]["ci_95"].__setitem__(0, -0.999),
    ],
)
def test_analysis_fails_closed_on_missing_seed_protocol_or_metric_tampering(
    mutator,
) -> None:
    payloads = _payloads()
    mutator(payloads)
    with pytest.raises(stage_b.StageBRLQAMultiseedError):
        stage_b.build_analysis(
            payloads["decoupled"], payloads["compositional"], _lock(), protocol=PROTOCOL
        )


def test_analysis_rejects_wrong_exploitable_count_and_post_ood_lock() -> None:
    payloads = _payloads()
    payloads["decoupled"]["per_episode"][1]["refs"]["oracle@2"]["correct"] = False
    payloads["decoupled"]["per_episode"][1]["policies"]["oracle"]["selections"]["2"][
        "qa"
    ]["correct"] = False
    payloads["decoupled"]["metrics"]["by_spec"]["decoupled"]["conditions"]["oracle"][
        "qa"
    ]["2"]["accuracy"] = 0.25
    with pytest.raises(stage_b.StageBRLQAMultiseedError, match="exploitable count"):
        stage_b.build_analysis(
            payloads["decoupled"], payloads["compositional"], _lock(), protocol=PROTOCOL
        )
    lock = _lock()
    lock["ood_state_at_lock"]["results_inspected"] = True
    payloads = _payloads()
    with pytest.raises(stage_b.StageBRLQAMultiseedError, match="results_inspected"):
        stage_b.build_analysis(
            payloads["decoupled"], payloads["compositional"], lock, protocol=PROTOCOL
        )


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def test_independent_validator_and_exclusive_output(tmp_path: Path) -> None:
    payloads = _payloads()
    dec, comp, lock, analysis = [
        tmp_path / name
        for name in ("dec.json", "comp.json", "lock.json", "analysis.json")
    ]
    _write(dec, payloads["decoupled"])
    _write(comp, payloads["compositional"])
    _write(lock, _lock())
    result = stage_b.analyze_files(dec, comp, lock, protocol=PROTOCOL)
    _write(analysis, result)
    assert (
        stage_b.validate_files(dec, comp, lock, analysis, protocol=PROTOCOL)["status"]
        == "pass"
    )
    tampered = copy.deepcopy(result)
    tampered["sources"]["decoupled"]["seeds"]["0"]["raw"]["qa"] = 0.0
    _write(analysis, tampered)
    assert (
        stage_b.validate_files(dec, comp, lock, analysis, protocol=PROTOCOL)["status"]
        == "fail"
    )
    existing = tmp_path / "existing.json"
    existing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(stage_b.StageBRLQAMultiseedError, match="refusing to overwrite"):
        stage_b._write_once(existing, {"status": "complete"})
