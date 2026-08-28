from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import stage_c_ood_analysis as stage_c


TEST_CONDITIONS = stage_c.CONDITION_ORDER
TEST_BASELINE_ADAPTERS = (
    ("sft-w-s0-k2", "test/adapters/sft/best-step-300"),
    ("rl-w-s0-k2", "test/adapters/rl-w/best-step-200"),
    ("rl-qa-s0-k2", "test/adapters/rl-qa/best-step-200"),
)
TEST_HYBRID_ADAPTER = "test/adapters/hybrid-lw0p25/best-step-200"
TEST_PROTOCOL = stage_c.StageCProtocol(
    model="test/model",
    conditions=TEST_CONDITIONS,
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
        stage_c.SourceContract(
            source="decoupled",
            results_path="test/decoupled-results.json",
            battery_path="test/decoupled-battery.json",
            n_episodes=4,
            n_items=12,
            exploitable_episodes=2,
        ),
        stage_c.SourceContract(
            source="compositional",
            results_path="test/compositional-results.json",
            battery_path="test/compositional-battery.json",
            n_episodes=3,
            n_items=9,
        ),
    ),
    baseline_adapters=TEST_BASELINE_ADAPTERS,
    hybrid_adapter=TEST_HYBRID_ADAPTER,
)


def _lock() -> dict:
    return {
        "schema_version": 1,
        "lock_status": "locked",
        "authorization": {"pre_ood": True},
        "selection_rule": [
            "maximize ID QA accuracy",
            "maximize workspace reward after a QA tie",
            "select the smallest lambda_w after an exact tie",
        ],
        "locked_configuration": {
            "method": "rl-hybrid",
            "model": TEST_PROTOCOL.model,
            "seed": 0,
            "split_seed": 0,
            "budget": 2,
            "lambda_qa": 1,
            "lambda_w": 0.25,
            "checkpoint_step": 200,
            "checkpoint_path": TEST_HYBRID_ADAPTER,
            "teacher_mismatch_override": False,
            "validator_status": "pass",
        },
        "reporter_correlations_used_for_selection": False,
        "ood_state_at_lock": {
            "results_inspected": False,
            "completed_artifact_exists": False,
        },
        "authorized_next_experiment": {
            "scope": "one-shot Stage C OOD",
            "batteries": ["decoupled", "compositional"],
            "primary_qa_batch_size": 1,
            "answer_tokens": 64,
            "bootstrap_samples": 32,
        },
    }


def _scores(episode_index: int, candidate_index: int) -> dict[str, float]:
    positive = candidate_index == 0
    jitter = episode_index * 0.001
    templates = {
        "original": (0.25, 0.65, 0.45),
        "sft-w-s0-k2": (0.68, 0.40, 0.20),
        "rl-w-s0-k2": (0.48, 0.55, 0.25),
        "rl-qa-s0-k2": (0.74, 0.35, 0.15),
        "rl-hybrid-s0-k2-lw0p25": (0.82, 0.30, 0.10),
        "workspace": (0.90, 0.32, 0.12),
        "oracle": (1.0, 0.0, 0.0),
    }
    result = {}
    # Deliberately use a non-condition insertion order. The evaluator's
    # per-item score object is not ordered like condition_order.
    for condition in (
        "workspace",
        "oracle",
        "original",
        "sft-w-s0-k2",
        "rl-w-s0-k2",
        "rl-qa-s0-k2",
        "rl-hybrid-s0-k2-lw0p25",
    ):
        value = templates[condition][candidate_index]
        result[condition] = value + (jitter if condition != "oracle" else 0.0)
    assert positive == (candidate_index == 0)
    return result


def _selection(condition: str, episode_index: int) -> list[int]:
    if condition == "original":
        return [0, 1] if episode_index % 2 == 0 else [1, 2]
    if condition == "rl-w-s0-k2":
        return [1, 0] if episode_index % 3 else [1, 2]
    if condition == "sft-w-s0-k2":
        return [0, 2] if episode_index != 2 else [1, 2]
    if condition in {
        "rl-qa-s0-k2",
        "rl-hybrid-s0-k2-lw0p25",
        "workspace",
        "oracle",
    }:
        return [0, 1]
    raise AssertionError(condition)


def _qa_correct(condition: str, episode_index: int, exploitable: bool) -> bool:
    if condition == "oracle":
        return exploitable
    patterns = {
        "original": (False, False, True, False),
        "sft-w-s0-k2": (False, True, False, False),
        "rl-w-s0-k2": (False, False, False, True),
        "rl-qa-s0-k2": (True, False, False, True),
        "rl-hybrid-s0-k2-lw0p25": (True, True, False, True),
        "workspace": (True, False, True, False),
    }
    return patterns[condition][episode_index % 4]


def _no_harm_correct(condition: str, episode_index: int) -> bool:
    patterns = {
        "original": (True, True, False, True),
        "sft-w-s0-k2": (True, False, False, True),
        "rl-w-s0-k2": (True, True, False, False),
        "rl-qa-s0-k2": (True, True, False, True),
        "rl-hybrid-s0-k2-lw0p25": (True, True, True, True),
    }
    return patterns[condition][episode_index % 4]


def _fixture_payload(contract: stage_c.SourceContract) -> dict:
    per_item = []
    per_episode = []
    labels = []
    score_vectors = {condition: [] for condition in TEST_CONDITIONS}
    qa_vectors = {condition: [] for condition in TEST_CONDITIONS}
    containment_vectors = {condition: [] for condition in TEST_CONDITIONS}
    for episode_index in range(contract.n_episodes):
        episode_uid = f"{contract.source}:episode:{episode_index:06d}"
        concepts = [f"concept-{episode_index}-{index}" for index in range(3)]
        episode_items = []
        for candidate_index in range(3):
            uid = f"{episode_uid}:candidate:{candidate_index:03d}"
            label = "load_bearing" if candidate_index == 0 else (
                "distractor" if candidate_index == 1 else "filler"
            )
            scores = _scores(episode_index, candidate_index)
            row = {
                "uid": uid,
                "episode_uid": episode_uid,
                "source": contract.source,
                "source_episode": episode_index,
                "candidate_index": candidate_index,
                "concept": concepts[candidate_index],
                "label": label,
                "scores": scores,
            }
            per_item.append(row)
            episode_items.append(row)
            labels.append(label == "load_bearing")
            for condition in TEST_CONDITIONS:
                score_vectors[condition].append(scores[condition])

        exploitable = (
            episode_index < contract.exploitable_episodes
            if contract.exploitable_episodes is not None
            else episode_index % 2 == 0
        )
        policies = {}
        for condition in TEST_CONDITIONS:
            indices = _selection(condition, episode_index)
            contains = 0 in indices
            correct = _qa_correct(condition, episode_index, exploitable)
            qa_vectors[condition].append(correct)
            containment_vectors[condition].append(contains)
            policies[condition] = {
                "within_episode_auc": 0.5,
                "selections": {
                    "2": {
                        "selected_indices": indices,
                        "selected_candidate_uids": [episode_items[index]["uid"] for index in indices],
                        "selected_concepts": [concepts[index] for index in indices],
                        "contains_load_bearing": contains,
                        "qa": {"correct": correct, "answer": "fixture"},
                    }
                },
            }
        no_harm_details = {
            condition: {
                "answer": "full-context fixture",
                "correct": _no_harm_correct(condition, episode_index),
            }
            for condition in stage_c.NO_HARM_CONDITIONS
        }
        per_episode.append(
            {
                "uid": episode_uid,
                "source": contract.source,
                "source_episode": episode_index,
                "policies": policies,
                "refs": {"oracle@2": {"correct": exploitable, "answer": "fixture"}},
                "no_harm_full_context": no_harm_details,
            }
        )

    conditions = {}
    aucs = {}
    for condition in TEST_CONDITIONS:
        auc = stage_c._binary_auc(labels, score_vectors[condition])
        aucs[condition] = auc
        conditions[condition] = {
            "classification": {
                "pooled_auc": auc,
                "within_episode_auc": 0.5,
                "n_items": contract.n_items,
                "n_episodes": contract.n_episodes,
                "n_within_episode_auc": contract.n_episodes,
                "pooled_auc_bootstrap": {
                    "estimate": auc,
                    "ci_95": [max(0.0, auc - 0.05), min(1.0, auc + 0.05)],
                    "bootstrap_samples_effective": TEST_PROTOCOL.bootstrap_samples,
                },
            },
            "selection": {
                "2": {
                    "top_k_containment": sum(containment_vectors[condition]) / contract.n_episodes,
                    "n_episodes": contract.n_episodes,
                }
            },
            "qa": {
                "2": {
                    "accuracy": sum(qa_vectors[condition]) / contract.n_episodes,
                    "n_episodes": contract.n_episodes,
                }
            },
        }
    paired_auc = []
    for left, right in itertools.combinations(TEST_CONDITIONS, 2):
        estimate = aucs[left] - aucs[right]
        paired_auc.append(
            {
                "a": left,
                "b": right,
                "direction": "a_minus_b",
                "pooled_auc_difference": {
                    "estimate": estimate,
                    "ci_95": [estimate - 0.05, estimate + 0.05],
                    "bootstrap_samples_effective": TEST_PROTOCOL.bootstrap_samples,
                    "probability_gt_zero": 0.5,
                },
                "within_episode_auc_difference": {
                    "estimate": 0.0,
                    "ci_95": [-0.1, 0.1],
                    "bootstrap_samples_effective": TEST_PROTOCOL.bootstrap_samples,
                    "probability_gt_zero": 0.5,
                },
            }
        )

    adapter_map = {**dict(TEST_BASELINE_ADAPTERS), stage_c.HYBRID: TEST_HYBRID_ADAPTER}
    no_harm_vectors = {
        condition: [
            _no_harm_correct(condition, episode_index)
            for episode_index in range(contract.n_episodes)
        ]
        for condition in stage_c.NO_HARM_CONDITIONS
    }
    no_harm_conditions = {
        condition: {
            "accuracy": sum(values) / contract.n_episodes,
            "n_episodes": contract.n_episodes,
        }
        for condition, values in no_harm_vectors.items()
    }
    no_harm_comparisons = []
    for adapter in stage_c.NO_HARM_CONDITIONS[1:]:
        original = no_harm_vectors["original"]
        adapted = no_harm_vectors[adapter]
        base_only = sum(a and not b for a, b in zip(original, adapted))
        adapter_only = sum(b and not a for a, b in zip(original, adapted))
        discordant = base_only + adapter_only
        if discordant:
            tail = min(base_only, adapter_only)
            numerator = sum(
                math.comb(discordant, index)
                for index in range(tail + 1)
            )
            p_value = min(1.0, 2.0 * numerator / (2**discordant))
        else:
            p_value = 1.0
        no_harm_comparisons.append(
            {
                "base": "original",
                "adapter": adapter,
                "adapter_minus_base_accuracy": (
                    no_harm_conditions[adapter]["accuracy"]
                    - no_harm_conditions["original"]["accuracy"]
                ),
                "base_only_correct": base_only,
                "adapter_only_correct": adapter_only,
                "discordant": discordant,
                "exact_mcnemar_p_value": p_value,
                "n": contract.n_episodes,
            }
        )
    scope = {
        "n_episodes": contract.n_episodes,
        "n_items": contract.n_items,
        "conditions": conditions,
        "bootstrap": {
            "method": "episode_cluster_percentile",
            "confidence": 0.95,
            "samples_requested": TEST_PROTOCOL.bootstrap_samples,
            "seed": 0,
            "skipped": False,
        },
        "paired_auc_differences": paired_auc,
    }
    return {
        "schema_version": 1,
        "config": {
            "model": TEST_PROTOCOL.model,
            "specs": [
                {
                    "name": contract.source,
                    "source": contract.source,
                    "results_path": contract.results_path,
                    "battery_path": contract.battery_path,
                }
            ],
            "adapters": adapter_map,
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
            "bootstrap_samples": TEST_PROTOCOL.bootstrap_samples,
            "bootstrap_seed": 0,
            "skip_qa": False,
            "skip_no_harm": False,
            "original_verbal_source": "precomputed_v_ref",
            "policy_input_fields": ["context", "candidate.concept"],
            "probe_visible_to_policy": False,
            "recall_model": "adapter-disabled base checkpoint",
        },
        "condition_order": list(TEST_CONDITIONS),
        "metrics": {"by_spec": {contract.source: scope}, "aggregate": scope},
        "refs": {"skipped": False},
        "mcnemar": {"skipped": False},
        "no_harm": {
            "skipped": False,
            "mode": "adapter_enabled_full_context_qa",
            "separate_from": "adapter_disabled_frozen_base_selection_recall",
            "summary": {
                "by_spec": {
                    contract.source: {
                        "conditions": no_harm_conditions,
                        "comparisons": no_harm_comparisons,
                    }
                },
                "aggregate": {
                    "conditions": copy.deepcopy(no_harm_conditions),
                    "comparisons": copy.deepcopy(no_harm_comparisons),
                },
            },
        },
        "per_item": per_item,
        "per_episode": per_episode,
    }


def _payloads():
    return {
        contract.source: _fixture_payload(contract)
        for contract in TEST_PROTOCOL.sources
    }


def test_stage_c_analysis_recomputes_metrics_and_orients_paired_statistics():
    payloads = _payloads()
    result = stage_c.build_analysis(
        payloads["decoupled"],
        payloads["compositional"],
        _lock(),
        protocol=TEST_PROTOCOL,
    )

    assert result["lock"]["lambda_w"] == 0.25
    decoupled = result["sources"]["decoupled"]
    assert decoupled["n_episodes"] == 4
    assert decoupled["exploitable_subset"]["n_episodes"] == 2
    assert decoupled["exploitable_subset"]["definition"] == "refs['oracle@2'].correct == true"
    assert set(decoupled["conditions"]) == set(TEST_CONDITIONS)

    comparison = decoupled["paired_comparisons"][
        "rl-hybrid-s0-k2-lw0p25_minus_original"
    ]
    hybrid_qa = decoupled["conditions"][stage_c.HYBRID]["qa_accuracy"]
    original_qa = decoupled["conditions"]["original"]["qa_accuracy"]
    assert comparison["qa"]["paired_episode_bootstrap"]["estimate"] == pytest.approx(
        hybrid_qa - original_qa
    )
    assert comparison["qa"]["paired_episode_bootstrap"]["bootstrap_samples_effective"] == 32
    assert comparison["qa"]["exact_mcnemar"]["n_episodes"] == 4
    assert comparison["containment"]["exact_mcnemar"]["n_episodes"] == 4
    assert (
        comparison["pooled_auc"]["direction"]
        == "rl-hybrid-s0-k2-lw0p25_minus_original"
    )
    assert comparison["pooled_auc"]["estimate"] == pytest.approx(
        decoupled["conditions"][stage_c.HYBRID]["pooled_auc"]
        - decoupled["conditions"]["original"]["pooled_auc"]
    )
    assert set(decoupled["exploitable_subset"]["paired_comparisons"]) == {
        "rl-hybrid-s0-k2-lw0p25_minus_rl-qa-s0-k2",
        "rl-hybrid-s0-k2-lw0p25_minus_sft-w-s0-k2",
        "rl-hybrid-s0-k2-lw0p25_minus_original",
    }
    no_harm = decoupled["no_harm_full_context"]
    assert no_harm["condition_order"] == list(stage_c.NO_HARM_CONDITIONS)
    assert no_harm["summary_verified"] == ["by_spec.decoupled", "aggregate"]
    assert no_harm["primary_hybrid_vs_original"][
        "adapter_minus_original_accuracy"
    ] == pytest.approx(0.25)
    assert no_harm["primary_hybrid_vs_original"]["exact_mcnemar"][
        "n_episodes"
    ] == 4


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p["decoupled"]["config"].__setitem__("qa_batch_size", 8),
        lambda p: p["decoupled"]["config"].__setitem__("no_harm_batch_size", 8),
        lambda p: p["decoupled"]["config"].__setitem__("skip_no_harm", True),
        lambda p: p["decoupled"]["no_harm"].__setitem__("skipped", True),
        lambda p: p["decoupled"]["per_episode"][0]["no_harm_full_context"].pop(
            stage_c.HYBRID
        ),
        lambda p: p["decoupled"]["no_harm"]["summary"]["by_spec"]["decoupled"][
            "conditions"
        ][stage_c.HYBRID].__setitem__("accuracy", 0.0),
        lambda p: p["decoupled"]["no_harm"]["summary"]["aggregate"]["conditions"][
            stage_c.HYBRID
        ].__setitem__("accuracy", 0.0),
        lambda p: p["decoupled"]["no_harm"]["summary"]["by_spec"]["decoupled"][
            "comparisons"
        ][-1].__setitem__("exact_mcnemar_p_value", 0.0),
        lambda p: p["decoupled"]["condition_order"].reverse(),
        lambda p: p["decoupled"]["config"]["adapters"].__setitem__(
            stage_c.HYBRID, "wrong/hybrid"
        ),
        lambda p: p["decoupled"]["per_item"][1].__setitem__(
            "uid", p["decoupled"]["per_item"][0]["uid"]
        ),
        lambda p: p["decoupled"]["metrics"]["by_spec"]["decoupled"]["conditions"][
            stage_c.HYBRID
        ]["classification"].__setitem__("pooled_auc", 0.123),
    ],
)
def test_stage_c_analysis_fails_closed_on_protocol_or_data_tampering(mutator):
    payloads = _payloads()
    mutator(payloads)
    with pytest.raises(stage_c.StageCOODAnalysisError):
        stage_c.build_analysis(
            payloads["decoupled"],
            payloads["compositional"],
            _lock(),
            protocol=TEST_PROTOCOL,
        )


def test_stage_c_analysis_rejects_wrong_exploitable_count_and_post_ood_lock():
    payloads = _payloads()
    payloads["decoupled"]["per_episode"][1]["refs"]["oracle@2"]["correct"] = False
    payloads["decoupled"]["per_episode"][1]["policies"]["oracle"]["selections"]["2"][
        "qa"
    ]["correct"] = False
    payloads["decoupled"]["metrics"]["by_spec"]["decoupled"]["conditions"]["oracle"][
        "qa"
    ]["2"]["accuracy"] = 0.25
    with pytest.raises(stage_c.StageCOODAnalysisError, match="exploitable episode count"):
        stage_c.build_analysis(
            payloads["decoupled"],
            payloads["compositional"],
            _lock(),
            protocol=TEST_PROTOCOL,
        )

    lock = _lock()
    lock["ood_state_at_lock"]["results_inspected"] = True
    payloads = _payloads()
    with pytest.raises(stage_c.StageCOODAnalysisError, match="results_inspected"):
        stage_c.build_analysis(
            payloads["decoupled"],
            payloads["compositional"],
            lock,
            protocol=TEST_PROTOCOL,
        )


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def test_stage_c_validator_independently_recomputes_raw_inputs(tmp_path: Path):
    payloads = _payloads()
    decoupled_path = tmp_path / "decoupled.json"
    compositional_path = tmp_path / "compositional.json"
    lock_path = tmp_path / "lock.json"
    analysis_path = tmp_path / "analysis.json"
    _write(decoupled_path, payloads["decoupled"])
    _write(compositional_path, payloads["compositional"])
    _write(lock_path, _lock())
    analysis = stage_c.analyze_files(
        decoupled_path,
        compositional_path,
        lock_path,
        protocol=TEST_PROTOCOL,
    )
    _write(analysis_path, analysis)

    validation = stage_c.validate_analysis_files(
        decoupled_path,
        compositional_path,
        lock_path,
        analysis_path,
        protocol=TEST_PROTOCOL,
    )
    assert validation["status"] == "pass"
    assert validation["errors"] == []

    tampered = copy.deepcopy(analysis)
    tampered["sources"]["decoupled"]["conditions"][stage_c.HYBRID][
        "qa_accuracy"
    ] = 0.0
    _write(analysis_path, tampered)
    failed = stage_c.validate_analysis_files(
        decoupled_path,
        compositional_path,
        lock_path,
        analysis_path,
        protocol=TEST_PROTOCOL,
    )
    assert failed["status"] == "fail"
    assert "qa_accuracy" in failed["errors"][0]


def test_stage_c_outputs_refuse_to_overwrite(tmp_path: Path):
    output = tmp_path / "already-exists.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(stage_c.StageCOODAnalysisError, match="refusing to overwrite"):
        stage_c._write_json_exclusive(output, {"status": "pass"})
    assert output.read_text(encoding="utf-8") == "{}\n"
