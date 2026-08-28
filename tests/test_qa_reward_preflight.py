from __future__ import annotations

import math
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.preflight_qa_reward import (  # noqa: E402
    exact_inclusion_probabilities,
    policy_prompts,
    sample_episode_sets,
)
from analysis.validate_qa_reward_preflight import validate_run  # noqa: E402
from memory_rl.data import file_sha256, write_split_manifest  # noqa: E402
from memory_rl.qa_preflight import (  # noqa: E402
    classify_gate_b0,
    select_temperature,
    summarize_group,
    summarize_preflight,
)


def _sample(episode: str, sample_id: int, selected: list[str], correct: bool, contains: bool):
    return {
        "episode_id": episode,
        "sample_id": sample_id,
        "selected_set": selected,
        "QA_correct": correct,
        "QA_reward": float(correct),
        "contains_load_bearing": contains,
    }


def test_group_summary_uses_population_std_and_canonical_set_identity():
    rows = [
        _sample("ep", index, ["b", "a"] if index < 8 else ["c", "a"], index < 8, index % 2 == 0)
        for index in range(16)
    ]
    result = summarize_group(rows)
    assert result["number_unique_selected_sets"] == 2
    assert result["fraction_unique_selected_sets"] == 0.125
    assert result["mean_QA_reward"] == 0.5
    assert result["QA_reward_std"] == 0.5
    assert result["mixed_QA_reward_group"] is True
    assert result["mixed_containment_group"] is True


def test_group_summary_rejects_bad_reward_and_duplicate_set_members():
    row = _sample("ep", 0, ["a", "a"], True, True)
    with pytest.raises(ValueError, match="duplicates"):
        summarize_group([row])
    row["selected_set"] = ["a", "b"]
    row["QA_reward"] = 0.0
    with pytest.raises(ValueError, match="must equal"):
        summarize_group([row])


@pytest.mark.parametrize(
    ("mixed", "median_unique", "status"),
    [
        (34 / 175, 6, "RED"),
        (35 / 175, 6, "AMBER"),
        (69 / 175, 6, "AMBER"),
        (70 / 175, 4, "GREEN"),
        (0.8, 3, "AMBER"),
    ],
)
def test_gate_b0_boundaries_and_diversity_gap_are_predeclared(
    mixed, median_unique, status
):
    assert classify_gate_b0(mixed, median_unique)["status"] == status


def test_temperature_selection_uses_lowest_diverse_candidate_or_highest_fallback():
    rows = [
        {"temperature": 5.0, "median_unique_selected_sets": 7},
        {"temperature": 0.7, "median_unique_selected_sets": 2},
        {"temperature": 2.0, "median_unique_selected_sets": 4},
    ]
    assert select_temperature(rows) == (
        2.0,
        "lowest candidate meeting the median diversity target",
    )
    assert select_temperature(rows, min_median_unique_sets=9)[0] == 5.0


def test_aggregate_recomputes_conditionals_references_and_gate():
    rows = [
        _sample("ep1", 0, ["a", "b"], True, True),
        _sample("ep1", 1, ["a", "c"], False, False),
        _sample("ep2", 0, ["d", "e"], True, True),
        _sample("ep2", 1, ["d", "f"], True, False),
    ]
    groups = [summarize_group(rows[:2]), summarize_group(rows[2:])]
    refs = [
        {
            "oracle_QA_correct": True,
            "full_context_QA_correct": True,
            "no_memory_QA_correct": False,
        },
        {
            "oracle_QA_correct": False,
            "full_context_QA_correct": True,
            "no_memory_QA_correct": False,
        },
    ]
    result = summarize_preflight(rows, groups, refs)
    relation = result["containment_QA_relationship"]
    assert relation["P_QA_correct_given_load_bearing_retained"] == 1.0
    assert relation["P_QA_correct_given_load_bearing_not_retained"] == 0.5
    assert relation["difference"] == 0.5
    assert result["references"]["exploitable_fraction"] == 0.5


def test_probe_boundary_is_enforced_on_the_real_prompt_constructor():
    tokenizer = SimpleNamespace(chat_template=None)
    episode = SimpleNamespace(
        uid="source:episode:000000",
        context="A harmless passage.",
        probe_question="UNIQUE_PROBE_SENTINEL?",
        candidates=(SimpleNamespace(concept="alpha"), SimpleNamespace(concept="beta")),
    )
    prompts, hashes = policy_prompts(tokenizer, episode)
    assert all(episode.probe_question not in prompt for prompt in prompts)
    assert len(hashes) == 2 and all(len(value) == 64 for value in hashes)

    episode.context += " UNIQUE_PROBE_SENTINEL?"
    with pytest.raises(RuntimeError, match="probe leaked"):
        policy_prompts(tokenizer, episode)


def test_exact_inclusion_probabilities_sum_to_budget():
    probabilities = exact_inclusion_probabilities(
        torch.tensor([-0.3, 0.2, 1.1]), budget=2, temperature=0.7
    )
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert math.isclose(sum(probabilities), 2.0, rel_tol=0.0, abs_tol=1e-6)


def test_episode_hashed_sampling_is_reorder_invariant_and_exact_budget():
    episodes = [
        SimpleNamespace(uid="source:episode:000001"),
        SimpleNamespace(uid="source:episode:000002"),
    ]
    scores = {
        episode.uid: {"selection_logits": [-1.0, -0.2, 0.4, 1.0]}
        for episode in episodes
    }
    first = sample_episode_sets(
        episodes, scores, budget=2, group_size=16, temperature=1.0, seed=0
    )
    second = sample_episode_sets(
        list(reversed(episodes)),
        scores,
        budget=2,
        group_size=16,
        temperature=1.0,
        seed=0,
    )
    assert first == second
    assert all(
        len(selected) == len(set(selected)) == 2
        for groups in first.values()
        for selected in groups
    )


def test_formal_validator_recomputes_artifacts_and_rejects_tampering(tmp_path):
    run = tmp_path / "b0"
    run.mkdir()
    train_episode = SimpleNamespace(uid="explicit:episode:000000", source="explicit")
    val_episode = SimpleNamespace(uid="explicit:episode:000001", source="explicit")
    manifest = write_split_manifest(
        run / "split_manifest.json", [train_episode], [val_episode], seed=0
    )
    config = {
        "stage": "B0",
        "training_performed": False,
        "optimizer_created": False,
        "group_size": 16,
        "budget": 2,
        "seed": 0,
        "split_seed": 0,
        "answer_tokens": 64,
        "max_length": 2048,
        "probe_visible_to_policy": False,
        "gold_answer_visible_to_policy": False,
        "sets_sampled_before_reward_prompts": True,
        "frozen_recall_adapter_disabled": True,
        "teacher_matches_policy_reference": True,
        "teacher_mismatch_override": False,
        "policy_input_fields": ["context", "candidate.concept"],
        "sampling": "exact-budget Gumbel top-k",
        "min_median_unique_sets": 4.0,
        "selected_temperature": 1.0,
        "limit_episodes": 0,
        "split_manifest_sha256": manifest["manifest_sha256"],
    }
    (run / "run_config.json").write_text(json.dumps(config))
    (run / "dropout_audit.json").write_text(
        json.dumps({"postcondition_satisfied": True, "remaining_nonzero": []})
    )
    calibration = {
        "selection_uses_QA_or_OOD": False,
        "selected_temperature": 1.0,
        "candidates": [{"temperature": 1.0, "median_unique_selected_sets": 4.0}],
    }
    (run / "temperature_calibration.json").write_text(json.dumps(calibration))

    candidate_ids = [f"explicit:episode:000000:candidate:{index:03d}" for index in range(4)]
    candidate_text = ["target", "one", "two", "three"]
    candidate_labels = ["load_bearing", "distractor", "filler", "filler"]
    chosen_sets = [(0, 1), (0, 2), (0, 3), (1, 2)] * 4
    log_probability = -math.log(6)
    samples = []
    for sample_id, indices in enumerate(chosen_sets):
        correct = sample_id % 2 == 0
        selected_lb = int(0 in indices)
        samples.append(
            {
                "episode_id": train_episode.uid,
                "source": "explicit",
                "sample_id": sample_id,
                "candidate_ids": candidate_ids,
                "candidate_text": candidate_text,
                "candidate_labels": candidate_labels,
                "policy_input_fields": ["context", "candidate.concept"],
                "probe_visible_to_policy": False,
                "policy_prompt_sha256": ["a" * 64] * 4,
                "action_logits_no_yes": [[0.0, 0.0]] * 4,
                "action_probabilities_no_yes": [[0.49999997, 0.50000003]] * 4,
                "selection_logits": [0.0] * 4,
                "selection_probabilities": [0.25000004, 0.25, 0.25, 0.25],
                "first_draw_probabilities": [0.25000004, 0.25, 0.25, 0.25],
                "inclusion_probabilities": [0.5] * 4,
                "yes_probabilities": [0.50000003] * 4,
                "selected_indices": list(indices),
                "selected_set": [candidate_ids[index] for index in indices],
                "selected_concepts": [candidate_text[index] for index in indices],
                "selected_set_log_probability": log_probability,
                "selected_set_probability": 1 / 6,
                "set_occurrence_in_group": 4,
                "exact_budget": True,
                "budget": 2,
                "temperature": 1.0,
                "workspace_scores": [0.8, 0.4, 0.2, 0.1],
                "workspace_percentiles": [1.0, 2 / 3, 1 / 3, 0.0],
                "verbal_scores": [0.5] * 4,
                "contains_load_bearing": bool(selected_lb),
                "selected_load_bearing_count": selected_lb,
                "contains_all_load_bearing": bool(selected_lb),
                "probe_question": "What is the target?",
                "gold_answer": "target",
                "generated_answer": "target" if correct else "wrong",
                "QA_correct": correct,
                "QA_reward": float(correct),
                "oracle_QA_correct": True,
                "full_context_QA_correct": True,
                "no_memory_QA_correct": False,
            }
        )
    group = summarize_group(samples)
    group["source"] = "explicit"
    reference = {
        "episode_id": train_episode.uid,
        "source": "explicit",
        "probe_question": "What is the target?",
        "gold_answer": "target",
        "oracle_set": candidate_ids[:2],
        "oracle_answer": "target",
        "oracle_QA_correct": True,
        "full_context_answer": "target",
        "full_context_QA_correct": True,
        "no_memory_answer": "wrong",
        "no_memory_QA_correct": False,
    }

    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    write_jsonl(run / "samples.jsonl", samples)
    write_jsonl(run / "groups.jsonl", [group])
    write_jsonl(run / "references.jsonl", [reference])
    summary = summarize_preflight(samples, [group], [reference])
    summary["by_source"] = {"explicit": summarize_preflight(samples, [group], [reference])}
    g8_groups = [summarize_group(samples[:8]), summarize_group(samples[8:])]
    g8_mixed = sum(value["mixed_QA_reward_group"] for value in g8_groups) / 2
    g8_unique = [value["number_unique_selected_sets"] for value in g8_groups]
    summary["g8_sensitivity"] = {
        "mixed_QA_reward_groups_fraction": g8_mixed,
        "median_unique_selected_sets": sum(g8_unique) / 2,
        "gate_if_G8_thresholds_were_applied": classify_gate_b0(
            g8_mixed, sum(g8_unique) / 2
        ),
    }
    summary.update({"status": "complete", "training_performed": False})
    summary["artifacts"] = {
        name: file_sha256(run / name)
        for name in (
            "samples.jsonl",
            "groups.jsonl",
            "references.jsonl",
            "temperature_calibration.json",
            "split_manifest.json",
        )
    }
    (run / "summary.json").write_text(json.dumps(summary))

    assert validate_run(run)["status"] == "pass"
    samples[0]["selected_set"] = [candidate_ids[0], candidate_ids[0]]
    write_jsonl(run / "samples.jsonl", samples)
    report = validate_run(run)
    assert report["status"] == "fail"
    assert "exact_budget" in {issue["code"] for issue in report["errors"]}
