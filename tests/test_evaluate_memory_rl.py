from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments import evaluate_memory_rl as evaluation
from memory_rl.data import CandidateRecord, EpisodeRecord


def _candidate(
    episode_uid: str,
    episode_index: int,
    candidate_index: int,
    label: str,
) -> CandidateRecord:
    return CandidateRecord(
        uid=f"{episode_uid}:candidate:{candidate_index:03d}",
        context=f"context {episode_index}",
        concept=f"concept-{episode_index}-{candidate_index}",
        label=label,
        w_ref=float(candidate_index),
        v_ref=0.5,
        w_percentile=float(candidate_index),
        workspace_target=candidate_index == 0,
        source="decoupled",
        source_episode=episode_index,
        candidate_index=candidate_index,
        episode_uid=episode_uid,
        role=None,
        fingerprint_sha256=f"candidate-fingerprint-{episode_index}-{candidate_index}",
        result={},
    )


def _episode(index: int) -> EpisodeRecord:
    uid = f"decoupled:episode:{index:06d}"
    return EpisodeRecord(
        uid=uid,
        source="decoupled",
        source_episode=index,
        context=f"context {index}",
        probe_question=f"question {index}?",
        answer=f"answer {index}",
        candidates=(
            _candidate(uid, index, 0, "load_bearing"),
            _candidate(uid, index, 1, "distractor"),
            _candidate(uid, index, 2, "filler"),
        ),
        fingerprint_sha256=f"episode-fingerprint-{index}",
    )


def test_binary_auc_is_tie_aware_and_handles_degenerate_labels() -> None:
    assert evaluation.binary_auc([False, True], [0.5, 0.5]) == 0.5
    assert evaluation.binary_auc(
        [True, False, True, False], [1.0, 1.0, 2.0, 0.0]
    ) == pytest.approx(0.875)
    assert evaluation.binary_auc([False, True], [1.0, 0.0]) == 0.0
    assert evaluation.binary_auc([False, True], [0.0, 1.0]) == 1.0
    assert evaluation.binary_auc([True, True], [0.0, 1.0]) is None


def test_exact_mcnemar_uses_two_sided_exact_binomial_tail() -> None:
    all_a = evaluation.exact_mcnemar(
        [True, True, True, True], [False, False, False, False]
    )
    assert all_a == {
        "a_only": 4,
        "b_only": 0,
        "discordant": 4,
        "p_value": 0.125,
        "n": 4,
    }

    balanced = evaluation.exact_mcnemar([True, False], [False, True])
    assert balanced["a_only"] == 1
    assert balanced["b_only"] == 1
    assert balanced["p_value"] == 1.0

    identical = evaluation.exact_mcnemar([True, False], [True, False])
    assert identical["discordant"] == 0
    assert identical["p_value"] == 1.0


def test_episode_cluster_bootstrap_is_deterministic_and_pairs_differences() -> None:
    episodes = [_episode(index) for index in range(7)]
    perfect: dict[str, float] = {}
    reverse: dict[str, float] = {}
    mixed: dict[str, float] = {}
    for episode_index, episode in enumerate(episodes):
        for candidate in episode.candidates:
            positive = candidate.label == "load_bearing"
            perfect[candidate.uid] = float(positive)
            reverse[candidate.uid] = float(not positive)
            mixed[candidate.uid] = float(
                positive if episode_index % 3 else not positive
            )
    scores = {"perfect": perfect, "mixed": mixed, "reverse": reverse}
    order = ["perfect", "mixed", "reverse"]

    first = evaluation.build_metric_scope(episodes, order, scores, budgets=[1])
    second = evaluation.build_metric_scope(episodes, order, scores, budgets=[1])
    evaluation.attach_episode_cluster_bootstrap(
        first, episodes, order, scores, samples=64, seed=23
    )
    evaluation.attach_episode_cluster_bootstrap(
        second, episodes, order, scores, samples=64, seed=23
    )

    assert first == second
    assert first["bootstrap"] == {
        "method": "episode_cluster_percentile",
        "confidence": 0.95,
        "samples_requested": 64,
        "seed": 23,
        "skipped": False,
    }
    for condition in order:
        classification = first["conditions"][condition]["classification"]
        assert (
            classification["pooled_auc_bootstrap"]["bootstrap_samples_effective"]
            == 64
        )
        assert (
            classification["within_episode_auc_bootstrap"]
            ["bootstrap_samples_effective"]
            == 64
        )

    perfect_vs_reverse = next(
        comparison
        for comparison in first["paired_auc_differences"]
        if comparison["a"] == "perfect" and comparison["b"] == "reverse"
    )
    assert perfect_vs_reverse["direction"] == "a_minus_b"
    assert perfect_vs_reverse["pooled_auc_difference"]["estimate"] == 1.0
    assert perfect_vs_reverse["pooled_auc_difference"]["ci_95"] == [1.0, 1.0]
    assert (
        perfect_vs_reverse["within_episode_auc_difference"]["estimate"] == 1.0
    )
    assert perfect_vs_reverse["within_episode_auc_difference"]["ci_95"] == [
        1.0,
        1.0,
    ]
    assert (
        perfect_vs_reverse["pooled_auc_difference"]["probability_gt_zero"]
        == 1.0
    )


def test_no_harm_scope_summary_reports_adapter_delta_and_exact_pairing() -> None:
    episodes = [SimpleNamespace(uid=f"episode-{index}") for index in range(4)]
    base_correct = [True, True, True, False]
    adapter_correct = [True, False, False, False]
    details = {
        "original": {
            episode.uid: {"answer": "base", "correct": correct}
            for episode, correct in zip(episodes, base_correct)
        },
        "adapter": {
            episode.uid: {"answer": "adapter", "correct": correct}
            for episode, correct in zip(episodes, adapter_correct)
        },
    }

    summary = evaluation.no_harm_scope_summary(
        episodes, ["original", "adapter"], details
    )
    assert summary["conditions"]["original"] == {
        "accuracy": 0.75,
        "n_episodes": 4,
    }
    assert summary["conditions"]["adapter"] == {
        "accuracy": 0.25,
        "n_episodes": 4,
    }
    comparison = summary["comparisons"][0]
    assert comparison["adapter_minus_base_accuracy"] == -0.5
    assert comparison["base_only_correct"] == 2
    assert comparison["adapter_only_correct"] == 0
    assert comparison["discordant"] == 2
    assert comparison["exact_mcnemar_p_value"] == 0.5
    assert comparison["n"] == 4


def test_evaluation_records_all_batching_provenance() -> None:
    args = SimpleNamespace(batch_size=16, qa_batch_size=1, no_harm_batch_size=4)
    assert evaluation.evaluation_batch_provenance(args) == {
        "admission_batch_size": 16,
        "qa_batch_size": 1,
        "no_harm_batch_size": 4,
    }


def test_main_refuses_existing_output_before_loading_evaluation_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("sealed\n", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        pytest.fail("evaluation data/model path was reached before output protection")

    monkeypatch.setattr(evaluation, "load_eval_specs", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_memory_rl.py",
            "--model",
            "unused-model",
            "--spec",
            "decoupled=missing-results.json::missing-battery.json",
            "--out",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        evaluation.main()

    assert exc_info.value.code == 2
    assert "refusing to overwrite existing output" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "sealed\n"
