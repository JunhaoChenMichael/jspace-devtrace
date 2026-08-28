from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.audit_unified_ood_shared_conditions import (  # noqa: E402
    FORMAL_CONFIG,
    SourceContract,
    audit_shared_condition_reproducibility,
    main,
    write_report_exclusive,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _score(condition: str, candidate_index: int, episode_index: int) -> float:
    condition_offset = {
        "original": 0.01,
        "sft-w-s0-k2": 0.02,
        "rl-qa-s0-k2": 0.03,
        "workspace": 0.04,
        "oracle": 0.05,
    }.get(condition, 0.06)
    # Candidate zero is load-bearing and always ranks first.  The remaining
    # candidates have stable strict ordering, so tolerance-only perturbations do
    # not alter the independently recomputed selection.
    return float(1.0 - 0.2 * candidate_index + condition_offset + episode_index * 0.001)


def _selection(items: list[dict], condition: str) -> dict:
    indices = sorted(
        range(len(items)),
        key=lambda index: (-items[index]["scores"][condition], index),
    )[:2]
    positives = [index for index, item in enumerate(items) if item["label"] == "load_bearing"]
    chosen = set(indices)
    selected_positive = [index for index in positives if index in chosen]
    concepts = [items[index]["concept"] for index in indices]
    qa_concepts = (
        concepts
        if condition == "oracle"
        else [items[index]["concept"] for index in sorted(set(indices))]
    )
    return {
        "selected_indices": indices,
        "selected_candidate_uids": [items[index]["uid"] for index in indices],
        "selected_concepts": concepts,
        "contains_load_bearing": bool(selected_positive),
        "contains_all_load_bearing": bool(positives)
        and len(selected_positive) == len(positives),
        "selected_load_bearing": len(selected_positive),
        "total_load_bearing": len(positives),
        "load_bearing_recall": len(selected_positive) / len(positives),
        "qa": {
            "selected_concepts": qa_concepts,
            "answer": f"answer:{','.join(qa_concepts)}",
            "correct": condition != "original",
        },
    }


def _payload(
    source: str,
    *,
    extra_condition: str,
    skip_no_harm: bool,
) -> dict:
    conditions = [
        "original",
        "sft-w-s0-k2",
        "rl-qa-s0-k2",
        extra_condition,
        "workspace",
        "oracle",
    ]
    adapters = {
        "sft-w-s0-k2": "sealed/sft/best-step-300",
        "rl-qa-s0-k2": "sealed/rlqa0/best-step-300",
        extra_condition: f"nonshared/{extra_condition}",
    }
    config = {
        **FORMAL_CONFIG,
        "specs": [
            {
                "name": source,
                "source": source,
                "results_path": f"results/{source}.json",
                "battery_path": f"battery/{source}.json",
            }
        ],
        "adapters": adapters,
        "rating_json": {},
        "embedding_model": None,
        "skip_no_harm": skip_no_harm,
        "no_harm_batch_size": 1,
    }
    per_item: list[dict] = []
    episode_items: list[list[dict]] = []
    for episode_index, n_candidates in enumerate((2, 3)):
        episode_uid = f"{source}:episode:{episode_index:06d}"
        items = []
        for candidate_index in range(n_candidates):
            uid = f"{episode_uid}:candidate:{candidate_index:03d}"
            scores = {
                condition: _score(condition, candidate_index, episode_index)
                for condition in conditions
            }
            row = {
                "uid": uid,
                "episode_uid": episode_uid,
                "source": source,
                "source_episode": episode_index,
                "candidate_index": candidate_index,
                "concept": f"concept-{episode_index}-{candidate_index}",
                "label": "load_bearing" if candidate_index == 0 else "distractor",
                "scores": scores,
                "model_log_odds": {
                    condition: scores[condition] - 0.5
                    for condition in adapters
                },
            }
            per_item.append(row)
            items.append(row)
        episode_items.append(items)

    per_episode = []
    for episode_index, items in enumerate(episode_items):
        policies = {
            condition: {
                "within_episode_auc": 1.0,
                "selections": {"2": _selection(items, condition)},
            }
            for condition in conditions
        }
        oracle = policies["oracle"]["selections"]["2"]
        episode = {
            "uid": f"{source}:episode:{episode_index:06d}",
            "source": source,
            "source_episode": episode_index,
            "probe_question": f"question {episode_index}",
            "gold_answer": "gold",
            "policies": policies,
            "refs": {
                "oracle@2": {
                    "selected_indices": oracle["selected_indices"],
                    "selected_concepts": oracle["selected_concepts"],
                    "answer": oracle["qa"]["answer"],
                    "correct": oracle["qa"]["correct"],
                },
                "full_context": {"answer": "base-full", "correct": True},
                "no_memory": {"answer": "base-empty", "correct": False},
            },
        }
        if not skip_no_harm:
            episode["no_harm_full_context"] = {
                condition: {
                    "answer": f"full:{condition}:{episode_index}",
                    "correct": condition != "original",
                }
                for condition in ("original", *adapters)
            }
        per_episode.append(episode)

    return {
        "schema_version": 1,
        "config": config,
        "condition_order": conditions,
        "metrics": {},
        "no_harm": {"skipped": skip_no_harm, "summary": {}},
        "refs": {"skipped": False, "summary": {}},
        "mcnemar": {"skipped": False},
        "per_item": per_item,
        "per_episode": per_episode,
    }


@pytest.fixture
def campaign(tmp_path: Path):
    contracts = {
        "decoupled": SourceContract(
            "decoupled",
            2,
            5,
            "results/decoupled.json",
            "battery/decoupled.json",
            True,
            None,
        ),
        "compositional": SourceContract(
            "compositional",
            2,
            5,
            "results/compositional.json",
            "battery/compositional.json",
            None,
            None,
        ),
    }
    pairs = {}
    payloads = {}
    for source in contracts:
        old_skip = source == "decoupled"
        old = _payload(source, extra_condition="hybrid-s0-k2-lw0p25", skip_no_harm=old_skip)
        # Decoupled intentionally exercises the case where new has no-harm data
        # but the old sensitivity raw does not.  This must remain explicitly N/A.
        new = _payload(source, extra_condition="rl-qa-s1-k2", skip_no_harm=False)
        old_path = tmp_path / f"{source}-old.json"
        new_path = tmp_path / f"{source}-new.json"
        _write(old_path, old)
        _write(new_path, new)
        pairs[source] = (old_path, new_path)
        payloads[source] = (old, new, old_path, new_path)
    return contracts, pairs, payloads


def _audit(campaign):
    contracts, pairs, _ = campaign
    return audit_shared_condition_reproducibility(pairs, contracts=contracts)


def test_passes_shared_semantics_and_marks_decoupled_no_harm_na(campaign) -> None:
    contracts, pairs, payloads = campaign
    # A tiny floating-point difference within the declared tolerance is allowed.
    payloads["decoupled"][1]["per_item"][0]["scores"]["sft-w-s0-k2"] += 5e-9
    _write(payloads["decoupled"][3], payloads["decoupled"][1])

    report = audit_shared_condition_reproducibility(pairs, contracts=contracts)
    assert report["status"] == "pass", report
    decoupled = report["sources"]["decoupled"]
    assert (
        decoupled["full_context_scope"]["condition_specific_adapter_enabled"]["status"]
        == "not-applicable"
    )
    assert "adapter_enabled_full_context" not in decoupled["comparison"]["counts"]
    assert decoupled["comparison"]["counts"]["admission_scores"]["records"] == 25
    compositional = report["sources"]["compositional"]
    assert compositional["comparison"]["counts"]["adapter_enabled_full_context"]["records"] == 6


def test_fails_on_admission_score_beyond_tolerance(campaign) -> None:
    _, _, payloads = campaign
    payloads["decoupled"][1]["per_item"][0]["scores"]["sft-w-s0-k2"] += 1e-4
    _write(payloads["decoupled"][3], payloads["decoupled"][1])
    report = _audit(campaign)
    comparison = report["sources"]["decoupled"]["comparison"]
    assert report["status"] == "fail"
    assert comparison["issue_counts_by_code"]["numeric_mismatch"] >= 1


def test_fails_closed_on_item_uid_order_even_when_both_could_look_complete(campaign) -> None:
    _, _, payloads = campaign
    new = payloads["compositional"][1]
    new["per_item"][0], new["per_item"][1] = new["per_item"][1], new["per_item"][0]
    _write(payloads["compositional"][3], new)
    report = _audit(campaign)
    validation = report["sources"]["compositional"]["new"]["validation"]
    assert report["status"] == "fail"
    assert validation["status"] == "fail"
    assert "candidate_order_mismatch" in validation["issue_counts_by_code"]
    assert report["sources"]["compositional"]["comparison"]["executed"] is False
    assert report["sources"]["compositional"]["comparison"]["status"] == "not-run"


def test_fails_on_selection_qa_semantic_difference(campaign) -> None:
    _, _, payloads = campaign
    new = payloads["decoupled"][1]
    new["per_episode"][0]["policies"]["original"]["selections"]["2"]["qa"][
        "answer"
    ] = "different answer"
    _write(payloads["decoupled"][3], new)
    report = _audit(campaign)
    comparison = report["sources"]["decoupled"]["comparison"]
    assert report["status"] == "fail"
    assert comparison["issue_counts_by_code"]["semantic_mismatch"] >= 1
    assert any(".qa.answer" in row["path"] for row in comparison["issue_examples"])


def test_compositional_full_context_is_compared_but_decoupled_is_na(campaign) -> None:
    _, _, payloads = campaign
    # Decoupled per-condition full-context is outside the available shared raw.
    payloads["decoupled"][1]["per_episode"][0]["no_harm_full_context"][
        "rl-qa-s0-k2"
    ]["answer"] = "ignored because old skipped"
    _write(payloads["decoupled"][3], payloads["decoupled"][1])
    report = _audit(campaign)
    assert report["status"] == "pass", report

    # The same endpoint is present on both Compositional raws and must match.
    payloads["compositional"][1]["per_episode"][0]["no_harm_full_context"][
        "rl-qa-s0-k2"
    ]["answer"] = "must fail"
    _write(payloads["compositional"][3], payloads["compositional"][1])
    report = _audit(campaign)
    comparison = report["sources"]["compositional"]["comparison"]
    assert report["status"] == "fail"
    assert any("no_harm_full_context" in row["path"] for row in comparison["issue_examples"])


def test_internal_selection_is_recomputed_from_scores(campaign) -> None:
    _, _, payloads = campaign
    new = payloads["compositional"][1]
    selection = new["per_episode"][0]["policies"]["workspace"]["selections"]["2"]
    selection["selected_indices"] = list(reversed(selection["selected_indices"]))
    selection["selected_candidate_uids"] = list(reversed(selection["selected_candidate_uids"]))
    selection["selected_concepts"] = list(reversed(selection["selected_concepts"]))
    _write(payloads["compositional"][3], new)
    report = _audit(campaign)
    validation = report["sources"]["compositional"]["new"]["validation"]
    assert report["status"] == "fail"
    assert "selection_not_reproducible_from_scores" in validation["issue_counts_by_code"]


def test_qa_concepts_use_candidate_order_not_score_rank_order(campaign) -> None:
    contracts, pairs, payloads = campaign
    for source in ("decoupled", "compositional"):
        for payload in payloads[source][:2]:
            items = [
                row
                for row in payload["per_item"]
                if row["source_episode"] == 0
            ]
            items[1]["scores"]["original"] = items[0]["scores"]["original"] + 0.1
            selection = _selection(items, "original")
            assert selection["selected_indices"] == [1, 0]
            assert selection["qa"]["selected_concepts"] == [
                items[0]["concept"],
                items[1]["concept"],
            ]
            payload["per_episode"][0]["policies"]["original"]["selections"][
                "2"
            ] = selection
        _write(payloads[source][2], payloads[source][0])
        _write(payloads[source][3], payloads[source][1])
    report = audit_shared_condition_reproducibility(pairs, contracts=contracts)
    assert report["status"] == "pass", report


def test_oracle_qa_preserves_oracle_reference_order(campaign) -> None:
    contracts, pairs, payloads = campaign
    for source in ("decoupled", "compositional"):
        for payload in payloads[source][:2]:
            items = [
                row
                for row in payload["per_item"]
                if row["source_episode"] == 0
            ]
            items[0]["label"] = "distractor"
            items[1]["label"] = "load_bearing"
            items[0]["scores"]["oracle"] = 0.0
            items[1]["scores"]["oracle"] = 1.0
            selection = _selection(items, "oracle")
            assert selection["selected_indices"] == [1, 0]
            assert selection["qa"]["selected_concepts"] == selection["selected_concepts"]
            payload["per_episode"][0]["policies"]["oracle"]["selections"][
                "2"
            ] = selection
            payload["per_episode"][0]["refs"]["oracle@2"] = {
                "selected_indices": selection["selected_indices"],
                "selected_concepts": selection["selected_concepts"],
                "answer": selection["qa"]["answer"],
                "correct": selection["qa"]["correct"],
            }
        _write(payloads[source][2], payloads[source][0])
        _write(payloads[source][3], payloads[source][1])
    report = audit_shared_condition_reproducibility(pairs, contracts=contracts)
    assert report["status"] == "pass", report


def test_shared_adapter_path_difference_fails_protocol(campaign) -> None:
    _, _, payloads = campaign
    payloads["decoupled"][1]["config"]["adapters"]["rl-qa-s0-k2"] = "wrong/path"
    _write(payloads["decoupled"][3], payloads["decoupled"][1])
    report = _audit(campaign)
    comparison = report["sources"]["decoupled"]["comparison"]
    assert report["status"] == "fail"
    assert any("config[shared].adapters" in row["path"] for row in comparison["issue_examples"])


def test_existing_output_is_refused_before_any_input_read(tmp_path: Path, capsys) -> None:
    output = tmp_path / "exists.json"
    output.write_text("sealed\n", encoding="utf-8")
    missing = tmp_path / "does-not-exist.json"
    result = main(
        [
            "--decoupled-old",
            str(missing),
            "--decoupled-new",
            str(missing),
            "--compositional-old",
            str(missing),
            "--compositional-new",
            str(missing),
            "--out",
            str(output),
        ]
    )
    assert result == 2
    assert output.read_text(encoding="utf-8") == "sealed\n"
    assert "overwrite" in capsys.readouterr().err


def test_missing_input_writes_new_fail_report_exclusively(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "audit.json"
    result = main(
        [
            "--decoupled-old",
            str(missing),
            "--decoupled-new",
            str(missing),
            "--compositional-old",
            str(missing),
            "--compositional-new",
            str(missing),
            "--out",
            str(output),
        ]
    )
    assert result == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert (
        report["sources"]["decoupled"]["old"]["validation"][
            "issue_counts_by_code"
        ]["input_read_error"]
        == 1
    )
    with pytest.raises(FileExistsError):
        write_report_exclusive(output, report)


def test_duplicate_json_key_and_nonfinite_tolerance_fail_closed(
    tmp_path: Path, campaign
) -> None:
    contracts, pairs, payloads = campaign
    payloads["decoupled"][3].write_text('{"schema_version":1,"schema_version":1}\n')
    report = audit_shared_condition_reproducibility(pairs, contracts=contracts)
    validation = report["sources"]["decoupled"]["new"]["validation"]
    assert report["status"] == "fail"
    assert "invalid_json" in validation["issue_counts_by_code"]
    with pytest.raises(ValueError):
        audit_shared_condition_reproducibility(
            pairs, contracts=contracts, abs_tol=float("nan")
        )
