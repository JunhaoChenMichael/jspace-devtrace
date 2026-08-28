from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
VALIDATOR = SRC_ROOT / "analysis" / "validate_memory_rl_run.py"
sys.path.insert(0, str(SRC_ROOT))

from analysis.validate_memory_rl_run import validate_run  # noqa: E402
from experiments.train_memory_rl import diagnose, write_json  # noqa: E402
from memory_rl.data import file_sha256  # noqa: E402
from memory_rl.reporter_correlations import (  # noqa: E402
    summarize_reporter_correlations,
    within_episode_utility_auc,
)
from memory_rl.training_diagnostics import summarize_selector_training  # noqa: E402


def _write_strict(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _append_strict(path: Path, values) -> None:
    path.write_text(
        "".join(json.dumps(value, allow_nan=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _object_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_counts(profile: str) -> dict[str, tuple[int, int]]:
    if profile == "formal":
        return {
            "explicit": (70, 18),
            "evoked": (60, 15),
            "evoked_g2": (45, 12),
        }
    return {"explicit": (4, 1)}


def _make_manifest_and_specs(root: Path, profile: str):
    sources = {}
    specs = []
    for source, (train_count, validation_count) in _source_counts(profile).items():
        battery = root / "inputs" / f"{source}_battery.json"
        results = root / "inputs" / f"{source}_results.json"
        _write_strict(battery, {"fixture": source, "kind": "battery"})
        _write_strict(results, {"fixture": source, "kind": "results"})
        sources[source] = {
            "battery_path": str(battery.resolve()),
            "battery_sha256": file_sha256(battery),
            "results_path": str(results.resolve()),
            "results_sha256": file_sha256(results),
            "train_episode_ids": [
                f"{source}:episode:train:{index:03d}" for index in range(train_count)
            ],
            "validation_episode_ids": [
                f"{source}:episode:validation:{index:03d}"
                for index in range(validation_count)
            ],
        }
        specs.append(
            {
                "source": source,
                "battery_path": str(battery.resolve()),
                "results_path": str(results.resolve()),
            }
        )
    manifest = {
        "schema_version": 1,
        "algorithm": "per-source sha256(seed:episode_id), exact ceil holdout",
        "seed": 0,
        "validation_fraction": 0.2,
        "sources": sources,
        "train_episode_count": sum(value[0] for value in _source_counts(profile).values()),
        "validation_episode_count": sum(value[1] for value in _source_counts(profile).values()),
    }
    manifest["manifest_sha256"] = _object_hash(manifest)
    return manifest, specs


def _dropout_audit() -> dict:
    return {
        "dropout_modules_zeroed": 0,
        "config_fields_zeroed": [],
        "dropout_modules_found": [],
        "dropout_modules_modified": [],
        "config_fields_found": [],
        "config_fields_modified": [],
        "remaining_nonzero": [],
        "postcondition_satisfied": True,
    }


def _make_adapter(path: Path, step: int, metric: float | None = None) -> None:
    path.mkdir(parents=True)
    state = {"step": step}
    if metric is not None:
        state["metric"] = metric
    _write_strict(path / "training_state.json", state)
    _write_strict(path / "adapter_config.json", {"peft_type": "LORA"})
    _write_strict(path / "tokenizer_config.json", {"fixture": True})
    (path / "adapter_model.safetensors").write_bytes(b"fixture-weights")


def _reporter_validation(event: str, step: int, auc: float) -> dict:
    return {
        "auc": auc,
        "within_episode_auc": auc - 0.01,
        "containment": 0.75,
        "yes_rate": 0.5,
        "event": event,
        "step": step,
        "selection_metric": auc,
    }


def _selector_validation(event: str, step: int, accuracy: float) -> dict:
    return {
        "qa_accuracy": accuracy,
        "containment": 0.75,
        "workspace_set_reward": 0.625,
        "episodes": 1,
        "event": event,
        "step": step,
        "selection_metric": accuracy,
    }


def _make_run(tmp_path: Path, profile: str, mode: str) -> Path:
    run = tmp_path / f"{profile}-{mode}"
    run.mkdir(parents=True)
    manifest, specs = _make_manifest_and_specs(tmp_path, profile)
    train_count = manifest["train_episode_count"]
    validation_count = manifest["validation_episode_count"]

    if profile == "smoke":
        model = "Qwen/Qwen2.5-3B-Instruct"
        teacher = "Qwen/Qwen2.5-7B-Instruct"
        match = False
        override = True
    elif profile == "formal":
        model = teacher = "Qwen/Qwen2.5-7B-Instruct"
        match = True
        override = False
    else:
        model = teacher = "local/matched-policy"
        match = True
        override = False

    config = {
        "mode": mode,
        "model": model,
        "teacher_tag": "fixture",
        "workspace_teacher_model": teacher,
        "allow_teacher_mismatch": override,
        "teacher_mismatch_override": override,
        "teacher_matches_policy_reference": match,
        "fixed_workspace_teacher": True,
        "constrained_yes_no_rollouts": True,
        "probe_visible_to_policy": False,
        "frozen_recall_adapter_disabled": mode in ("rl-qa", "rl-hybrid"),
        "workspace_reward_used_for_optimization": mode == "rl-hybrid",
        "dry_run": profile == "dry-run",
        "seed": 0,
        "split_seed": 0,
        "val_fraction": 0.2,
        "train_spec": specs,
        "train_episode_count": train_count,
        "validation_episode_count": validation_count,
        "limit_train_episodes": 0,
        "limit_validation_episodes": 0,
        "val_eval_episodes": 0,
        "max_steps": 1,
        "eval_every": 1,
        "diagnostic_every": 1,
        "group_size": 2,
        "budget": 2,
        "workspace_objective": "rank-continuous",
        "workspace_set_reward": "mean",
        "lambda_qa": 1.0,
        "lambda_w": 0.5,
        "lora_dropout": 0.0,
        "effective_advantage_normalization": "center" if mode == "rl-w" else "zscore",
        "software_versions": {
            "python": "3.12",
            "cuda": "12.1",
            "torch": "2.5",
            "transformers": "4.50",
            "peft": "0.14",
            "numpy": "2.0",
            "scikit-learn": "1.5",
        },
    }
    if profile != "dry-run":
        config.update(
            {
                "resolved_model_name_or_path": model,
                "resolved_model_commit": "fixture-commit",
                "trainable_parameter_count": 100,
                "total_parameter_count": 10_000,
            }
        )
    _write_strict(run / "run_config.json", config)
    _write_strict(run / "split_manifest.json", manifest)
    if profile == "dry-run":
        return run

    if mode in ("sft-w", "rl-w"):
        baseline = _reporter_validation("baseline", 0, 0.55)
        validation = _reporter_validation("validation", 1, 0.65)
    else:
        baseline = _selector_validation("baseline", 0, 0.4)
        validation = _selector_validation("validation", 1, 0.5)

    if mode == "sft-w":
        train = {
            "event": "train",
            "step": 1,
            "loss": 0.4,
            "reward_mean": None,
            "reward_std": None,
            "kl": 0.0,
            "yes_rate": 0.5,
            "grad_norm": 0.3,
        }
        dropout = None
    elif mode == "rl-w":
        train = {
            "event": "train",
            "step": 1,
            "candidate_id": "explicit:episode:train:000:candidate:000",
            "loss": 0.2,
            "reward_mean": 0.0,
            "reward_std": 0.5,
            "kl": 0.02,
            "grad_norm": 0.4,
            "yes_rate": 0.5,
            "workspace_percentile": 0.75,
        }
        _append_strict(
            run / "rollouts.jsonl",
            [
                {
                    "step": 1,
                    "episode_id": "explicit:episode:train:000",
                    "candidate_ids": [train["candidate_id"]],
                    "actions": ["Yes", "No"],
                    "rewards": [0.5, -0.5],
                    "kl": 0.02,
                }
            ],
        )
        dropout = _dropout_audit()
    else:
        rewards = [11.0 / 12.0, 1.0 / 6.0] if mode == "rl-hybrid" else [1.0, 0.0]
        train = {
            "event": "train",
            "step": 1,
            "episode_id": "explicit:episode:train:000",
            "loss": 0.25,
            "reward_mean": sum(rewards) / 2,
            "reward_std": abs(rewards[0] - rewards[1]) / 2,
            "qa_reward": 0.5,
            "workspace_reward": 0.625,
            "kl": 0.02,
            "yes_rate": None,
            "grad_norm": 0.4,
            "candidate_count": 3,
            "number_unique_selected_sets": 2,
            "fraction_unique_selected_sets": 1.0,
            "mixed_QA_reward_group": True,
            "containment_rate": 1.0,
            "mixed_containment_group": False,
            "policy_set_entropy": 1.0,
            "normalized_policy_set_entropy": 0.9,
            "yes_probabilities": [0.2, 0.5, 0.8],
        }
        candidates = [
            f"explicit:episode:train:000:candidate:{index:03d}" for index in range(3)
        ]
        rows = [
            {
                "step": 1,
                "group_index": 0,
                "episode_id": train["episode_id"],
                "candidate_ids": candidates,
                "selection_probability": 0.3,
                "selection_log_probability": math.log(0.3),
                "selected_set": [candidates[1], candidates[2]],
                "selected_concepts": ["middle", "high"],
                "contains_load_bearing": True,
                "answer": "correct",
                "QA_correct": True,
                "oracle_QA_correct": True,
                "full_context_correct": True,
                "workspace_scores": [0.0, 1.0, 2.0],
                "verbal_scores": [None, 0.4, 0.8],
                "QA_reward": 1.0,
                "workspace_reward": 0.75,
                "reward": rewards[0],
                "KL": 0.02,
                "failure_type": "none",
            },
            {
                "step": 1,
                "group_index": 1,
                "episode_id": train["episode_id"],
                "candidate_ids": candidates,
                "selection_probability": 0.4,
                "selection_log_probability": math.log(0.4),
                "selected_set": [candidates[0], candidates[2]],
                "selected_concepts": ["low", "high"],
                "contains_load_bearing": True,
                "answer": "wrong",
                "QA_correct": False,
                "oracle_QA_correct": True,
                "full_context_correct": True,
                "workspace_scores": [0.0, 1.0, 2.0],
                "verbal_scores": [None, 0.4, 0.8],
                "QA_reward": 0.0,
                "workspace_reward": 0.5,
                "reward": rewards[1],
                "KL": 0.02,
                "failure_type": "recall_or_composition",
            },
        ]
        _append_strict(run / "rollouts.jsonl", rows)
        dropout = _dropout_audit()

    _append_strict(run / "metrics.jsonl", [baseline, train, validation])
    if dropout is not None:
        _write_strict(run / "dropout_audit.json", dropout)

    best = run / "best-step-1"
    final = run / "final_adapter"
    _make_adapter(best, 1, validation["selection_metric"])
    _make_adapter(final, 1)
    _write_strict(
        run / "best_checkpoint.json",
        {"path": str(best.resolve()), "step": 1, "metric": validation["selection_metric"]},
    )
    summary = {
        "mode": mode,
        "steps_completed": 1,
        "best_validation_metric": validation["selection_metric"],
        "best_checkpoint": str(best.resolve()),
        "final_adapter": str(final.resolve()),
        "baseline_validation": baseline,
        "last_validation": validation,
        "workspace_teacher_model": teacher,
        "teacher_matches_policy_reference": match,
        "effective_advantage_normalization": config["effective_advantage_normalization"],
        "dropout_audit": dropout,
        "elapsed_seconds": 1.5,
        "diagnostics": [],
        "selector_training_diagnostics": (
            summarize_selector_training([train], window_size=1)
            if mode in ("rl-qa", "rl-hybrid")
            else None
        ),
    }
    _write_strict(run / "summary.json", summary)
    return run


def _codes(report) -> set[str]:
    return {error["code"] for error in report["errors"]}


@pytest.mark.parametrize("mode", ["sft-w", "rl-w", "rl-qa", "rl-hybrid"])
def test_dry_run_contract_accepts_all_modes_and_rejects_changed_source(
    tmp_path: Path, mode: str
) -> None:
    run = _make_run(tmp_path, "dry-run", mode)
    report = validate_run(run, profile="dry-run")
    assert report["status"] == "pass", report
    assert report["details"]["split"] == {
        "sources": ["explicit"],
        "train_episode_count": 4,
        "validation_episode_count": 1,
    }

    source = next((tmp_path / "inputs").glob("*_battery.json"))
    source.write_text("changed", encoding="utf-8")
    report = validate_run(run, profile="dry-run")
    assert report["status"] == "fail"
    assert "source_hash_mismatch" in _codes(report)


def test_strict_json_writer_uses_null_for_na_and_rejects_nonfinite(tmp_path: Path) -> None:
    path = tmp_path / "strict.json"
    write_json(path, {"not_applicable": None, "finite": 1.0})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "not_applicable": None,
        "finite": 1.0,
    }
    with pytest.raises(ValueError, match="Out of range float values"):
        write_json(path, {"applicable": float("nan")})
    assert diagnose(
        [
            {
                "reward_mean": None,
                "reward_std": None,
                "kl": 0.0,
                "yes_rate": 0.5,
            }
        ],
        "sft-w",
    ) == []


def test_smoke_rl_w_cross_checks_metrics_rollout_teacher_and_dropout(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "smoke", "rl-w")
    report = validate_run(run, profile="smoke")
    assert report["status"] == "pass", report
    assert "teacher_mismatch_smoke_only" in {
        warning["code"] for warning in report["warnings"]
    }

    rollout = json.loads((run / "rollouts.jsonl").read_text(encoding="utf-8"))
    rollout["rewards"] = [0.5, 0.5]
    _append_strict(run / "rollouts.jsonl", [rollout])
    report = validate_run(run, profile="smoke")
    assert "rollout_aggregate_mismatch" in _codes(report)


def test_selector_and_formal_contract_reject_reward_set_and_dropout_tampering(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path, "formal", "rl-hybrid")
    report = validate_run(
        run,
        profile="formal",
        expected_config={"beta": None},
    )
    # The deliberately wrong lock proves that formal recipe comparison is hard.
    assert "locked_config_mismatch" in _codes(report)

    report = validate_run(run, profile="formal")
    assert report["status"] == "pass", report

    rows = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines()]
    rows[0]["selected_set"] = [rows[0]["selected_set"][0]] * 2
    _append_strict(run / "rollouts.jsonl", rows)
    report = validate_run(run, profile="formal")
    assert "selected_set_invalid" in _codes(report)


def test_validator_recomputes_reporter_correlations_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path, "smoke", "rl-hybrid")
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    config.update(
        {
            "reporter_correlation_diagnostics": True,
            "reporter_correlation_scope": "fixed ID validation candidates",
            "reporter_correlations_used_for_selection": False,
            "reporter_correlation_artifact": "reporter_correlations.jsonl",
            "reporter_correlation_bootstrap_samples": 25,
            "reporter_correlation_bootstrap_seed": 3,
        }
    )
    _write_strict(run / "run_config.json", config)
    rows = [
        {
            "episode_id": "explicit:episode:validation:000",
            "candidate_id": "explicit:episode:validation:000:candidate:000",
            "source": "explicit",
            "v_rl": 0.1,
            "w_ref": 0.2,
            "y_utility": 0,
        },
        {
            "episode_id": "explicit:episode:validation:000",
            "candidate_id": "explicit:episode:validation:000:candidate:001",
            "source": "explicit",
            "v_rl": 0.8,
            "w_ref": 0.9,
            "y_utility": 1,
        },
        {
            "episode_id": "explicit:episode:validation:000",
            "candidate_id": "explicit:episode:validation:000:candidate:002",
            "source": "explicit",
            "v_rl": 0.3,
            "w_ref": 0.4,
            "y_utility": 0,
        },
    ]
    correlation_summary = summarize_reporter_correlations(
        rows, bootstrap_samples=25, bootstrap_seed=3
    )
    metrics = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation_metrics = []
    diagnostic_records = []
    for metric in metrics:
        if metric["event"] not in ("baseline", "validation"):
            continue
        metric.update(
            {
                "verbal_auc": correlation_summary["utility_auc"],
                "verbal_within_episode_auc": within_episode_utility_auc(rows),
                "verbal_yes_rate": correlation_summary["yes_rate"],
                "combined_reward": (
                    metric["qa_accuracy"] + 0.5 * metric["workspace_set_reward"]
                )
                / 1.5,
                "reporter_correlations": correlation_summary,
            }
        )
        validation_metrics.append(metric)
        diagnostic_records.append(
            {
                "schema_version": 1,
                "scope": "fixed_id_validation",
                "event": metric["event"],
                "step": metric["step"],
                "summary": correlation_summary,
                "rows": rows,
            }
        )
    _append_strict(run / "metrics.jsonl", metrics)
    _append_strict(run / "reporter_correlations.jsonl", diagnostic_records)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["baseline_validation"] = validation_metrics[0]
    summary["last_validation"] = validation_metrics[1]
    summary["best_validation"] = validation_metrics[1]
    summary["reporter_correlation_artifact"] = str(
        run / "reporter_correlations.jsonl"
    )
    _write_strict(run / "summary.json", summary)

    report = validate_run(run, profile="smoke")
    assert report["status"] == "pass", report
    assert report["details"]["reporter_correlations"] == {
        "records": 2,
        "candidate_counts": [3, 3],
        "bootstrap_samples": 25,
        "bootstrap_seed": 3,
    }

    diagnostic_records[1]["rows"][0]["v_rl"] = 0.7
    _append_strict(run / "reporter_correlations.jsonl", diagnostic_records)
    report = validate_run(run, profile="smoke")
    assert "reporter_correlation_summary_mismatch" in _codes(report)

    # Restore rollouts, then make the independently recorded dropout post-scan fail.
    run = _make_run(tmp_path / "second", "formal", "rl-hybrid")
    dropout = json.loads((run / "dropout_audit.json").read_text())
    dropout["remaining_nonzero"] = [{"kind": "module", "name": "bad", "value": 0.1}]
    dropout["postcondition_satisfied"] = False
    _write_strict(run / "dropout_audit.json", dropout)
    summary = json.loads((run / "summary.json").read_text())
    summary["dropout_audit"] = dropout
    _write_strict(run / "summary.json", summary)
    report = validate_run(run, profile="formal")
    assert "dropout_remaining_nonzero" in _codes(report)


def test_formal_sft_checkpoint_and_teacher_are_hard_requirements(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "formal", "sft-w")
    report = validate_run(run, profile="formal")
    assert report["status"] == "pass", report

    (run / "best-step-1" / "adapter_model.safetensors").write_bytes(b"")
    config = json.loads((run / "run_config.json").read_text())
    config["teacher_mismatch_override"] = True
    config["allow_teacher_mismatch"] = True
    _write_strict(run / "run_config.json", config)
    report = validate_run(run, profile="formal")
    assert {"adapter_weights_missing", "teacher_mismatch_forbidden"}.issubset(_codes(report))


def test_cli_emits_one_strict_json_report_and_nonzero_on_manifest_mismatch(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path, "dry-run", "rl-w")
    process = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(run),
            "--profile",
            "dry-run",
            "--expected-manifest-sha256",
            "0" * 64,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 1
    report = json.loads(
        process.stdout,
        parse_constant=lambda token: pytest.fail(f"non-strict constant: {token}"),
    )
    assert report["status"] == "fail"
    assert "manifest_mismatch" in _codes(report)
    assert process.stderr == ""


def test_metrics_jsonl_rejects_na_for_applicable_values(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "smoke", "rl-w")
    text = (run / "metrics.jsonl").read_text(encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        text.replace('"loss": 0.2', '"loss": NaN'), encoding="utf-8"
    )
    report = validate_run(run, profile="smoke")
    assert "invalid_jsonl" in _codes(report)
