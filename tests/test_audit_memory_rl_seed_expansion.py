from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from analysis.audit_memory_rl_seed_expansion import (  # noqa: E402
    EXPECTED_REPORTER_CONFIG,
    audit_seed_expansion,
    main,
    write_report_exclusive,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _adapter(path: Path, step: int, metric: float | None = None) -> None:
    path.mkdir(parents=True)
    state = {"step": step}
    if metric is not None:
        state["metric"] = metric
    _write_json(path / "training_state.json", state)
    _write_json(path / "adapter_config.json", {"peft_type": "LORA"})
    (path / "adapter_model.safetensors").write_bytes(b"weights")


def _base_config(seed0: Path) -> dict:
    return {
        "mode": "rl-qa",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "teacher_tag": "7B-Instruct",
        "workspace_teacher_model": "Qwen/Qwen2.5-7B-Instruct",
        "allow_teacher_mismatch": False,
        "teacher_mismatch_override": False,
        "teacher_matches_policy_reference": True,
        "fixed_workspace_teacher": True,
        "constrained_yes_no_rollouts": True,
        "probe_visible_to_policy": False,
        "frozen_recall_adapter_disabled": True,
        "workspace_reward_used_for_optimization": False,
        "seed": 0,
        "split_seed": 0,
        "out_dir": str(seed0.resolve()),
        "val_fraction": 0.2,
        "train_spec": [
            {"source": "explicit", "battery_path": "train-a", "results_path": "train-b"},
            {"source": "evoked", "battery_path": "train-c", "results_path": "train-d"},
            {"source": "evoked_g2", "battery_path": "train-e", "results_path": "train-f"},
        ],
        "train_episode_count": 175,
        "validation_episode_count": 45,
        "limit_train_episodes": 0,
        "limit_validation_episodes": 0,
        "val_eval_episodes": 0,
        "budget": 2,
        "workspace_top_k": 2,
        "workspace_objective": "rank-continuous",
        "workspace_set_reward": "mean",
        "max_steps": 300,
        "group_size": 8,
        "grpo_epochs": 2,
        "temperature": 5.0,
        "learning_rate": 1e-6,
        "beta": 0.03,
        "lambda_qa": 1.0,
        "lambda_w": 0.0,
        "advantage_normalization": "auto",
        "effective_advantage_normalization": "zscore",
        "clip_epsilon": 0.2,
        "batch_size": 4,
        "lora_rank": 32,
        "lora_alpha": 0,
        "lora_dropout": 0.0,
        "dtype": "bfloat16",
        "device": "cuda",
        "max_length": 2048,
        "answer_tokens": 64,
        "eval_every": 100,
        "diagnostic_every": 25,
        "save_every": 300,
        "early_stop_patience": 3,
        "dry_run": False,
        "resolved_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "resolved_model_commit": "fixture-commit",
        "trainable_parameter_count": 100,
        "total_parameter_count": 1000,
        "software_versions": {"python": "3.12", "torch": "2.4"},
    }


def _make_campaign(tmp_path: Path):
    seed0 = tmp_path / "formal_rl-qa_fixture_s0"
    seed0.mkdir()
    base = _base_config(seed0)
    manifest = {
        "schema_version": 1,
        "seed": 0,
        "manifest_sha256": "fixture-manifest",
        "train_episode_count": 175,
        "validation_episode_count": 45,
    }
    _write_json(seed0 / "run_config.json", base)
    _write_json(seed0 / "split_manifest.json", manifest)
    lock = {key: value for key, value in base.items() if key != "out_dir"}
    lock_path = tmp_path / "seed0-lock.json"
    _write_json(lock_path, lock)

    runs = {}
    for seed in (1, 2):
        run = tmp_path / f"formal_rl-qa_fixture_s{seed}"
        run.mkdir()
        config = dict(base)
        config.update(EXPECTED_REPORTER_CONFIG)
        config["seed"] = seed
        config["out_dir"] = str(run.resolve())
        _write_json(run / "run_config.json", config)
        # Copy exact bytes: the audit requires byte identity, not object equality.
        (run / "split_manifest.json").write_bytes(
            (seed0 / "split_manifest.json").read_bytes()
        )

        metrics = [
            {
                "event": "baseline",
                "step": 0,
                "qa_accuracy": 0.5,
                "selection_metric": 0.5,
                "containment": 0.5,
                "workspace_set_reward": 0.5,
                "episodes": 45,
            }
        ]
        validation_scores = {100: 0.6, 200: 0.7, 300: 0.7}
        for step in range(1, 301):
            mixed = step % 2 == 0
            metrics.append(
                {
                    "event": "train",
                    "step": step,
                    "loss": 0.1,
                    "reward_mean": 0.5,
                    "reward_std": 0.5 if mixed else 0.0,
                    "qa_reward": 0.5,
                    "workspace_reward": 0.5,
                    "kl": 0.01,
                    "grad_norm": 0.2,
                    "number_unique_selected_sets": 3 if mixed else 2,
                    "mixed_QA_reward_group": mixed,
                    "containment_rate": 0.5,
                    "mixed_containment_group": mixed,
                    "policy_set_entropy": 1.0,
                    "normalized_policy_set_entropy": 0.5,
                    "yes_probabilities": [0.2, 0.8],
                }
            )
            if step in validation_scores:
                score = validation_scores[step]
                metrics.append(
                    {
                        "event": "validation",
                        "step": step,
                        "qa_accuracy": score,
                        "selection_metric": score,
                        "containment": 0.6,
                        "workspace_set_reward": 0.55,
                        "episodes": 45,
                    }
                )
        _write_jsonl(run / "metrics.jsonl", metrics)
        _write_jsonl(
            run / "rollouts.jsonl",
            [
                {"step": step, "group_index": group, "reward": float(group % 2)}
                for step in range(1, 301)
                for group in range(8)
            ],
        )
        _write_jsonl(
            run / "reporter_correlations.jsonl",
            [
                {
                    "schema_version": 1,
                    "scope": "fixed_id_validation",
                    "event": event,
                    "step": step,
                    "summary": {},
                    "rows": [],
                }
                for event, step in (
                    ("baseline", 0),
                    ("validation", 100),
                    ("validation", 200),
                    ("validation", 300),
                )
            ],
        )
        dropout = {"postcondition_satisfied": True, "remaining_nonzero": []}
        _write_json(run / "dropout_audit.json", dropout)
        best_dir = run / "best-step-200"
        final_dir = run / "final_adapter"
        _adapter(best_dir, 200, 0.7)
        _adapter(final_dir, 300)
        best = {"path": str(best_dir.resolve()), "step": 200, "metric": 0.7}
        _write_json(run / "best_checkpoint.json", best)
        _write_json(
            run / "summary.json",
            {
                "mode": "rl-qa",
                "steps_completed": 300,
                "best_validation_metric": 0.7,
                "best_checkpoint": str(best_dir.resolve()),
                "final_adapter": str(final_dir.resolve()),
                "reporter_correlation_artifact": str(
                    (run / "reporter_correlations.jsonl").resolve()
                ),
            },
        )
        runs[seed] = run
    return seed0, lock_path, runs


def _passing_validator(*args, **kwargs):
    return {"status": "pass", "errors": [], "warnings": [], "details": {}}


def _codes(result, seed: int) -> set[str]:
    return {error["code"] for error in result["runs"][str(seed)]["errors"]}


def test_seed_expansion_audit_passes_and_records_required_diagnostics(tmp_path: Path) -> None:
    seed0, lock, runs = _make_campaign(tmp_path)
    report = audit_seed_expansion(
        seed0_run=seed0,
        lock_path=lock,
        runs=runs,
        validator=_passing_validator,
    )
    assert report["status"] == "pass", report
    for seed in (1, 2):
        audited = report["runs"][str(seed)]
        assert audited["counts"]["train_records"] == 300
        assert audited["counts"]["rollout_records"] == 2400
        assert audited["counts"]["validation_steps"] == [100, 200, 300]
        assert audited["checkpoint_selection"]["expected_best_step"] == 200
        assert audited["checkpoint_selection"]["passed"] is True
        assert audited["training_diagnostics"]["mixed_reward"]["groups"] == 150
        assert audited["training_diagnostics"]["gradients"]["nonzero_groups"] == 300
        assert audited["reporter_diagnostics"]["used_for_selection"] is False
        assert all(field["passed"] for field in audited["lock_fields"].values())


def test_audit_rejects_later_tied_checkpoint_and_manifest_byte_change(tmp_path: Path) -> None:
    seed0, lock, runs = _make_campaign(tmp_path)
    best_path = runs[1] / "best_checkpoint.json"
    best = json.loads(best_path.read_text())
    best.update({"path": str((runs[1] / "best-step-300").resolve()), "step": 300})
    _adapter(runs[1] / "best-step-300", 300, 0.7)
    _write_json(best_path, best)
    (runs[2] / "split_manifest.json").write_text(
        (seed0 / "split_manifest.json").read_text() + "\n",
        encoding="utf-8",
    )

    report = audit_seed_expansion(
        seed0_run=seed0,
        lock_path=lock,
        runs=runs,
        validator=_passing_validator,
    )
    assert report["status"] == "fail"
    assert "first_maximum_checkpoint_mismatch" in _codes(report, 1)
    assert "manifest_not_byte_identical" in _codes(report, 2)
    assert report["runs"]["2"]["formal_validator"]["status"] == "not_run"


def test_audit_fails_closed_on_lock_and_reporter_selection_mismatch(tmp_path: Path) -> None:
    seed0, lock, runs = _make_campaign(tmp_path)
    config_path = runs[1] / "run_config.json"
    config = json.loads(config_path.read_text())
    config["beta"] = 0.1
    _write_json(config_path, config)
    config_path = runs[2] / "run_config.json"
    config = json.loads(config_path.read_text())
    config["reporter_correlations_used_for_selection"] = True
    _write_json(config_path, config)

    calls = []

    def validator(*args, **kwargs):
        calls.append((args, kwargs))
        return _passing_validator()

    report = audit_seed_expansion(
        seed0_run=seed0,
        lock_path=lock,
        runs=runs,
        validator=validator,
    )
    assert report["status"] == "fail"
    assert "lock_field_mismatch" in _codes(report, 1)
    assert "reporter_config_mismatch" in _codes(report, 2)
    assert calls == []  # unsafe parity prevents following any declared artifact paths


def test_audit_propagates_formal_validator_failure(tmp_path: Path) -> None:
    seed0, lock, runs = _make_campaign(tmp_path)

    def validator(*args, **kwargs):
        return {
            "status": "fail",
            "errors": [{"code": "fixture_failure", "message": "bad"}],
            "warnings": [],
            "details": {},
        }

    report = audit_seed_expansion(
        seed0_run=seed0,
        lock_path=lock,
        runs=runs,
        validator=validator,
    )
    assert report["status"] == "fail"
    assert "formal_validator_failed" in _codes(report, 1)
    assert "formal_validator_failed" in _codes(report, 2)


def test_audit_fails_closed_on_missing_artifact_and_nonfinite_jsonl(tmp_path: Path) -> None:
    seed0, lock, runs = _make_campaign(tmp_path)
    (runs[1] / "reporter_correlations.jsonl").unlink()
    metrics_path = runs[2] / "metrics.jsonl"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8").replace(
            '"loss": 0.1', '"loss": NaN', 1
        ),
        encoding="utf-8",
    )
    report = audit_seed_expansion(
        seed0_run=seed0,
        lock_path=lock,
        runs=runs,
        validator=_passing_validator,
    )
    assert report["status"] == "fail"
    assert "required_artifacts_missing" in _codes(report, 1)
    assert "invalid_jsonl" in _codes(report, 2)


def test_exclusive_output_refuses_overwrite_and_cli_checks_before_inputs(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "audit.json"
    report = {"status": "pass", "finite": 1.0}
    write_report_exclusive(output, report)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_report_exclusive(output, {"status": "fail"})
    assert output.read_bytes() == original

    result = main(
        [
            "--seed0-run",
            str(tmp_path / "currently-training-do-not-read"),
            "--lock",
            str(tmp_path / "missing-lock"),
            "--run",
            f"1={tmp_path / 'missing-one'}",
            "--run",
            f"2={tmp_path / 'missing-two'}",
            "--out",
            str(output),
        ]
    )
    assert result == 2
    assert output.read_bytes() == original
    assert "refusing to overwrite" in capsys.readouterr().err


def test_cli_writes_new_fail_report_for_missing_inputs(tmp_path: Path) -> None:
    output = tmp_path / "new-audit.json"
    result = main(
        [
            "--seed0-run",
            str(tmp_path / "missing-seed0"),
            "--lock",
            str(tmp_path / "missing-lock"),
            "--run",
            f"1={tmp_path / 'missing-one'}",
            "--run",
            f"2={tmp_path / 'missing-two'}",
            "--out",
            str(output),
        ]
    )
    assert result == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "fail"
    assert {error["code"] for error in saved["errors"]} == {
        "seed0_run_missing",
        "lock_missing",
    }
