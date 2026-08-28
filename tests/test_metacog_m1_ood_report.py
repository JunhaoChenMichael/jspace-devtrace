from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analysis.evaluate_metacog_m1_ood import gate_decision, summarize_condition  # noqa: E402
from analysis.report_metacog_m1 import build_report  # noqa: E402
from run_metacog_alignment_campaign import validate_report  # noqa: E402


def _rows(*, after: bool) -> list[dict[str, object]]:
    rows = []
    for episode in range(2):
        rows.extend(
            [
                {
                    "episode": episode,
                    "candidate_index": 0,
                    "concept": f"foil-{episode}",
                    "label": "distractor",
                    "V": 0.2 if after else 0.6,
                    "W_rr": 0.3,
                },
                {
                    "episode": episode,
                    "candidate_index": 1,
                    "concept": f"target-{episode}",
                    "label": "load_bearing",
                    "V": 0.8 if after else 0.4,
                    "W_rr": 0.7,
                },
            ]
        )
    return rows


def _full_context() -> dict[str, object]:
    before = [
        {"episode": 0, "answer": "a", "correct": True},
        {"episode": 1, "answer": "b", "correct": True},
    ]
    after = [
        {"episode": 0, "answer": "a", "correct": True},
        {"episode": 1, "answer": "b", "correct": True},
    ]
    return {
        "before_accuracy": 1.0,
        "after_accuracy": 1.0,
        "after_minus_before": 0.0,
        "drop_percentage_points": 0.0,
        "per_episode": {"before": before, "after": after},
    }


def test_ood_summary_uses_paired_episode_bootstrap_and_green_gate() -> None:
    summary = summarize_condition(
        _rows(after=False),
        _rows(after=True),
        _full_context(),
        bootstrap_samples=40,
        bootstrap_seed=0,
    )
    assert summary["verbal"]["before"]["pooled_auc"] == 0.0
    assert summary["verbal"]["after"]["pooled_auc"] == 1.0
    assert summary["verbal"]["delta_pooled_auc"] == 1.0
    assert (
        summary["verbal"]["paired_episode_bootstrap"]["after_minus_before"]
        ["bootstrap_samples_effective"]
        == 40
    )
    assert summary["workspace"]["delta_pooled_auc"] == 0.0
    decision, reasons, strong, controlled = gate_decision(summary)
    assert decision == "GREEN"
    assert reasons == ["all preregistered M1 GREEN conditions passed"]
    assert strong is True
    assert controlled is False


def test_report_populates_all_required_sections_with_real_audit_values(tmp_path: Path) -> None:
    condition = summarize_condition(
        _rows(after=False),
        _rows(after=True),
        _full_context(),
        bootstrap_samples=40,
        bootstrap_seed=0,
    )
    ood = {
        "model": "Qwen/Qwen3-8B",
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "checkpoint_path": "/run/m1/checkpoints/step-000137",
        "bootstrap": {"samples": 4000, "unit": "episode_cluster"},
        "conditions": {"decoupled": condition, "compositional": condition},
        "decision": "GREEN",
        "strong_green": True,
        "controlled_branch_authorized": False,
        "decision_reasons": ["all preregistered M1 GREEN conditions passed"],
    }
    candidate = {
        "step": 137,
        "verbal_auc": 0.8,
        "yes_rate": 0.4,
        "checkpoint_path": "checkpoints/step-000137",
    }
    lock = {
        "step": 137,
        "validation_auc": 0.8,
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_tree_sha256": "c" * 64,
        "candidate_checkpoints": [candidate],
    }
    canary = {
        "status": "PASS",
        "finite_loss_and_gradients": True,
        "checkpoint_save_load": {"readable": True},
        "adapter_enable_disable_check": {"passed": True},
        "workspace_post_training_evaluation": {"performed": True, "all_finite": True},
        "throughput": {"mean_tokens_per_second": 10.0},
        "gpu_memory": {"max_allocated_mib": 20000},
    }
    run_config = {
        "model": "Qwen/Qwen3-8B",
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "precision": "bfloat16",
        "lora_rank": 16,
        "gradient_accumulation": 4,
        "learning_rate": 1e-5,
        "epochs": 2,
        "target_optimizer_steps": 137,
        "gradient_checkpointing": True,
    }
    provenance = {
        "teacher": {"frozen_original": True, "student_workspace_used_for_labels": False},
        "data_isolation": {"ood_loaded": False, "ood_evaluated": False},
    }
    teacher_audit = {
        "method": "stable W_rr descending",
        "top_k": 2,
        "train": {"episodes": 10},
        "validation": {"episodes": 2},
        "train_target_counts": {"yes": 20, "no": 30},
        "validation_target_counts": {"yes": 4, "no": 6},
    }
    ledger = [
        {
            "event": "gpu_preflight_passed",
            "gpu": {
                "name": "NVIDIA RTX A5000",
                "total_mib": 24564,
                "free_mib": 23000,
            },
        },
        {
            "event": "command_finished",
            "artifact_hashes": {"m0/decoupled.json": "d" * 64},
        },
    ]
    report = build_report(
        m0_gate={
            "decision": "GREEN",
            "gate": {
                "reference": {"V": 0.337, "W_rr": 0.654},
                "observed": {"V": 0.34, "W_rr": 0.65},
                "absolute_delta": {"V": 0.003, "W_rr": 0.004},
                "tolerance": 0.05,
            },
        },
        canary=canary,
        m1_summary={
            "throughput": {"mean_tokens_per_second": 9.0},
            "gpu_memory": {"max_allocated_mib": 21000},
        },
        lock=lock,
        ood=ood,
        ledger=ledger,
        metadata={
            "model": "Qwen/Qwen3-8B",
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "chat_template_sha256": "e" * 64,
        },
        run_config=run_config,
        provenance=provenance,
        teacher_label_audit=teacher_audit,
        training_metrics=[{"loss": 1.0, "grad_norm": 0.5}, {"loss": 0.8, "grad_norm": 0.4}],
    )
    report_path = tmp_path / "report.md"
    report_path.write_text(report, encoding="utf-8")
    validate_report(report_path)
    assert "N/A" not in report
    assert "finite_loss_and_gradients" in report
    assert "student_workspace_used_for_labels" in report
