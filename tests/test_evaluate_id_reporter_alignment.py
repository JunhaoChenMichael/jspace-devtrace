from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.evaluate_id_reporter_alignment import (  # noqa: E402
    COMMON_CONDITION_LOCK,
    CONDITION_LOCKS,
    DEFAULT_MODEL_COMMIT,
    _active_adapters,
    _load_locked_run,
)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fake_run(tmp_path: Path, name: str) -> tuple[Path, dict]:
    run_dir = tmp_path / name
    adapter = run_dir / "best-step-200"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    manifest = {"manifest_sha256": "locked"}
    config = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "teacher_tag": "7B-Instruct",
        "validation_episode_count": 45,
        "resolved_model_commit": DEFAULT_MODEL_COMMIT,
        "probe_visible_to_policy": False,
        **COMMON_CONDITION_LOCK,
        **CONDITION_LOCKS[name],
    }
    _write(run_dir / "run_config.json", config)
    _write(run_dir / "split_manifest.json", manifest)
    _write(
        run_dir / "best_checkpoint.json",
        {"path": str(adapter), "step": 200, "metric": 0.75},
    )
    _write(
        run_dir / "summary.json",
        {"best_checkpoint": str(adapter), "best_validation_metric": 0.75},
    )
    _write(adapter / "training_state.json", {"step": 200, "metric": 0.75})
    return run_dir, manifest


def _load(name: str, run_dir: Path, manifest: dict):
    return _load_locked_run(
        name,
        run_dir,
        model="Qwen/Qwen2.5-7B-Instruct",
        teacher_tag="7B-Instruct",
        split_seed=0,
        workspace_top_k=2,
        expected_manifest=manifest,
        expected_model_commit=DEFAULT_MODEL_COMMIT,
    )


def test_locked_id_reporter_run_binds_label_protocol_and_checkpoint(tmp_path):
    run_dir, manifest = _fake_run(tmp_path, "hybrid-lw0.25")
    result = _load("hybrid-lw0.25", run_dir, manifest)

    assert result["mode"] == "rl-hybrid"
    assert result["checkpoint_step"] == 200
    assert result["training_protocol"]["lambda_w"] == 0.25
    assert result["legacy_config_probe_flag_missing"] is False
    assert len(result["adapter_weights_sha256"]) == 64

    with pytest.raises(ValueError, match="mode=.*expected"):
        _load("rl-qa", run_dir, manifest)


def test_locked_id_reporter_run_allows_only_legacy_sft_missing_probe_flag(tmp_path):
    run_dir, manifest = _fake_run(tmp_path, "sft-w")
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text())
    config.pop("probe_visible_to_policy")
    _write(config_path, config)

    result = _load("sft-w", run_dir, manifest)
    assert result["legacy_config_probe_flag_missing"] is True

    run_dir, manifest = _fake_run(tmp_path, "rl-qa")
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text())
    config.pop("probe_visible_to_policy")
    _write(config_path, config)
    with pytest.raises(ValueError, match="probe_visible"):
        _load("rl-qa", run_dir, manifest)


def test_locked_id_reporter_run_rejects_checkpoint_escape_or_state_mismatch(tmp_path):
    run_dir, manifest = _fake_run(tmp_path, "hybrid-lw1.0")
    outside = tmp_path / "outside" / "best-step-200"
    outside.mkdir(parents=True)
    _write(outside / "training_state.json", {"step": 200, "metric": 0.75})
    (outside / "adapter_model.safetensors").write_bytes(b"weights")
    _write(
        run_dir / "best_checkpoint.json",
        {"path": str(outside), "step": 200, "metric": 0.75},
    )
    with pytest.raises(ValueError, match="escapes its run directory"):
        _load("hybrid-lw1.0", run_dir, manifest)

    run_dir, manifest = _fake_run(tmp_path / "state", "hybrid-lw1.0")
    _write(
        run_dir / "best-step-200" / "training_state.json",
        {"step": 199, "metric": 0.75},
    )
    with pytest.raises(ValueError, match="training_state disagrees"):
        _load("hybrid-lw1.0", run_dir, manifest)


def test_base_model_peft_mixin_without_adapter_is_reported_as_inactive():
    class BaseModel:
        def active_adapters(self):
            raise ValueError("No adapter loaded. Please load an adapter first.")

    class Policy:
        model = BaseModel()

    assert _active_adapters(Policy()) == []
