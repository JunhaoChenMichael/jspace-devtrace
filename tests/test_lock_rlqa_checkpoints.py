from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "lock_rlqa_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("lock_rlqa_checkpoints", MODULE_PATH)
assert SPEC and SPEC.loader
locker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = locker
SPEC.loader.exec_module(locker)


def _make_run(root: Path, seed: int, *, step: int = 300, **overrides) -> Path:
    run_dir = root / f"seed{seed}"
    checkpoint = run_dir / f"best-step-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(f"adapter-{seed}".encode())
    config = {
        "model": locker.EXPECTED_MODEL,
        "seed": seed,
        "mode": "rl-qa",
        "lambda_qa": 1.0,
        "lambda_w": 0.0,
        "budget": 2,
        "group_size": 8,
        "grpo_epochs": 2,
        "lora_rank": 32,
        "max_steps": 300,
        "learning_rate": 1e-6,
        "beta": 0.03,
        "split_seed": 0,
        "temperature": 5.0,
        "resolved_model_commit": locker.EXPECTED_REVISION,
    }
    config.update(overrides)
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "best_checkpoint.json").write_text(
        json.dumps({"path": str(checkpoint), "step": step, "metric": 0.61}), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"best_checkpoint": str(checkpoint)}), encoding="utf-8"
    )
    (run_dir / "split_manifest.json").write_text(
        json.dumps({"manifest_sha256": "shared-split"}), encoding="utf-8"
    )
    return run_dir


def test_lock_binds_three_seeds_and_self_hashes(tmp_path: Path) -> None:
    runs = {seed: _make_run(tmp_path, seed) for seed in (0, 1, 2)}
    payload = locker.build_lock(runs)
    assert payload["schema_version"] == locker.LOCK_SCHEMA
    assert [entry["seed"] for entry in payload["seeds"]] == [0, 1, 2]
    assert payload["ood_authorization"]["attempt_limit"] == 1
    unsigned = {k: v for k, v in payload.items() if k != "manifest_sha256"}
    assert locker.canonical_json_sha256(unsigned) == payload["manifest_sha256"]
    assert len({entry["checkpoint_tree_sha256"] for entry in payload["seeds"]}) == 3


def test_lock_refuses_partial_seeds_and_recipe_drift(tmp_path: Path) -> None:
    runs = {seed: _make_run(tmp_path, seed) for seed in (0, 1)}
    with pytest.raises(locker.LockError, match=r"seeds \[0, 1, 2\] must be locked together"):
        locker.build_lock(runs)

    drifted = tmp_path / "drift"
    runs = {
        0: _make_run(drifted, 0, lambda_w=0.25),
        1: _make_run(drifted, 1),
        2: _make_run(drifted, 2),
    }
    with pytest.raises(locker.LockError, match="changed lambda_w"):
        locker.build_lock(runs)


def test_lock_refuses_a_drifted_temperature_or_model_commit(tmp_path: Path) -> None:
    runs = {0: _make_run(tmp_path, 0, temperature=0.7), 1: _make_run(tmp_path, 1), 2: _make_run(tmp_path, 2)}
    with pytest.raises(locker.LockError, match="changed temperature"):
        locker.build_lock(runs)

    other = tmp_path / "commit"
    runs = {
        0: _make_run(other, 0, resolved_model_commit="c" * 40),
        1: _make_run(other, 1),
        2: _make_run(other, 2),
    }
    with pytest.raises(locker.LockError, match="different model commit"):
        locker.build_lock(runs)


def test_lock_refuses_a_run_that_touched_sealed_conditions(tmp_path: Path) -> None:
    runs = {
        0: _make_run(tmp_path, 0, eval_spec="decoupled=data/benchmarks/battery_v4_final.json"),
        1: _make_run(tmp_path, 1),
        2: _make_run(tmp_path, 2),
    }
    with pytest.raises(locker.LockError, match="sealed data"):
        locker.build_lock(runs)


def test_single_seed_lock_is_allowed_when_the_plan_authorises_one_seed(tmp_path: Path) -> None:
    """The 32B scaling gate authorises seed 0 alone, and nothing beyond it."""

    runs = {0: _make_run(tmp_path, 0)}
    payload = locker.build_lock(runs, [0])
    assert [entry["seed"] for entry in payload["seeds"]] == [0]
    assert payload["ood_authorization"]["attempt_limit"] == 1

    # a seed that was not authorised is still refused
    extra = {0: _make_run(tmp_path / "x", 0), 1: _make_run(tmp_path / "x", 1)}
    with pytest.raises(locker.LockError, match="must be locked together"):
        locker.build_lock(extra, [0])


def test_lock_refuses_a_model_with_no_pinned_revision(tmp_path: Path, monkeypatch) -> None:
    """An unpinned model must not silently skip the commit check.

    Before this guard, a model absent from PINNED_REVISIONS left EXPECTED_REVISION
    as None, and the commit comparison was skipped entirely -- so any resolved
    commit would have been accepted.
    """
    monkeypatch.setattr(locker, "EXPECTED_REVISION", None)
    runs = {0: _make_run(tmp_path, 0), 1: _make_run(tmp_path, 1), 2: _make_run(tmp_path, 2)}
    with pytest.raises(locker.LockError, match="no pinned revision"):
        locker.build_lock(runs)
