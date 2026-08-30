from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_metacog_alignment_campaign.py"
SPEC = importlib.util.spec_from_file_location("metacog_campaign_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _make_trainer_lock(m1_dir: Path, selected_step: int = 137) -> Path:
    candidates = []
    for step, score in ((0, 0.55), (100, 0.68), (selected_step, 0.72)):
        checkpoint = m1_dir / "checkpoints" / f"step-{step:06d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(f"adapter-{step}".encode())
        tree_hash, _ = campaign.checkpoint_tree_hash(checkpoint)
        candidates.append(
            {
                "step": step,
                "selection_scope": "id_validation",
                "selection_metric": "verbal_auc",
                "verbal_auc": score,
                "verbal_within_episode_auc": score - 0.01,
                "yes_rate": 0.4,
                "checkpoint_path": f"checkpoints/step-{step:06d}",
                "checkpoint_tree_sha256": tree_hash,
            }
        )
    run_config = {
        "model": campaign.EXPECTED_MODEL,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "seed": 0,
    }
    _write_json(m1_dir / "run_config.json", run_config)
    _write_json(m1_dir / "split_manifest.json", {"split": "id"})
    _write_json(m1_dir / "provenance.json", {"teacher": "frozen"})
    (m1_dir / "validation_metrics.jsonl").write_text('{"step":0}\n', encoding="utf-8")
    selected = candidates[-1]
    lock = {
        "campaign_stage": "M1",
        "status": "LOCKED",
        "selection_scope": "id_validation",
        "selection_metric": "verbal_auc",
        "tie_break": "earliest_step",
        "ood_evaluated": False,
        "eligible_for_ood": True,
        "checkpoint_path": selected["checkpoint_path"],
        "step": selected["step"],
        "validation_auc": selected["verbal_auc"],
        "checkpoint_tree_sha256": selected["checkpoint_tree_sha256"],
        "split_manifest_sha256": campaign.sha256_file(m1_dir / "split_manifest.json"),
        "run_config_sha256": campaign.sha256_file(m1_dir / "run_config.json"),
        "provenance_sha256": campaign.sha256_file(m1_dir / "provenance.json"),
        "validation_metrics_sha256": campaign.sha256_file(
            m1_dir / "validation_metrics.jsonl"
        ),
        "candidate_steps": [row["step"] for row in candidates],
        "id_selection_table": candidates,
        "candidate_checkpoints": candidates,
    }
    lock["manifest_sha256"] = campaign.canonical_json_sha256(lock)
    path = m1_dir / "lock_manifest.json"
    _write_json(path, lock)
    return path


def _make_m0_artifacts(run_dir: Path) -> None:
    gate_conditions = {}
    for condition in ("explicit", "evoked", "decoupled", "compositional", "evoked_g2"):
        raw_path = run_dir / "m0" / f"{condition}.json"
        battery_path = run_dir / "source_batteries" / f"{condition}.json"
        _write_json(raw_path, [{"condition": condition}])
        _write_json(battery_path, [{"condition": condition}])
        raw_hash = campaign.sha256_file(raw_path)
        metadata_path = Path(f"{raw_path}.metadata")
        metadata = {
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "battery": str(battery_path),
            "hashes": {
                "raw_output_sha256": raw_hash,
                "battery_file_sha256": campaign.sha256_file(battery_path),
            },
        }
        _write_json(metadata_path, metadata)
        if condition != "evoked_g2":
            gate_conditions[condition] = {
                "source": {
                    "path": str(raw_path),
                    "sha256": raw_hash,
                    "metadata": {
                        "path": str(metadata_path),
                        "sha256": campaign.sha256_file(metadata_path),
                    },
                }
            }
    _write_json(
        run_dir / "m0" / "gate.json",
        {"decision": "GREEN", "conditions": gate_conditions},
    )


def test_default_plan_is_fixed_to_m0_m1_and_date_stamped(tmp_path: Path) -> None:
    context = {
        "python": sys.executable,
        "repo": str(REPO_ROOT),
        "run_dir": str(tmp_path / "run"),
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "ood_attempt_id": "c" * 32,
    }
    plan = campaign._render(campaign.default_plan(), context)
    campaign.validate_plan(plan, tmp_path / "run")
    commands = [spec for _, spec in campaign._all_command_specs(plan)]
    rendered = " ".join(token for spec in commands for token in spec["argv"]).lower()
    assert "h100" not in rendered
    assert "train_memory_rl" not in rendered
    assert campaign.EXPECTED_MODEL.lower() in rendered
    assert len(plan["stages"]["m0"]) == 4
    assert plan["stages"]["teacher_prep"]["name"] == "m1_prepare_frozen_teacher_evoked_g2"
    assert "evoked_g2" not in " ".join(
        spec["name"] for spec in plan["stages"]["m0"]
    )
    dated = campaign._default_run_dir(
        tmp_path, datetime(2026, 8, 27, tzinfo=timezone.utc)
    )
    assert dated.name == f"2026-08-27_{campaign.MODEL_LABEL}_{campaign.GPU_LABEL}_seed0"
    for seed in campaign.ALLOWED_SEEDS:
        seeded = campaign._render(campaign.default_plan(seed), context)
        campaign.validate_plan(seeded, tmp_path / "run")
        assert seeded["stages"]["m1"]["name"] == f"m1_seed{seed}_formal_pilot"
        assert ["--seed", str(seed)] == [
            token
            for index, token in enumerate(seeded["stages"]["m1"]["argv"])
            if token == "--seed" or seeded["stages"]["m1"]["argv"][index - 1] == "--seed"
        ]
    with pytest.raises(campaign.CampaignError, match="campaign seed"):
        campaign.default_plan(3)


def test_gpu_preflight_parsers_require_one_free_unoccupied_campaign_gpu() -> None:
    gpu_name = campaign.EXPECTED_GPU
    inventory = campaign.parse_gpu_inventory(
        f"0, GPU-A, {gpu_name}, 24564, 23100\n"
        f"1, GPU-B, {gpu_name}, 24564, 21500\n"
    )
    gpu, processes, mode = campaign.select_campaign_gpu(
        inventory,
        [{"gpu_uuid": "GPU-B", "pid": "42", "process_name": "python", "used_memory_mib": "1024"}],
        min_free_mib=22_000,
        requested_index="0",
    )
    assert gpu == {
        "index": "0",
        "uuid": "GPU-A",
        "name": gpu_name,
        "total_mib": 24564,
        "free_mib": 23100,
    }
    assert processes == []
    assert mode == "explicit_index"
    assert campaign.parse_compute_processes("") == []
    resident = [
        {"gpu_uuid": "GPU-A", "pid": "7", "process_name": "daemon", "used_memory_mib": "227"}
    ]
    with pytest.raises(campaign.CampaignError, match="no idle"):
        campaign.select_campaign_gpu(
            inventory,
            resident,
            min_free_mib=22_000,
            requested_index="0",
        )
    selected, observed, _ = campaign.select_campaign_gpu(
        inventory,
        resident,
        min_free_mib=22_000,
        requested_index="0",
        allowed_existing_process_mib=512,
    )
    assert selected["index"] == "0"
    assert observed == resident
    wrong_device = campaign.parse_gpu_inventory(
        "0, GPU-X, NVIDIA GeForce RTX 4090, 24564, 24000\n"
    )
    with pytest.raises(campaign.CampaignError, match="no idle"):
        campaign.select_campaign_gpu(
            wrong_device, [], min_free_mib=22_000, requested_index="0"
        )
    low_memory = campaign.parse_gpu_inventory(
        f"0, GPU-A, {gpu_name}, 24564, 21000\n"
    )
    with pytest.raises(campaign.CampaignError, match="no idle"):
        campaign.select_campaign_gpu(
            low_memory, [], min_free_mib=22_000, requested_index="0"
        )
    assert campaign.parse_compute_processes("GPU-1, 42, python, 1024\n")[0]["pid"] == "42"


def test_checkpoint_tree_hash_rejects_symlinks(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter")
    digest, files = campaign.checkpoint_tree_hash(checkpoint)
    assert len(digest) == 64
    assert files == {"adapter.bin": campaign.sha256_file(checkpoint / "adapter.bin")}
    (checkpoint / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(campaign.CampaignError, match="symlink"):
        campaign.checkpoint_tree_hash(checkpoint)


def test_trainer_lock_accepts_actual_terminal_step_and_recomputes_id_selection(
    tmp_path: Path,
) -> None:
    m1_dir = tmp_path / "m1"
    source = _make_trainer_lock(m1_dir, selected_step=137)
    payload, checkpoint, _, _ = campaign.validate_trainer_lock(
        source,
        m1_dir,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    assert payload["step"] == 137
    assert checkpoint.name == "step-000137"

    payload["step"] = 100
    payload["checkpoint_path"] = "checkpoints/step-000100"
    payload["checkpoint_tree_sha256"] = payload["candidate_checkpoints"][1][
        "checkpoint_tree_sha256"
    ]
    _write_json(source, payload)
    with pytest.raises(campaign.CampaignError, match="highest ID verbal AUC"):
        campaign.validate_trainer_lock(source, m1_dir)


def test_canary_failure_is_a_controlled_stop(tmp_path: Path) -> None:
    manifest = tmp_path / "canary.json"
    _write_json(
        manifest,
        {"status": "FAIL", "canary_passed": False, "eligible_for_ood": False},
    )
    with pytest.raises(campaign.ControlledStop, match="canary did not pass"):
        campaign.validate_canary_manifest(manifest)


def test_ood_attempt_is_consumed_before_command_and_cannot_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    _make_m0_artifacts(run_dir)
    source = _make_trainer_lock(run_dir / "m1", selected_step=137)
    runner = campaign.CampaignRunner(
        repo_root=REPO_ROOT,
        run_dir=run_dir,
        plan={},
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        nvidia_smi="nvidia-smi",
        min_free_mib=22_000,
        requested_gpu_index="0",
        allowed_existing_process_mib=0,
        attempt_id="d" * 32,
    )
    lock_path = runner.create_id_lock(source)
    lock_hash = campaign.sha256_file(lock_path)
    result_path = run_dir / "ood" / "result.json"
    payload = {
        "attempt_id": "d" * 32,
        "lock_manifest_sha256": lock_hash,
        "bootstrap": {"samples": 4000, "unit": "episode_cluster"},
        "conditions": {"decoupled": {}, "compositional": {}},
        "decision": "GREEN",
    }
    code = (
        "from pathlib import Path; import json; "
        f"Path({str(result_path)!r}).write_text(json.dumps({payload!r}))"
    )
    spec = {
        "name": "synthetic_ood",
        "argv": [sys.executable, "-c", code],
        "outputs": [str(result_path)],
        "result_path": str(result_path),
    }
    monkeypatch.setattr(
        campaign,
        "validate_ood_result",
        lambda *_args, **_kwargs: {"decision": "GREEN"},
    )
    assert runner.one_shot_ood(spec, lock_path)["decision"] == "GREEN"
    assert (run_dir / "ood" / "attempt_started.json").is_file()
    assert (run_dir / "ood" / "attempt_result.json").is_file()
    with pytest.raises(campaign.CampaignError, match="refusing to overwrite"):
        runner.one_shot_ood(spec, lock_path)
    finished = [
        json.loads(line)
        for line in (run_dir / "decision_ledger.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "command_finished"
    ]
    assert len(finished) == 1
    assert finished[0]["exit_code"] == 0
    assert finished[0]["artifact_hashes"]["ood/result.json"] == campaign.sha256_file(
        result_path
    )


def _runner_for(run_dir: Path, *, seed: int = 0, resume: bool = False):
    return campaign.CampaignRunner(
        repo_root=REPO_ROOT,
        run_dir=run_dir,
        plan={},
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        nvidia_smi="nvidia-smi",
        min_free_mib=22_000,
        requested_gpu_index="0",
        allowed_existing_process_mib=0,
        attempt_id="e" * 32,
        seed=seed,
        resume=resume,
    )


def test_id_lock_phase_pauses_with_ood_still_sealed(tmp_path: Path) -> None:
    """The multi-seed contract locks every seed BEFORE any seed sees OOD."""

    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    _make_m0_artifacts(run_dir)
    source = _make_trainer_lock(run_dir / "m1", selected_step=137)
    runner = _runner_for(run_dir)
    lock_path = runner.create_id_lock(source)

    assert runner._pause_at_id_lock(lock_path) == "id_lock_complete_ood_sealed"
    marker = json.loads((run_dir / "ID_LOCK_STOP.json").read_text())
    assert marker["ood_opened"] is False
    assert marker["lock_manifest_sha256"] == campaign.sha256_file(lock_path)
    assert not (run_dir / "ood").exists()
    events = [row["event"] for row in campaign.read_ledger_events(run_dir / "decision_ledger.jsonl")]
    assert "campaign_paused_at_id_lock" in events

    # A resumed phase appends to the same ledger and keeps command-log names unique.
    (run_dir / "command_logs").mkdir(exist_ok=True)
    (run_dir / "command_logs" / "01_existing.log").write_text("x", encoding="utf-8")
    resumed = _runner_for(run_dir, resume=True)
    assert resumed.command_counter == 1
    assert resumed.ledger.path == runner.ledger.path


def test_resume_refuses_a_second_ood_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    _make_m0_artifacts(run_dir)
    source = _make_trainer_lock(run_dir / "m1", selected_step=137)
    lock_path = _runner_for(run_dir).create_id_lock(source)

    # No pause marker yet: the ID phase never stopped cleanly at its lock.
    with pytest.raises(campaign.CampaignError, match="did not stop cleanly"):
        _runner_for(run_dir, resume=True).resume_ood()

    _write_json(run_dir / "ID_LOCK_STOP.json", {"ood_opened": False})
    (run_dir / "ood").mkdir()
    with pytest.raises(campaign.CampaignError, match="already consumed its OOD attempt"):
        _runner_for(run_dir, resume=True).resume_ood()

    (run_dir / "ood").rmdir()
    with (run_dir / "decision_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "ood_attempt_started"}) + "\n")
    with pytest.raises(campaign.CampaignError, match="already records an OOD attempt"):
        _runner_for(run_dir, resume=True).resume_ood()
    assert lock_path.is_file()


def test_seeds_beyond_the_preregistered_set_are_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    with pytest.raises(campaign.CampaignError, match="campaign seed must be one of"):
        _runner_for(run_dir, seed=7)
