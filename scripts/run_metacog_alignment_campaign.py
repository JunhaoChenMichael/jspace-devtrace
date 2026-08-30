#!/usr/bin/env python3
"""Fail-closed A5000 orchestrator for the Metacognitive Alignment M0/M1 campaign.

This launcher deliberately has no H100 stage.  It executes exactly:

    M0 -> M0 gate -> canary -> M1 -> ID lock -> one-shot OOD -> report -> STOP

The scientific commands are described by a small JSON-compatible plan.  The
default plan targets the repository CLIs documented in
``docs/METACOG_ALIGNMENT_CAMPAIGN.md``.  A custom plan is useful for testing or
while those CLIs evolve, but it cannot add stages or bypass the safety gates.
Commands are executed directly (never through a shell).
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CAMPAIGN_SCHEMA = "metacog-alignment-campaign/v1"
PLAN_SCHEMA = "metacog-alignment-command-plan/v1"
LOCK_SCHEMA = "metacog-alignment-id-lock/v1"
# The campaign target is configurable but never open: exactly one model per
# campaign, drawn from a closed allowlist.  Qwen3-8B stays the default so the
# completed 8B campaigns reproduce unchanged; Qwen3-32B is the scale point
# authorised by the 32B seed-0 plans.  MoE variants are deliberately absent —
# substituting Qwen3-30B-A3B would confound scale with architecture.
SUPPORTED_MODELS = {
    "Qwen/Qwen3-8B": "qwen3-8b",
    "Qwen/Qwen3-32B": "qwen3-32b",
}
EXPECTED_MODEL = os.environ.get("METACOG_EXPECTED_MODEL", "Qwen/Qwen3-8B")
if EXPECTED_MODEL not in SUPPORTED_MODELS:
    raise SystemExit(
        f"METACOG_EXPECTED_MODEL={EXPECTED_MODEL!r} is not an approved campaign model; "
        f"choose one of {sorted(SUPPORTED_MODELS)}"
    )
MODEL_LABEL = SUPPORTED_MODELS[EXPECTED_MODEL]
# Campaign hardware is configurable but never permissive: the launcher still
# matches ONE exact device name, drawn from a closed allowlist.  A5000 remains
# the default so the completed seed-0 campaign reproduces byte-for-byte; the
# A100 entries exist for the multi-seed replication that
# docs/H100_NEXT_CAMPAIGNS.md authorises on capable-scale hardware.
SUPPORTED_GPUS = {
    "NVIDIA RTX A5000": "a5000",
    "NVIDIA A100-SXM4-80GB": "a100",
    "NVIDIA A100-SXM4-40GB": "a100",
    "NVIDIA H100 80GB HBM3": "h100",
    "NVIDIA H100 PCIe": "h100",
}
EXPECTED_GPU = os.environ.get("METACOG_EXPECTED_GPU", "NVIDIA RTX A5000")
if EXPECTED_GPU not in SUPPORTED_GPUS:
    raise SystemExit(
        f"METACOG_EXPECTED_GPU={EXPECTED_GPU!r} is not an approved campaign device; "
        f"choose one of {sorted(SUPPORTED_GPUS)}"
    )
GPU_LABEL = SUPPORTED_GPUS[EXPECTED_GPU]
ALLOWED_SEEDS = (0, 1, 2)
DEFAULT_MIN_FREE_MIB = 22_000
DEFAULT_BOOTSTRAP_DRAWS = 4_000
PINNED_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PIPELINE_STAGES = (
    "m0",
    "m0_gate",
    "teacher_prep",
    "canary",
    "m1",
    "id_lock",
    "ood",
    "report",
    "stop",
)


class CampaignError(RuntimeError):
    """A fail-closed campaign validation or execution error."""


class ControlledStop(CampaignError):
    """An expected scientific gate stop, not an orchestration crash."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_tree_hash(root: Path) -> tuple[str, dict[str, str]]:
    """Return the canonical checkpoint tree hash and its per-file hashes.

    The tree hash is SHA-256 over sorted records of the form
    ``relative/path\0FILE_SHA256\n``.  Symlinks are rejected so a locked tree
    cannot later redirect outside the campaign directory.
    """

    if root.is_symlink() or not root.is_dir():
        raise CampaignError(f"checkpoint is not a real directory: {root}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise CampaignError(f"checkpoint tree contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files[relative] = sha256_file(path)
        elif not path.is_dir():
            raise CampaignError(f"checkpoint tree contains a non-regular entry: {path}")
    if not files:
        raise CampaignError(f"checkpoint tree is empty: {root}")
    digest = hashlib.sha256()
    for relative, file_hash in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), files


def artifact_hashes(paths: Iterable[Path], run_dir: Path) -> dict[str, str]:
    """Hash declared artifacts, including every regular file in directories."""

    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_symlink():
            raise CampaignError(f"artifact must not be a symlink: {path}")
        if path.is_file():
            hashes[path.relative_to(run_dir).as_posix()] = sha256_file(path)
        elif path.is_dir():
            tree_hash, files = checkpoint_tree_hash(path)
            key = path.relative_to(run_dir).as_posix().rstrip("/") + "/"
            hashes[key] = tree_hash
            for relative, file_hash in files.items():
                hashes[f"{key}{relative}"] = file_hash
        else:
            raise CampaignError(f"declared artifact does not exist: {path}")
    return dict(sorted(hashes.items()))


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise CampaignError(f"refusing to overwrite existing artifact: {path}") from exc


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"missing or unsafe {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"{label} must be a JSON object: {path}")
    return payload


def require_within(path: Path, root: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise CampaignError(f"{label} escapes run directory: {path}") from exc
    return path


class DecisionLedger:
    """Append-only JSONL record of commands, decisions, statuses, and hashes."""

    def __init__(self, path: Path, *, append_existing: bool = False) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if append_existing:
            if path.is_symlink() or not path.is_file():
                raise CampaignError(f"cannot resume: missing decision ledger: {path}")
            return
        try:
            with path.open("x", encoding="utf-8"):
                pass
        except FileExistsError as exc:
            raise CampaignError(f"refusing to overwrite decision ledger: {path}") from exc

    def append(self, event: str, **fields: Any) -> None:
        row = {"timestamp_utc": utc_now(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int
    log_path: Path
    log_sha256: str
    output: str | None
    artifact_hashes: dict[str, str]


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "command"


def _command_spec(name: str, argv: Sequence[str], outputs: Sequence[str]) -> dict[str, Any]:
    return {"name": name, "argv": list(argv), "outputs": list(outputs)}


def default_plan(seed: int = 0) -> dict[str, Any]:
    """Return the fixed M0/M1 plan for one optimisation seed."""

    if seed not in ALLOWED_SEEDS:
        raise CampaignError(f"campaign seed must be one of {ALLOWED_SEEDS}")

    measure_common = [
        "{python}",
        "{repo}/src/experiments/measure.py",
        "--model",
        EXPECTED_MODEL,
        "--model-revision",
        "{model_revision}",
        "--tokenizer-revision",
        "{tokenizer_revision}",
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--end-only",
        "--no-verbal-raw",
    ]
    m0_batteries = {
        "explicit": "data/benchmarks/battery_v1_final.json",
        "evoked": "data/benchmarks/battery_v2_final.json",
        "decoupled": "data/benchmarks/battery_v4_final.json",
        "compositional": "data/benchmarks/battery_v3d.json",
    }
    m0_commands = []
    for condition, battery in m0_batteries.items():
        output = f"{{run_dir}}/m0/{condition}.json"
        m0_commands.append(
            _command_spec(
                f"m0_measure_{condition}",
                [
                    *measure_common,
                    "--battery",
                    f"{{repo}}/{battery}",
                    "--out",
                    output,
                ],
                [output, f"{output}.metadata"],
            )
        )
    teacher_output = "{run_dir}/m0/evoked_g2.json"
    teacher_prep = _command_spec(
        "m1_prepare_frozen_teacher_evoked_g2",
        [
            *measure_common,
            "--battery",
            "{repo}/data/benchmarks/battery_v2_g2.json",
            "--out",
            teacher_output,
        ],
        [teacher_output, f"{teacher_output}.metadata"],
    )

    train_common = [
        "{python}",
        "{repo}/src/experiments/train_metacog_m1.py",
        "--model",
        EXPECTED_MODEL,
        "--model-revision",
        "{model_revision}",
        "--tokenizer-revision",
        "{tokenizer_revision}",
        "--seed",
        str(seed),
        "--train-spec",
        "explicit={run_dir}/m0/explicit.json::{repo}/data/benchmarks/battery_v1_final.json",
        "--train-spec",
        "evoked={run_dir}/m0/evoked.json::{repo}/data/benchmarks/battery_v2_final.json",
        "--train-spec",
        "evoked_g2={run_dir}/m0/evoked_g2.json::{repo}/data/benchmarks/battery_v2_g2.json",
    ]
    return {
        "schema_version": PLAN_SCHEMA,
        "stages": {
            "m0": m0_commands,
            "m0_gate": {
                **_command_spec(
                    "m0_reproduction_gate",
                    [
                        "{python}",
                        "{repo}/src/analysis/gate_metacog_m0.py",
                        "--explicit",
                        "{run_dir}/m0/explicit.json",
                        "--evoked",
                        "{run_dir}/m0/evoked.json",
                        "--decoupled",
                        "{run_dir}/m0/decoupled.json",
                        "--compositional",
                        "{run_dir}/m0/compositional.json",
                        "--paper-v",
                        "0.337",
                        "--paper-w-rr",
                        "0.654",
                        "--tolerance",
                        "0.05",
                        # Qwen3-8B reproduces the published AUCs; a new scale
                        # point has no reason to, and is gated instead on the
                        # existence of a repairable reporting gap.
                        *(
                            ("--mode", "paper_reproduction")
                            if EXPECTED_MODEL == "Qwen/Qwen3-8B"
                            else ("--mode", "scale_gap", "--min-reporting-gap", "0.10")
                        ),
                        "--out-json",
                        "{run_dir}/m0/gate.json",
                        "--out-md",
                        "{run_dir}/m0/gate.md",
                    ],
                    ["{run_dir}/m0/gate.json", "{run_dir}/m0/gate.md"],
                ),
                "decision_path": "{run_dir}/m0/gate.json",
            },
            "teacher_prep": teacher_prep,
            "canary": {
                **_command_spec(
                    "m1_training_canary",
                    [
                        *train_common,
                        "--out-dir",
                        "{run_dir}/canary",
                        "--canary-steps",
                        "10",
                    ],
                    ["{run_dir}/canary"],
                ),
                "manifest_path": "{run_dir}/canary/canary_manifest.json",
            },
            "m1": {
                **_command_spec(
                    f"m1_seed{seed}_formal_pilot",
                    [*train_common, "--out-dir", "{run_dir}/m1"],
                    ["{run_dir}/m1"],
                ),
                "lock_manifest_path": "{run_dir}/m1/lock_manifest.json",
            },
            # These two expected interfaces are intentionally separate from the
            # launcher.  They must implement the contract documented in the
            # runbook before a real campaign reaches this point.
            "ood": {
                **_command_spec(
                    "m1_one_shot_ood",
                    [
                        "{python}",
                        "{repo}/src/analysis/evaluate_metacog_m1_ood.py",
                        "--lock-manifest",
                        "{run_dir}/id_lock/lock_manifest.json",
                        "--attempt-id",
                        "{ood_attempt_id}",
                        "--baseline-decoupled",
                        "{run_dir}/m0/decoupled.json",
                        "--baseline-compositional",
                        "{run_dir}/m0/compositional.json",
                        "--decoupled-battery",
                        "{repo}/data/benchmarks/battery_v4_final.json",
                        "--compositional-battery",
                        "{repo}/data/benchmarks/battery_v3d.json",
                        "--bootstrap-samples",
                        "4000",
                        "--bootstrap-seed",
                        "0",
                        "--out-json",
                        "{run_dir}/ood/result.json",
                    ],
                    ["{run_dir}/ood/result.json"],
                ),
                "result_path": "{run_dir}/ood/result.json",
            },
            "report": {
                **_command_spec(
                    "m1_gate_report",
                    [
                        "{python}",
                        "{repo}/src/analysis/report_metacog_m1.py",
                        "--m0-gate",
                        "{run_dir}/m0/gate.json",
                        "--canary-manifest",
                        "{run_dir}/canary/canary_manifest.json",
                        "--m1-summary",
                        "{run_dir}/m1/summary.json",
                        "--lock-manifest",
                        "{run_dir}/id_lock/lock_manifest.json",
                        "--ood-result",
                        "{run_dir}/ood/result.json",
                        "--decision-ledger",
                        "{run_dir}/decision_ledger.jsonl",
                        "--out",
                        "{run_dir}/report/M1_GATE_REPORT.md",
                    ],
                    ["{run_dir}/report/M1_GATE_REPORT.md"],
                ),
                "report_path": "{run_dir}/report/M1_GATE_REPORT.md",
            },
        },
    }


def _render(value: Any, context: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(context)
        except KeyError as exc:
            raise CampaignError(f"unknown command-plan placeholder: {exc.args[0]}") from exc
    if isinstance(value, list):
        return [_render(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render(item, context) for key, item in value.items()}
    return value


def _all_command_specs(plan: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    stages = plan["stages"]
    result: list[tuple[str, Mapping[str, Any]]] = []
    for spec in stages["m0"]:
        result.append(("m0", spec))
    for stage in ("m0_gate", "teacher_prep", "canary", "m1", "ood", "report"):
        result.append((stage, stages[stage]))
    return result


def validate_plan(plan: Mapping[str, Any], run_dir: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise CampaignError(f"command plan schema must be {PLAN_SCHEMA!r}")
    stages = plan.get("stages")
    if not isinstance(stages, dict):
        raise CampaignError("command plan stages must be an object")
    expected = {
        "m0",
        "m0_gate",
        "teacher_prep",
        "canary",
        "m1",
        "ood",
        "report",
    }
    if set(stages) != expected:
        raise CampaignError(f"command plan must contain exactly these stages: {sorted(expected)}")
    if not isinstance(stages["m0"], list) or not stages["m0"]:
        raise CampaignError("M0 must contain at least one command")

    for stage, spec in _all_command_specs(plan):
        if not isinstance(spec, dict):
            raise CampaignError(f"{stage} command specification must be an object")
        name, argv, outputs = spec.get("name"), spec.get("argv"), spec.get("outputs")
        if not isinstance(name, str) or not name:
            raise CampaignError(f"{stage} command has no name")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise CampaignError(f"{stage}/{name} argv must be a non-empty string array")
        if not isinstance(outputs, list) or not all(isinstance(x, str) and x for x in outputs):
            raise CampaignError(f"{stage}/{name} outputs must be a string array")
        lowered = " ".join(argv).lower()
        forbidden = ("h100", "train_memory_rl", "rl-qa", "rl_qa")
        if any(token in lowered for token in forbidden):
            raise CampaignError(f"forbidden H100/RL command in {stage}/{name}")
        for index, token in enumerate(argv[:-1]):
            if token == "--model" and argv[index + 1] != EXPECTED_MODEL:
                raise CampaignError(f"only the primary target {EXPECTED_MODEL} is allowed")
            if token == "--seed" and argv[index + 1] not in {str(s) for s in ALLOWED_SEEDS}:
                raise CampaignError(f"campaign seed must be one of {ALLOWED_SEEDS}")
        for output in outputs:
            require_within(Path(output), run_dir, f"declared output for {stage}/{name}")

    ood_argv = stages["ood"]["argv"]
    for flag, expected_value in (
        ("--bootstrap-samples", str(DEFAULT_BOOTSTRAP_DRAWS)),
        ("--bootstrap-seed", "0"),
    ):
        if flag not in ood_argv or ood_argv.index(flag) == len(ood_argv) - 1:
            raise CampaignError(f"OOD command must declare {flag}")
        if ood_argv[ood_argv.index(flag) + 1] != expected_value:
            raise CampaignError(f"OOD command {flag} must equal {expected_value}")

    required_paths = {
        "m0_gate": "decision_path",
        "canary": "manifest_path",
        "m1": "lock_manifest_path",
        "ood": "result_path",
        "report": "report_path",
    }
    for stage, field in required_paths.items():
        value = stages[stage].get(field)
        if not isinstance(value, str) or not value:
            raise CampaignError(f"{stage} must declare {field}")
        require_within(Path(value), run_dir, f"{stage}.{field}")


def parse_gpu_inventory(output: str) -> list[dict[str, Any]]:
    rows = [row for row in csv.reader(output.splitlines()) if row]
    if not rows:
        raise CampaignError("nvidia-smi returned no GPU inventory records")
    inventory: list[dict[str, Any]] = []
    seen_indices: set[str] = set()
    seen_uuids: set[str] = set()
    for row in rows:
        if len(row) != 5:
            raise CampaignError(f"cannot parse nvidia-smi GPU inventory row: {row!r}")
        index, gpu_uuid, name, total_raw, free_raw = (field.strip() for field in row)
        if not index or not gpu_uuid or index in seen_indices or gpu_uuid in seen_uuids:
            raise CampaignError("nvidia-smi returned duplicate or empty GPU identity")
        try:
            total_mib, free_mib = int(total_raw), int(free_raw)
        except ValueError as exc:
            raise CampaignError("nvidia-smi returned non-integer memory values") from exc
        inventory.append(
            {
                "index": index,
                "uuid": gpu_uuid,
                "name": name,
                "total_mib": total_mib,
                "free_mib": free_mib,
            }
        )
        seen_indices.add(index)
        seen_uuids.add(gpu_uuid)
    return inventory


def parse_compute_processes(output: str) -> list[dict[str, str]]:
    processes: list[dict[str, str]] = []
    for row in csv.reader(output.splitlines()):
        if not row or not any(field.strip() for field in row):
            continue
        if row[0].strip().lower().startswith("no running processes"):
            continue
        if len(row) != 4:
            raise CampaignError(f"cannot parse nvidia-smi compute-process row: {row!r}")
        processes.append(
            {
                "gpu_uuid": row[0].strip(),
                "pid": row[1].strip(),
                "process_name": row[2].strip(),
                "used_memory_mib": row[3].strip(),
            }
        )
    return processes


def select_campaign_gpu(
    inventory: Sequence[Mapping[str, Any]],
    processes: Sequence[Mapping[str, str]],
    *,
    min_free_mib: int,
    requested_index: str | None,
    allowed_existing_process_mib: int = 0,
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    """Select one physical, idle campaign GPU from a possibly multi-GPU host."""

    by_index = {str(gpu["index"]): dict(gpu) for gpu in inventory}
    if requested_index is not None:
        if requested_index not in by_index:
            raise CampaignError(f"requested GPU index {requested_index!r} is not present")
        candidates = [by_index[requested_index]]
        mode = "explicit_index"
    else:
        candidates = [dict(gpu) for gpu in inventory]
        mode = f"auto_most_free_idle_{GPU_LABEL}"

    eligible: list[dict[str, Any]] = []
    rejection_reasons: list[str] = []
    for gpu in candidates:
        target_processes = [
            dict(process) for process in processes if process.get("gpu_uuid") == gpu["uuid"]
        ]
        reasons = []
        if gpu["name"] != EXPECTED_GPU:
            reasons.append(f"name={gpu['name']!r}")
        if int(gpu["total_mib"]) < min_free_mib or int(gpu["free_mib"]) < min_free_mib:
            reasons.append(
                f"memory total/free={gpu['total_mib']}/{gpu['free_mib']} MiB < {min_free_mib}"
            )
        process_memory = []
        for process in target_processes:
            try:
                process_memory.append(int(process["used_memory_mib"]))
            except (KeyError, TypeError, ValueError):
                reasons.append(f"unparseable_compute_process={process}")
        if sum(process_memory) > allowed_existing_process_mib:
            reasons.append(
                f"compute_process_memory={sum(process_memory)} MiB > explicitly allowed "
                f"{allowed_existing_process_mib} MiB"
            )
        if reasons:
            rejection_reasons.append(f"GPU {gpu['index']}: " + "; ".join(reasons))
        else:
            eligible.append(gpu)
    if not eligible:
        raise CampaignError(
            f"no idle {GPU_LABEL} satisfies campaign preflight: " + " | ".join(rejection_reasons)
        )
    # A deterministic auto-selection chooses one physical card; every child is
    # then pinned to only that index.  Explicit --gpu-index remains preferable.
    selected = sorted(
        eligible,
        key=lambda gpu: (-int(gpu["free_mib"]), str(gpu["index"]).zfill(12)),
    )[0]
    selected_processes = [
        dict(process) for process in processes if process.get("gpu_uuid") == selected["uuid"]
    ]
    return selected, selected_processes, mode


def validate_canary_manifest(path: Path) -> dict[str, Any]:
    payload = load_json_object(path, "canary manifest")
    passed = payload.get("canary_passed") is True
    status = str(payload.get("status", "")).upper()
    if not passed or status not in {"PASS", "PASSED", "GREEN", "COMPLETE"}:
        raise ControlledStop(f"canary did not pass: status={payload.get('status')!r}")
    if payload.get("eligible_for_ood") is not False:
        raise CampaignError("canary manifest must declare eligible_for_ood=false")
    if payload.get("finite_loss_and_gradients") is not True:
        raise ControlledStop("canary loss/gradient finiteness audit did not pass")
    checkpoint = payload.get("checkpoint_save_load")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("passed") is not True
        or checkpoint.get("readable") is not True
        or not isinstance(checkpoint.get("live_peft_roundtrip"), dict)
        or checkpoint["live_peft_roundtrip"].get("passed") is not True
    ):
        raise ControlledStop("canary checkpoint save/load audit did not pass")
    adapter = payload.get("adapter_enable_disable_check")
    if not isinstance(adapter, dict) or adapter.get("passed") is not True:
        raise ControlledStop("canary adapter enable/disable audit did not pass")
    workspace = payload.get("workspace_post_training_evaluation")
    if (
        not isinstance(workspace, dict)
        or workspace.get("performed") is not True
        or workspace.get("all_finite") is not True
        or workspace.get("candidate_rows") != workspace.get("expected_candidate_rows")
    ):
        raise ControlledStop("canary workspace evaluation audit did not pass")
    throughput = payload.get("throughput")
    if not isinstance(throughput, dict) or any(
        isinstance(throughput.get(field), bool)
        or not isinstance(throughput.get(field), (int, float))
        or not math.isfinite(float(throughput[field]))
        or float(throughput[field]) <= 0
        for field in ("mean_tokens_per_second", "mean_examples_per_second")
    ):
        raise ControlledStop("canary throughput audit did not pass")
    return payload


def collect_m0_bindings(
    run_dir: Path,
    *,
    model_revision: str,
    tokenizer_revision: str,
) -> dict[str, Any]:
    """Bind GREEN gate inputs, metadata sidecars, and source batteries."""

    gate_path = run_dir / "m0" / "gate.json"
    gate = load_json_object(gate_path, "M0 gate")
    if gate.get("decision") != "GREEN":
        raise CampaignError("ID lock cannot bind a non-GREEN M0 gate")
    gate_conditions = gate.get("conditions")
    if not isinstance(gate_conditions, dict) or set(gate_conditions) != {
        "explicit",
        "evoked",
        "decoupled",
        "compositional",
    }:
        raise CampaignError("M0 gate does not bind the four required conditions")

    bindings: dict[str, Any] = {}
    for condition in ("explicit", "evoked", "decoupled", "compositional"):
        source = gate_conditions[condition].get("source")
        if not isinstance(source, dict):
            raise CampaignError(f"M0 gate has no source binding for {condition}")
        raw_path = Path(str(source.get("path", "")))
        require_within(raw_path, run_dir, f"M0 {condition} raw result")
        raw_hash = _valid_sha(source.get("sha256"), f"M0 {condition} raw hash")
        if not raw_path.is_file() or raw_path.is_symlink() or sha256_file(raw_path) != raw_hash:
            raise CampaignError(f"M0 {condition} raw result changed after the gate")
        reduced_metadata = source.get("metadata")
        if not isinstance(reduced_metadata, dict):
            raise CampaignError(f"M0 {condition} gate source has no metadata binding")
        metadata_path = Path(str(reduced_metadata.get("path", "")))
        require_within(metadata_path, run_dir, f"M0 {condition} metadata")
        metadata_hash = _valid_sha(
            reduced_metadata.get("sha256"), f"M0 {condition} metadata hash"
        )
        if (
            not metadata_path.is_file()
            or metadata_path.is_symlink()
            or sha256_file(metadata_path) != metadata_hash
        ):
            raise CampaignError(f"M0 {condition} metadata changed after the gate")
        metadata = load_json_object(metadata_path, f"M0 {condition} metadata")
        if metadata.get("model_revision") != model_revision:
            raise CampaignError(f"M0 {condition} model revision differs from campaign pin")
        if metadata.get("tokenizer_revision") != tokenizer_revision:
            raise CampaignError(f"M0 {condition} tokenizer revision differs from campaign pin")
        battery_path = Path(str(metadata.get("battery", "")))
        if not battery_path.is_file() or battery_path.is_symlink():
            raise CampaignError(f"M0 {condition} source battery is missing or unsafe")
        hashes = metadata.get("hashes")
        if not isinstance(hashes, dict):
            raise CampaignError(f"M0 {condition} metadata has no hashes")
        battery_hash = _valid_sha(
            hashes.get("battery_file_sha256"), f"M0 {condition} battery hash"
        )
        if sha256_file(battery_path) != battery_hash:
            raise CampaignError(f"M0 {condition} battery changed after measurement")
        if hashes.get("raw_output_sha256") != raw_hash:
            raise CampaignError(f"M0 {condition} metadata/raw hash binding changed")
        bindings[condition] = {
            "raw_path": raw_path.relative_to(run_dir).as_posix(),
            "raw_sha256": raw_hash,
            "metadata_path": metadata_path.relative_to(run_dir).as_posix(),
            "metadata_sha256": metadata_hash,
            "battery_path": str(battery_path.resolve()),
            "battery_sha256": battery_hash,
        }

    # Evoked-G2 is a training-only teacher source, so the four-condition gate
    # does not consume it.  Bind it independently with the same strict sidecar
    # contract before locking the selected checkpoint.
    condition = "evoked_g2"
    raw_path = run_dir / "m0" / "evoked_g2.json"
    metadata_path = Path(f"{raw_path}.metadata")
    if raw_path.is_symlink() or metadata_path.is_symlink():
        raise CampaignError("M0 Evoked-G2 artifacts must not be symlinks")
    metadata = load_json_object(metadata_path, "M0 Evoked-G2 metadata")
    hashes = metadata.get("hashes")
    if not raw_path.is_file() or not isinstance(hashes, dict):
        raise CampaignError("M0 Evoked-G2 raw/metadata artifacts are incomplete")
    raw_hash = sha256_file(raw_path)
    if hashes.get("raw_output_sha256") != raw_hash:
        raise CampaignError("M0 Evoked-G2 metadata does not bind its raw result")
    if metadata.get("model_revision") != model_revision:
        raise CampaignError("M0 Evoked-G2 model revision differs from campaign pin")
    if metadata.get("tokenizer_revision") != tokenizer_revision:
        raise CampaignError("M0 Evoked-G2 tokenizer revision differs from campaign pin")
    battery_path = Path(str(metadata.get("battery", "")))
    battery_hash = _valid_sha(hashes.get("battery_file_sha256"), "M0 Evoked-G2 battery hash")
    if not battery_path.is_file() or battery_path.is_symlink() or sha256_file(battery_path) != battery_hash:
        raise CampaignError("M0 Evoked-G2 source battery changed after measurement")
    bindings[condition] = {
        "raw_path": raw_path.relative_to(run_dir).as_posix(),
        "raw_sha256": raw_hash,
        "metadata_path": metadata_path.relative_to(run_dir).as_posix(),
        "metadata_sha256": sha256_file(metadata_path),
        "battery_path": str(battery_path.resolve()),
        "battery_sha256": battery_hash,
    }
    return {
        "gate_path": gate_path.relative_to(run_dir).as_posix(),
        "gate_sha256": sha256_file(gate_path),
        "conditions": bindings,
    }


def validate_m0_bindings(bindings: Any, run_dir: Path) -> None:
    if not isinstance(bindings, dict):
        raise CampaignError("campaign ID lock has no M0 artifact bindings")
    gate_relative = bindings.get("gate_path")
    if not isinstance(gate_relative, str) or not gate_relative:
        raise CampaignError("campaign ID lock has no M0 gate path")
    gate_path = run_dir / gate_relative
    require_within(gate_path, run_dir, "locked M0 gate")
    if not gate_path.is_file() or gate_path.is_symlink():
        raise CampaignError("locked M0 gate is missing or unsafe")
    if sha256_file(gate_path) != _valid_sha(bindings.get("gate_sha256"), "M0 gate hash"):
        raise CampaignError("M0 gate changed after ID lock")
    conditions = bindings.get("conditions")
    expected = {"explicit", "evoked", "evoked_g2", "decoupled", "compositional"}
    if not isinstance(conditions, dict) or set(conditions) != expected:
        raise CampaignError("campaign ID lock has incomplete M0 condition bindings")
    for condition, binding in conditions.items():
        if not isinstance(binding, dict):
            raise CampaignError(f"invalid locked M0 binding for {condition}")
        for path_field, hash_field, must_be_inside in (
            ("raw_path", "raw_sha256", True),
            ("metadata_path", "metadata_sha256", True),
            ("battery_path", "battery_sha256", False),
        ):
            raw_value = binding.get(path_field)
            if not isinstance(raw_value, str) or not raw_value:
                raise CampaignError(f"locked M0 {condition} has no {path_field}")
            artifact = (run_dir / raw_value) if must_be_inside else Path(raw_value)
            if must_be_inside:
                require_within(artifact, run_dir, f"locked M0 {condition} {path_field}")
            if artifact.is_symlink() or not artifact.is_file():
                raise CampaignError(f"locked M0 {condition} artifact is missing or unsafe")
            if sha256_file(artifact) != _valid_sha(binding.get(hash_field), hash_field):
                raise CampaignError(f"locked M0 {condition} {path_field} changed")


def _valid_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise CampaignError(f"lock manifest has invalid {field}")
    return value


def validate_trainer_lock(
    source_path: Path,
    m1_dir: Path,
    *,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], Path, str, dict[str, str]]:
    payload = load_json_object(source_path, "trainer lock manifest")
    expected_fields = {
        "campaign_stage": "M1",
        "status": "LOCKED",
        "selection_scope": "id_validation",
        "selection_metric": "verbal_auc",
        "tie_break": "earliest_step",
        "ood_evaluated": False,
        "eligible_for_ood": True,
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise CampaignError(f"trainer lock {field} must equal {expected!r}")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= 500:
        raise CampaignError("trainer lock step must be an integer in [0, 500]")
    validation_auc = payload.get("validation_auc")
    if (
        isinstance(validation_auc, bool)
        or not isinstance(validation_auc, (int, float))
        or not math.isfinite(float(validation_auc))
        or not 0.0 <= float(validation_auc) <= 1.0
    ):
        raise CampaignError("trainer lock validation_auc must be finite and in [0, 1]")
    checkpoint_raw = payload.get("checkpoint_path")
    if not isinstance(checkpoint_raw, str) or not checkpoint_raw:
        raise CampaignError("trainer lock has no checkpoint_path")
    checkpoint = Path(checkpoint_raw)
    if not checkpoint.is_absolute():
        checkpoint = m1_dir / checkpoint
    require_within(checkpoint, m1_dir, "locked checkpoint")
    actual_tree_hash, files = checkpoint_tree_hash(checkpoint)
    recorded_tree_hash = _valid_sha(payload.get("checkpoint_tree_sha256"), "checkpoint_tree_sha256")
    if actual_tree_hash != recorded_tree_hash:
        raise CampaignError(
            f"checkpoint tree hash mismatch: recorded={recorded_tree_hash} actual={actual_tree_hash}"
        )
    # The trainer binds selection to these immutable ID inputs.  Exact field
    # names are part of the agreed launcher/trainer contract.
    for field in ("split_manifest_sha256", "run_config_sha256", "validation_metrics_sha256"):
        _valid_sha(payload.get(field), field)

    candidates = payload.get("candidate_checkpoints")
    if not isinstance(candidates, list) or not candidates:
        raise CampaignError("trainer lock must contain a non-empty candidate_checkpoints table")
    if payload.get("id_selection_table") != candidates:
        raise CampaignError("trainer lock ID selection tables disagree")
    candidate_steps: list[int] = []
    normalized: list[tuple[int, float, str, str]] = []
    for index, row in enumerate(candidates):
        if not isinstance(row, dict):
            raise CampaignError(f"candidate checkpoint row {index} is not an object")
        candidate_step = row.get("step")
        candidate_auc = row.get("verbal_auc")
        if (
            isinstance(candidate_step, bool)
            or not isinstance(candidate_step, int)
            or not 0 <= candidate_step <= 500
        ):
            raise CampaignError(f"candidate checkpoint row {index} has an invalid step")
        if (
            isinstance(candidate_auc, bool)
            or not isinstance(candidate_auc, (int, float))
            or not math.isfinite(float(candidate_auc))
            or not 0.0 <= float(candidate_auc) <= 1.0
        ):
            raise CampaignError(f"candidate checkpoint row {index} has an invalid verbal_auc")
        if row.get("selection_scope") != "id_validation" or row.get("selection_metric") != "verbal_auc":
            raise CampaignError(f"candidate checkpoint row {index} is not ID-only verbal selection")
        candidate_path = row.get("checkpoint_path")
        candidate_hash = row.get("checkpoint_tree_sha256")
        if not isinstance(candidate_path, str) or not candidate_path:
            raise CampaignError(f"candidate checkpoint row {index} has no path")
        _valid_sha(candidate_hash, f"candidate checkpoint row {index} hash")
        candidate_absolute = m1_dir / candidate_path
        require_within(candidate_absolute, m1_dir, f"candidate checkpoint row {index}")
        for metric in ("verbal_within_episode_auc", "yes_rate"):
            value = row.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise CampaignError(f"candidate checkpoint row {index} has invalid {metric}")
        candidate_steps.append(candidate_step)
        normalized.append((candidate_step, float(candidate_auc), candidate_path, candidate_hash))
    if len(set(candidate_steps)) != len(candidate_steps) or candidate_steps != sorted(candidate_steps):
        raise CampaignError("candidate checkpoint steps must be unique and sorted")
    if payload.get("candidate_steps") != candidate_steps:
        raise CampaignError("candidate_steps disagrees with candidate_checkpoints")
    best_auc = max(row[1] for row in normalized)
    expected_best = min((row for row in normalized if row[1] == best_auc), key=lambda row: row[0])
    if expected_best[0] != step or expected_best[1] != float(validation_auc):
        raise CampaignError("locked checkpoint is not the highest ID verbal AUC with earliest tie-break")
    selected_relative = checkpoint.relative_to(m1_dir.resolve()).as_posix()
    if expected_best[2] != selected_relative or expected_best[3] != actual_tree_hash:
        raise CampaignError("locked checkpoint path/hash disagrees with the selected ID table row")

    bound_files = {
        "split_manifest_sha256": m1_dir / "split_manifest.json",
        "run_config_sha256": m1_dir / "run_config.json",
        "provenance_sha256": m1_dir / "provenance.json",
        "validation_metrics_sha256": m1_dir / "validation_metrics.jsonl",
    }
    for field, bound_path in bound_files.items():
        expected_hash = _valid_sha(payload.get(field), field)
        if bound_path.is_symlink() or not bound_path.is_file():
            raise CampaignError(f"trainer lock bound file is missing or unsafe: {bound_path}")
        if sha256_file(bound_path) != expected_hash:
            raise CampaignError(f"trainer lock bound file changed: {bound_path.name}")
    manifest_hash = _valid_sha(payload.get("manifest_sha256"), "manifest_sha256")
    unsigned_manifest = dict(payload)
    unsigned_manifest.pop("manifest_sha256")
    if canonical_json_sha256(unsigned_manifest) != manifest_hash:
        raise CampaignError("trainer lock manifest self-hash is invalid")

    run_config_path = m1_dir / "run_config.json"
    run_config = load_json_object(run_config_path, "M1 run config")
    if run_config.get("model") != EXPECTED_MODEL or run_config.get("seed") != seed:
        raise CampaignError("M1 run config changed the fixed model or seed")
    if model_revision is not None and run_config.get("model_revision") != model_revision:
        raise CampaignError("M1 run config model revision differs from the campaign pin")
    if tokenizer_revision is not None and run_config.get("tokenizer_revision") != tokenizer_revision:
        raise CampaignError("M1 run config tokenizer revision differs from the campaign pin")
    return payload, checkpoint.resolve(), actual_tree_hash, files


def validate_orchestrator_lock(path: Path, run_dir: Path) -> dict[str, Any]:
    payload = load_json_object(path, "campaign ID lock")
    if payload.get("schema_version") != LOCK_SCHEMA:
        raise CampaignError("campaign ID lock has the wrong schema")
    if payload.get("model") != EXPECTED_MODEL:
        raise CampaignError("campaign ID lock changed the primary model")
    for field in ("model_revision", "tokenizer_revision"):
        value = payload.get(field)
        if not isinstance(value, str) or not PINNED_REVISION_RE.fullmatch(value):
            raise CampaignError(f"campaign ID lock has no immutable {field}")
    source_relative = payload.get("source_lock_manifest")
    if not isinstance(source_relative, str) or not source_relative:
        raise CampaignError("campaign ID lock has no source manifest path")
    source_path = run_dir / source_relative
    require_within(source_path, run_dir, "source lock manifest")
    if not source_path.is_file() or source_path.is_symlink():
        raise CampaignError("campaign source lock manifest is missing or unsafe")
    source_hash = _valid_sha(payload.get("source_lock_manifest_sha256"), "source lock hash")
    if sha256_file(source_path) != source_hash:
        raise CampaignError("trainer lock manifest changed after ID lock")
    checkpoint_relative = payload.get("checkpoint_path")
    if not isinstance(checkpoint_relative, str) or not checkpoint_relative:
        raise CampaignError("campaign ID lock has no checkpoint path")
    checkpoint = run_dir / checkpoint_relative
    require_within(checkpoint, run_dir, "campaign locked checkpoint")
    tree_hash, _ = checkpoint_tree_hash(checkpoint)
    if tree_hash != _valid_sha(payload.get("checkpoint_tree_sha256"), "checkpoint tree hash"):
        raise CampaignError("checkpoint changed after ID lock")
    authorization = payload.get("ood_authorization")
    expected = {
        "conditions": ["decoupled", "compositional"],
        "attempt_limit": 1,
        "bootstrap_samples": DEFAULT_BOOTSTRAP_DRAWS,
    }
    if authorization != expected:
        raise CampaignError("campaign lock has invalid OOD authorization")
    validate_m0_bindings(payload.get("m0_artifacts"), run_dir)
    return payload


def _ood_number(value: Any, label: str, *, low: float = -1.0, high: float = 1.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not low <= float(value) <= high
    ):
        raise CampaignError(f"OOD result has invalid {label}")
    return float(value)


def _expect_close(
    observed: Any,
    expected: float,
    label: str,
    tolerance: float = 1e-10,
    *,
    low: float = -1.0,
    high: float = 1.0,
) -> None:
    value = _ood_number(observed, label, low=low, high=high)
    if not math.isclose(value, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise CampaignError(f"OOD result {label} mismatch: recorded={value} recomputed={expected}")


def _ood_auc(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    labels = [row["label"] == "load_bearing" for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise CampaignError("OOD AUC requires both utility classes")
    scores = [float(row[key]) for row in rows]
    order = sorted(range(len(rows)), key=lambda index: scores[index])
    ranks = [0.0] * len(rows)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for offset in range(cursor, end):
            ranks[order[offset]] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _validate_ood_rows(raw_rows: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list) or not raw_rows:
        raise CampaignError(f"OOD {label} per_item rows must be a non-empty array")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise CampaignError(f"OOD {label} row {index} is not an object")
        episode, candidate = raw.get("episode"), raw.get("candidate_index")
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode < 0
            or isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < 0
        ):
            raise CampaignError(f"OOD {label} row {index} has invalid identity")
        identity = (episode, candidate)
        if identity in identities:
            raise CampaignError(f"OOD {label} duplicates candidate identity {identity}")
        identities.add(identity)
        if not isinstance(raw.get("concept"), str) or not raw["concept"]:
            raise CampaignError(f"OOD {label} row {index} has invalid concept")
        if raw.get("label") not in {"load_bearing", "distractor", "filler"}:
            raise CampaignError(f"OOD {label} row {index} has invalid utility label")
        for key in ("V", "W_rr"):
            _ood_number(raw.get(key), f"{label}[{index}].{key}", low=0.0, high=1.0)
        rows.append(dict(raw))
    return rows


def _ood_channel_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    by_episode: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    within = []
    for episode_rows in by_episode.values():
        positives = sum(row["label"] == "load_bearing" for row in episode_rows)
        if 0 < positives < len(episode_rows):
            within.append(_ood_auc(episode_rows, key))
    if not within:
        raise CampaignError("OOD result has no evaluable within-episode AUC")
    result = {
        "pooled_auc": _ood_auc(rows, key),
        "within_episode_auc": sum(within) / len(within),
        "n_within_episode_auc": len(within),
        "candidate_count": len(rows),
        "episode_count": len(by_episode),
    }
    if key == "V":
        result["yes_rate"] = sum(float(row[key]) >= 0.5 for row in rows) / len(rows)
    return result


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ood_bootstrap(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    key: str,
    *,
    samples: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    before_by_episode: dict[int, list[Mapping[str, Any]]] = {}
    after_by_episode: dict[int, list[Mapping[str, Any]]] = {}
    for row in before:
        before_by_episode.setdefault(int(row["episode"]), []).append(row)
    for row in after:
        after_by_episode.setdefault(int(row["episode"]), []).append(row)
    episodes = sorted(before_by_episode)
    if episodes != sorted(after_by_episode):
        raise CampaignError("OOD before/after episode sets differ")
    rng = random.Random(seed)
    before_draws, after_draws, delta_draws = [], [], []
    for _ in range(samples):
        sampled = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        sampled_before = [row for episode in sampled for row in before_by_episode[episode]]
        sampled_after = [row for episode in sampled for row in after_by_episode[episode]]
        before_auc = _ood_auc(sampled_before, key)
        after_auc = _ood_auc(sampled_after, key)
        before_draws.append(before_auc)
        after_draws.append(after_auc)
        delta_draws.append(after_auc - before_auc)
    return before_draws, after_draws, delta_draws


def _validate_bootstrap_interval(
    raw: Any,
    *,
    estimate: float,
    draws: Sequence[float],
    label: str,
) -> None:
    if not isinstance(raw, dict):
        raise CampaignError(f"OOD {label} bootstrap interval is missing")
    _expect_close(raw.get("estimate"), estimate, f"{label}.estimate")
    if raw.get("bootstrap_samples_effective") != DEFAULT_BOOTSTRAP_DRAWS:
        raise CampaignError(f"OOD {label} must have 4,000 effective draws")
    ci = raw.get("ci_95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise CampaignError(f"OOD {label} ci_95 must have two endpoints")
    _expect_close(ci[0], _percentile(draws, 0.025), f"{label}.ci_95[0]")
    _expect_close(ci[1], _percentile(draws, 0.975), f"{label}.ci_95[1]")


def _validate_condition_payload(
    raw: Any,
    *,
    condition: str,
    bootstrap_seed: int,
) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise CampaignError(f"OOD {condition} payload must be an object")
    per_item = raw.get("per_item")
    if not isinstance(per_item, dict):
        raise CampaignError(f"OOD {condition} has no per_item audit rows")
    before = _validate_ood_rows(per_item.get("before"), f"{condition}.before")
    after = _validate_ood_rows(per_item.get("after"), f"{condition}.after")
    identities_before = [
        (row["episode"], row["candidate_index"], row["concept"], row["label"])
        for row in before
    ]
    identities_after = [
        (row["episode"], row["candidate_index"], row["concept"], row["label"])
        for row in after
    ]
    if identities_before != identities_after:
        raise CampaignError(f"OOD {condition} before/after candidate identity differs")
    expected_identity_hash = canonical_json_sha256([list(item) for item in identities_before])
    if raw.get("candidate_identity_sha256") != expected_identity_hash:
        raise CampaignError(f"OOD {condition} candidate identity hash mismatch")

    points: dict[str, float] = {}
    for channel, key, seed_offset in (("verbal", "V", 0), ("workspace", "W_rr", 1)):
        metrics = raw.get(channel)
        if not isinstance(metrics, dict):
            raise CampaignError(f"OOD {condition}.{channel} metrics are missing")
        before_summary = _ood_channel_summary(before, key)
        after_summary = _ood_channel_summary(after, key)
        for scope, summary in (("before", before_summary), ("after", after_summary)):
            recorded = metrics.get(scope)
            if not isinstance(recorded, dict):
                raise CampaignError(f"OOD {condition}.{channel}.{scope} is missing")
            for field, expected in summary.items():
                if isinstance(expected, int):
                    if recorded.get(field) != expected:
                        raise CampaignError(
                            f"OOD {condition}.{channel}.{scope}.{field} count mismatch"
                        )
                else:
                    _expect_close(
                        recorded.get(field),
                        float(expected),
                        f"{condition}.{channel}.{scope}.{field}",
                    )
            if key == "W_rr" and "yes_rate" in recorded:
                raise CampaignError("workspace W_rr must not be reported as a pseudo Yes rate")
        delta_pooled = after_summary["pooled_auc"] - before_summary["pooled_auc"]
        delta_within = (
            after_summary["within_episode_auc"] - before_summary["within_episode_auc"]
        )
        _expect_close(
            metrics.get("delta_pooled_auc"),
            delta_pooled,
            f"{condition}.{channel}.delta_pooled_auc",
        )
        _expect_close(
            metrics.get("delta_within_episode_auc"),
            delta_within,
            f"{condition}.{channel}.delta_within_episode_auc",
        )
        if key == "V":
            _expect_close(
                metrics.get("delta_yes_rate"),
                after_summary["yes_rate"] - before_summary["yes_rate"],
                f"{condition}.verbal.delta_yes_rate",
            )
        elif "delta_yes_rate" in metrics:
            raise CampaignError("workspace W_rr must not have delta_yes_rate")
        before_draws, after_draws, delta_draws = _ood_bootstrap(
            before,
            after,
            key,
            samples=DEFAULT_BOOTSTRAP_DRAWS,
            seed=bootstrap_seed + seed_offset,
        )
        bootstrap = metrics.get("paired_episode_bootstrap")
        if not isinstance(bootstrap, dict):
            raise CampaignError(f"OOD {condition}.{channel} bootstrap is missing")
        _validate_bootstrap_interval(
            bootstrap.get("before"),
            estimate=before_summary["pooled_auc"],
            draws=before_draws,
            label=f"{condition}.{channel}.before",
        )
        _validate_bootstrap_interval(
            bootstrap.get("after"),
            estimate=after_summary["pooled_auc"],
            draws=after_draws,
            label=f"{condition}.{channel}.after",
        )
        delta_interval = bootstrap.get("after_minus_before")
        _validate_bootstrap_interval(
            delta_interval,
            estimate=delta_pooled,
            draws=delta_draws,
            label=f"{condition}.{channel}.after_minus_before",
        )
        _expect_close(
            delta_interval.get("probability_gt_zero"),
            sum(value > 0 for value in delta_draws) / len(delta_draws),
            f"{condition}.{channel}.probability_gt_zero",
            tolerance=1e-10,
        )
        points[f"{channel}_before"] = before_summary["pooled_auc"]
        points[f"{channel}_after"] = after_summary["pooled_auc"]

    qa = raw.get("full_context_qa")
    if not isinstance(qa, dict) or not isinstance(qa.get("per_episode"), dict):
        raise CampaignError(f"OOD {condition} full-context QA audit is missing")
    qa_before = qa["per_episode"].get("before")
    qa_after = qa["per_episode"].get("after")
    if not isinstance(qa_before, list) or not qa_before or not isinstance(qa_after, list):
        raise CampaignError(f"OOD {condition} full-context QA rows are missing")
    before_ids, after_ids = [], []
    for rows, identities, scope in (
        (qa_before, before_ids, "before"),
        (qa_after, after_ids, "after"),
    ):
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("episode"), int):
                raise CampaignError(f"OOD {condition} QA {scope} row {index} is invalid")
            if not isinstance(row.get("answer"), str) or not isinstance(row.get("correct"), bool):
                raise CampaignError(f"OOD {condition} QA {scope} row {index} lacks answer/correct")
            identities.append(row["episode"])
    if before_ids != after_ids or len(set(before_ids)) != len(before_ids):
        raise CampaignError(f"OOD {condition} QA before/after episode identity differs")
    qa_before_accuracy = sum(row["correct"] for row in qa_before) / len(qa_before)
    qa_after_accuracy = sum(row["correct"] for row in qa_after) / len(qa_after)
    qa_delta = qa_after_accuracy - qa_before_accuracy
    _expect_close(qa.get("before_accuracy"), qa_before_accuracy, f"{condition}.qa.before")
    _expect_close(qa.get("after_accuracy"), qa_after_accuracy, f"{condition}.qa.after")
    _expect_close(qa.get("after_minus_before"), qa_delta, f"{condition}.qa.delta")
    _expect_close(
        qa.get("drop_percentage_points"),
        -100.0 * qa_delta,
        f"{condition}.qa.drop_pp",
        tolerance=1e-10,
        low=-100.0,
        high=100.0,
    )
    qa_rng = random.Random(bootstrap_seed + 2)
    qa_draws = []
    for _ in range(DEFAULT_BOOTSTRAP_DRAWS):
        sampled = [qa_rng.randrange(len(qa_before)) for _ in qa_before]
        before_accuracy = sum(qa_before[index]["correct"] for index in sampled) / len(sampled)
        after_accuracy = sum(qa_after[index]["correct"] for index in sampled) / len(sampled)
        qa_draws.append(after_accuracy - before_accuracy)
    _validate_bootstrap_interval(
        qa.get("paired_episode_bootstrap"),
        estimate=qa_delta,
        draws=qa_draws,
        label=f"{condition}.qa.after_minus_before",
    )
    points["qa_drop_pp"] = -100.0 * qa_delta
    return points


def _expected_ood_gate(points: Mapping[str, float]) -> tuple[str, bool, bool]:
    delta_v = points["verbal_after"] - points["verbal_before"]
    delta_w = points["workspace_after"] - points["workspace_before"]
    qa_drop_pp = points["qa_drop_pp"]
    if delta_v <= 0.05 or delta_w < -0.03 or qa_drop_pp > 2.0:
        return "RED", False, False
    if (
        delta_v >= 0.15
        and points["verbal_after"] > 0.50
        and abs(delta_w) < 0.03
        and qa_drop_pp <= 2.0
    ):
        return "GREEN", delta_v >= 0.25, False
    controlled = 0.05 < delta_v < 0.15 and abs(delta_w) < 0.03 and qa_drop_pp <= 2.0
    return "AMBER", False, controlled


def validate_ood_result(
    path: Path,
    lock_hash: str,
    attempt_id: str,
    *,
    lock: Mapping[str, Any],
    run_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    payload = load_json_object(path, "one-shot OOD result")
    if payload.get("schema_version") != "metacog-alignment-m1-ood/v1" or payload.get("stage") != "M1_OOD":
        raise CampaignError("OOD result has the wrong schema or stage")
    if payload.get("lock_manifest_sha256") != lock_hash:
        raise CampaignError("OOD result is not bound to the validated campaign lock")
    if payload.get("attempt_id") != attempt_id:
        raise CampaignError("OOD result attempt_id does not match the precommitted attempt")
    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, dict) or bootstrap.get("samples") != DEFAULT_BOOTSTRAP_DRAWS:
        raise CampaignError("OOD result must use 4,000 bootstrap draws")
    if bootstrap != {
        "samples": DEFAULT_BOOTSTRAP_DRAWS,
        "seed": 0,
        "unit": "episode_cluster",
        "paired_before_after": True,
    }:
        raise CampaignError("OOD bootstrap protocol differs from the locked 4,000-draw contract")
    if payload.get("model") != lock.get("model"):
        raise CampaignError("OOD result model differs from the campaign lock")
    for field in ("model_revision", "tokenizer_revision", "checkpoint_tree_sha256"):
        if payload.get(field) != lock.get(field):
            raise CampaignError(f"OOD result {field} differs from the campaign lock")
    checkpoint = Path(str(payload.get("checkpoint_path", ""))).resolve()
    expected_checkpoint = (run_dir / str(lock["checkpoint_path"])).resolve()
    if checkpoint != expected_checkpoint:
        raise CampaignError("OOD result checkpoint path differs from the campaign lock")
    conditions = payload.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != {"decoupled", "compositional"}:
        raise CampaignError("OOD result must contain exactly Decoupled and Compositional")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"decoupled", "compositional"}:
        raise CampaignError("OOD result input bindings are incomplete")
    for condition in ("decoupled", "compositional"):
        binding = lock["m0_artifacts"]["conditions"][condition]
        baseline_path = run_dir / binding["raw_path"]
        battery_path = Path(binding["battery_path"])
        expected_input = {
            "baseline_path": str(baseline_path.resolve()),
            "baseline_sha256": binding["raw_sha256"],
            "battery_path": str(battery_path.resolve()),
            "battery_sha256": binding["battery_sha256"],
        }
        if inputs.get(condition) != expected_input:
            raise CampaignError(f"OOD {condition} input provenance differs from the lock")
        locked_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        per_item = conditions[condition].get("per_item")
        if not isinstance(per_item, dict) or per_item.get("before") != locked_baseline:
            raise CampaignError(f"OOD {condition} before rows differ from locked M0 baseline")
    points_by_condition = {
        condition: _validate_condition_payload(
            conditions[condition],
            condition=condition,
            bootstrap_seed=bootstrap["seed"] + index * 10,
        )
        for index, condition in enumerate(("decoupled", "compositional"))
    }
    expected_decision, expected_strong, expected_controlled = _expected_ood_gate(
        points_by_condition["decoupled"]
    )
    if payload.get("decision") != expected_decision:
        raise CampaignError("OOD GREEN/AMBER/RED decision does not match recomputed metrics")
    if payload.get("strong_green") is not expected_strong:
        raise CampaignError("OOD strong_green does not match recomputed metrics")
    if payload.get("controlled_branch_authorized") is not expected_controlled:
        raise CampaignError("OOD controlled-branch authorization is invalid")
    protocol = payload.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("checkpoint_selected_on") != "id_validation_only"
        or protocol.get("qa_prompt_and_item_order_identical") is not True
        or protocol.get("h100_used") is not False
    ):
        raise CampaignError("OOD result protocol provenance is incomplete")
    provenance = payload.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("gpu_name") != EXPECTED_GPU
        or provenance.get("parameter_dtype") != "torch.bfloat16"
        or provenance.get("model_revision_requested") != lock["model_revision"]
        or provenance.get("tokenizer_revision_requested") != lock["tokenizer_revision"]
    ):
        raise CampaignError("OOD runtime/model provenance is incomplete")
    yes_ids, no_ids = provenance.get("yes_token_ids"), provenance.get("no_token_ids")
    if (
        not isinstance(yes_ids, list)
        or not yes_ids
        or not isinstance(no_ids, list)
        or not no_ids
        or set(yes_ids) & set(no_ids)
    ):
        raise CampaignError("OOD Yes/No token provenance is invalid")
    source_root = repo_root.resolve() if repo_root is not None else None
    for field, relative in (
        ("evaluator_source_sha256", Path("src/analysis/evaluate_metacog_m1_ood.py")),
        ("measure_source_sha256", Path("src/experiments/measure.py")),
        ("workspace_lens_source_sha256", Path("src/jlens.py")),
        ("recall_source_sha256", Path("src/memory_rl/recall.py")),
    ):
        source = source_root / relative if source_root is not None else None
        if source is not None and source.is_file() and provenance.get(field) != sha256_file(source):
            raise CampaignError(f"OOD {field} does not match the executed source")
        _valid_sha(provenance.get(field), field)
    _valid_sha(provenance.get("chat_template_sha256"), "OOD chat template hash")
    return payload


REPORT_MARKERS = (
    "m0 baseline reproduction status",
    "model/tokenizer revisions",
    f"{GPU_LABEL} memory/throughput configuration",
    "teacher-label construction audit",
    "training configuration",
    "loss/gradient health",
    "id checkpoint-selection table",
    "locked checkpoint",
    "decoupled v before/after",
    "decoupled w before/after",
    "compositional v/w before/after",
    "within-episode metrics",
    "yes-rate before/after",
    "full-context qa before/after",
    "bootstrap cis",
    "green / amber / red decision",
    "artifact paths and hashes",
    "no h100 job was launched",
)


def validate_report(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"missing or unsafe M1 report: {path}")
    text = " ".join(path.read_text(encoding="utf-8").lower().split())
    missing = [marker for marker in REPORT_MARKERS if marker not in text]
    if missing:
        raise CampaignError("M1 report is missing required section(s): " + ", ".join(missing))


class CampaignRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        run_dir: Path,
        plan: Mapping[str, Any],
        model_revision: str,
        tokenizer_revision: str,
        nvidia_smi: str,
        min_free_mib: int,
        requested_gpu_index: str | None,
        allowed_existing_process_mib: int,
        attempt_id: str,
        seed: int = 0,
        resume: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.run_dir = run_dir.resolve()
        self.plan = plan
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.nvidia_smi = nvidia_smi
        self.min_free_mib = min_free_mib
        self.requested_gpu_index = requested_gpu_index
        self.allowed_existing_process_mib = allowed_existing_process_mib
        self.attempt_id = attempt_id
        if seed not in ALLOWED_SEEDS:
            raise CampaignError(f"campaign seed must be one of {ALLOWED_SEEDS}")
        self.seed = seed
        self.resume = resume
        # A resumed OOD phase appends to the same ledger and must not reuse a
        # command-log filename from the ID phase.
        existing_logs = sorted((self.run_dir / "command_logs").glob("*.log")) if resume else []
        self.command_counter = len(existing_logs)
        self.gpu_index: str | None = None
        self.gpu_lock_handle: Any | None = None
        self.gpu: dict[str, Any] | None = None
        self.ledger = DecisionLedger(
            self.run_dir / "decision_ledger.jsonl", append_existing=resume
        )

    def _run_command(
        self,
        stage: str,
        spec: Mapping[str, Any],
        *,
        capture: bool = False,
    ) -> CommandResult:
        argv = tuple(str(item) for item in spec["argv"])
        outputs = [Path(item) for item in spec.get("outputs", [])]
        for output in outputs:
            require_within(output, self.run_dir, f"output of {spec['name']}")
            if output.exists() or output.is_symlink():
                raise CampaignError(f"refusing to overwrite command output: {output}")

        self.command_counter += 1
        log_dir = self.run_dir / "command_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.command_counter:02d}_{_safe_slug(str(spec['name']))}.log"
        if log_path.exists() or log_path.is_symlink():
            raise CampaignError(f"refusing to overwrite command log: {log_path}")
        env = os.environ.copy()
        if self.gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_index
        self.ledger.append(
            "command_started",
            stage=stage,
            name=spec["name"],
            argv=list(argv),
            argv_shell_escaped=shlex.join(argv),
            cwd=str(self.repo_root),
            cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
            expected_outputs=[str(path.relative_to(self.run_dir)) for path in outputs],
        )
        started = time.monotonic()
        captured: list[str] | None = [] if capture else None
        exit_code = 127
        with log_path.open("x", encoding="utf-8") as log_handle:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=self.repo_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log_handle.write(line)
                    log_handle.flush()
                    if capture:
                        assert captured is not None
                        captured.append(line)
                    else:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                exit_code = process.wait()
            except OSError as exc:
                message = f"launcher could not execute command: {exc}\n"
                log_handle.write(message)
                if capture:
                    assert captured is not None
                    captured.append(message)
            log_handle.flush()
            os.fsync(log_handle.fileno())

        output_hashes: dict[str, str] = {}
        outputs_complete = all(path.exists() and not path.is_symlink() for path in outputs)
        if outputs_complete:
            output_hashes = artifact_hashes(outputs, self.run_dir)
        log_hash = sha256_file(log_path)
        self.ledger.append(
            "command_finished",
            stage=stage,
            name=spec["name"],
            argv=list(argv),
            exit_code=exit_code,
            duration_seconds=round(time.monotonic() - started, 3),
            log_path=log_path.relative_to(self.run_dir).as_posix(),
            log_sha256=log_hash,
            outputs_complete=outputs_complete,
            artifact_hashes=output_hashes,
        )
        return CommandResult(
            argv=argv,
            exit_code=exit_code,
            log_path=log_path,
            log_sha256=log_hash,
            output="".join(captured) if captured is not None else None,
            artifact_hashes=output_hashes,
        )

    def _must_succeed(self, stage: str, spec: Mapping[str, Any]) -> CommandResult:
        result = self._run_command(stage, spec)
        if result.exit_code != 0:
            raise CampaignError(f"{stage} command failed with exit status {result.exit_code}")
        if not result.artifact_hashes and spec.get("outputs"):
            raise CampaignError(f"{stage} command did not create every declared output")
        return result

    def gpu_preflight(self) -> dict[str, Any]:
        inventory_spec = _command_spec(
            "nvidia_smi_inventory",
            [
                self.nvidia_smi,
                "--query-gpu=index,uuid,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            [],
        )
        inventory_result = self._run_command("gpu_preflight", inventory_spec, capture=True)
        if inventory_result.exit_code != 0:
            raise CampaignError("nvidia-smi inventory query failed")
        inventory = parse_gpu_inventory(inventory_result.output or "")

        processes_spec = _command_spec(
            "nvidia_smi_compute_processes",
            [
                self.nvidia_smi,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            [],
        )
        process_result = self._run_command("gpu_preflight", processes_spec, capture=True)
        if process_result.exit_code != 0:
            raise CampaignError("nvidia-smi process query failed")
        processes = parse_compute_processes(process_result.output or "")
        gpu, selected_processes, selection_mode = select_campaign_gpu(
            inventory,
            processes,
            min_free_mib=self.min_free_mib,
            requested_index=self.requested_gpu_index,
            allowed_existing_process_mib=self.allowed_existing_process_mib,
        )

        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.gpu_index = str(gpu["index"])
        # Key the exclusion lock on the PHYSICAL GPU UUID, not on the visible
        # index.  Under a cgroup-isolated scheduler every job sees its own card
        # as index 0, so an index-keyed lock made two different physical GPUs
        # on one node collide.  The UUID is what "one campaign per GPU"
        # actually means.
        lock_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(gpu.get("uuid") or self.gpu_index))
        lock_root = Path(os.environ.get("METACOG_LOCK_DIR", "/tmp"))
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f"metacog_alignment_{GPU_LABEL}_{lock_name}.lock"
        self.gpu_lock_handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.gpu_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.gpu_lock_handle.close()
            self.gpu_lock_handle = None
            raise CampaignError(f"campaign GPU lock is already held: {lock_path}") from exc
        self.ledger.append(
            "gpu_preflight_passed",
            gpu=gpu,
            host_gpu_count=len(inventory),
            selection_mode=selection_mode,
            requested_gpu_index=self.requested_gpu_index,
            inherited_cuda_visible_devices=inherited,
            existing_compute_processes_on_selected_gpu=selected_processes,
            allowed_existing_process_mib=self.allowed_existing_process_mib,
            minimum_free_mib=self.min_free_mib,
            pinned_cuda_visible_devices=self.gpu_index,
            process_records_checked=len(processes),
            gpu_lock_path=str(lock_path),
        )
        self.gpu = gpu
        return gpu

    def create_id_lock(self, source_path: Path) -> Path:
        trainer_lock, checkpoint, tree_hash, files = validate_trainer_lock(
            source_path,
            self.run_dir / "m1",
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            seed=self.seed,
        )
        source_hash = sha256_file(source_path)
        m0_bindings = collect_m0_bindings(
            self.run_dir,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        lock_path = self.run_dir / "id_lock" / "lock_manifest.json"
        lock_payload = {
            "schema_version": LOCK_SCHEMA,
            "campaign_schema": CAMPAIGN_SCHEMA,
            "created_at_utc": utc_now(),
            "source_lock_manifest": source_path.relative_to(self.run_dir).as_posix(),
            "source_lock_manifest_sha256": source_hash,
            "checkpoint_path": checkpoint.relative_to(self.run_dir).as_posix(),
            "checkpoint_tree_sha256": tree_hash,
            "checkpoint_file_hashes": files,
            "model": EXPECTED_MODEL,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "selection_scope": trainer_lock["selection_scope"],
            "selection_metric": trainer_lock["selection_metric"],
            "tie_break": trainer_lock["tie_break"],
            "step": trainer_lock["step"],
            "validation_auc": trainer_lock["validation_auc"],
            "candidate_steps": trainer_lock["candidate_steps"],
            "candidate_checkpoints": trainer_lock["candidate_checkpoints"],
            "m0_artifacts": m0_bindings,
            "ood_evaluated_at_lock": False,
            "ood_authorization": {
                "conditions": ["decoupled", "compositional"],
                "attempt_limit": 1,
                "bootstrap_samples": DEFAULT_BOOTSTRAP_DRAWS,
            },
        }
        write_json_exclusive(lock_path, lock_payload)
        self.ledger.append(
            "id_checkpoint_locked",
            source_manifest=source_path.relative_to(self.run_dir).as_posix(),
            source_manifest_sha256=source_hash,
            lock_manifest=lock_path.relative_to(self.run_dir).as_posix(),
            lock_manifest_sha256=sha256_file(lock_path),
            checkpoint_path=checkpoint.relative_to(self.run_dir).as_posix(),
            checkpoint_tree_sha256=tree_hash,
            selection_scope="id_validation",
            selection_metric="verbal_auc",
            tie_break="earliest_step",
        )
        return lock_path

    def one_shot_ood(self, spec: Mapping[str, Any], lock_path: Path) -> dict[str, Any]:
        lock = validate_orchestrator_lock(lock_path, self.run_dir)
        lock_hash = sha256_file(lock_path)
        marker_path = self.run_dir / "ood" / "attempt_started.json"
        write_json_exclusive(
            marker_path,
            {
                "schema_version": CAMPAIGN_SCHEMA,
                "stage": "M1_OOD",
                "attempt_id": self.attempt_id,
                "started_at_utc": utc_now(),
                "lock_manifest": lock_path.relative_to(self.run_dir).as_posix(),
                "lock_manifest_sha256": lock_hash,
                "conditions": ["decoupled", "compositional"],
                "attempt_limit": 1,
                "bootstrap": {
                    "samples": DEFAULT_BOOTSTRAP_DRAWS,
                    "seed": 0,
                    "unit": "episode_cluster",
                },
                "m0_artifacts": {
                    condition: lock["m0_artifacts"]["conditions"][condition]
                    for condition in ("decoupled", "compositional")
                },
                "command_protocol": {
                    "name": spec["name"],
                    "argv": list(spec["argv"]),
                    "argv_sha256": canonical_json_sha256(list(spec["argv"])),
                },
            },
        )
        self.ledger.append(
            "ood_attempt_consumed",
            attempt_id=self.attempt_id,
            attempt_marker=marker_path.relative_to(self.run_dir).as_posix(),
            attempt_marker_sha256=sha256_file(marker_path),
            lock_manifest_sha256=lock_hash,
            conditions=["decoupled", "compositional"],
        )
        result = self._run_command("ood", spec)
        result_marker = self.run_dir / "ood" / "attempt_result.json"
        write_json_exclusive(
            result_marker,
            {
                "schema_version": CAMPAIGN_SCHEMA,
                "attempt_id": self.attempt_id,
                "finished_at_utc": utc_now(),
                "exit_code": result.exit_code,
                "command_log": result.log_path.relative_to(self.run_dir).as_posix(),
                "command_log_sha256": result.log_sha256,
                "artifact_hashes": result.artifact_hashes,
                "retry_permitted": False,
            },
        )
        self.ledger.append(
            "ood_attempt_finished",
            attempt_id=self.attempt_id,
            exit_code=result.exit_code,
            retry_permitted=False,
            result_marker=result_marker.relative_to(self.run_dir).as_posix(),
            result_marker_sha256=sha256_file(result_marker),
        )
        if result.exit_code != 0:
            raise CampaignError(
                f"one-shot OOD command failed with exit status {result.exit_code}; retry is forbidden"
            )
        if not result.artifact_hashes:
            raise CampaignError("one-shot OOD command did not create its declared result")
        result_path = Path(spec["result_path"])
        return validate_ood_result(
            result_path,
            lock_hash,
            self.attempt_id,
            lock=lock,
            run_dir=self.run_dir,
            repo_root=self.repo_root,
        )

    def run(self, *, stop_after: str = "report") -> str:
        if stop_after not in {"id_lock", "report"}:
            raise CampaignError("--stop-after must be 'id_lock' or 'report'")
        self.ledger.append(
            "campaign_started",
            schema_version=CAMPAIGN_SCHEMA,
            run_dir=str(self.run_dir),
            pipeline=list(PIPELINE_STAGES),
            model=EXPECTED_MODEL,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            seed=self.seed,
            h100_stage_present=False,
            rl_campaign_reused=False,
        )
        gpu = self.gpu_preflight()
        stages = self.plan["stages"]

        for spec in stages["m0"]:
            self._must_succeed("m0", spec)
        # The gate CLI intentionally returns 1 after writing an INVESTIGATE
        # decision.  Read and ledger that decision instead of treating it as a
        # malformed run; every other non-zero/missing-output case fails closed.
        gate_result = self._run_command("m0_gate", stages["m0_gate"])
        if gate_result.exit_code not in {0, 1} or not gate_result.artifact_hashes:
            raise CampaignError(
                f"M0 gate command failed with exit status {gate_result.exit_code}"
            )
        gate_path = Path(stages["m0_gate"]["decision_path"])
        gate = load_json_object(gate_path, "M0 gate")
        decision = gate.get("decision")
        if not isinstance(decision, str):
            raise CampaignError("M0 gate has no string decision")
        decision = decision.upper()
        self.ledger.append(
            "m0_gate_decision",
            decision=decision,
            gate_path=gate_path.relative_to(self.run_dir).as_posix(),
            gate_sha256=sha256_file(gate_path),
        )
        if decision != "GREEN":
            raise ControlledStop(f"M0 decision is {decision}; training is forbidden")

        # Evoked-G2 is an ID-only frozen-teacher source, not part of the
        # four-condition M0 reproduction.  Prepare it only after M0 is GREEN.
        self._must_succeed("teacher_prep", stages["teacher_prep"])

        canary_result = self._run_command("canary", stages["canary"])
        if canary_result.exit_code != 0:
            raise ControlledStop(
                f"canary command failed with exit status {canary_result.exit_code}; formal M1 is forbidden"
            )
        if not canary_result.artifact_hashes:
            raise ControlledStop("canary did not create its manifest; formal M1 is forbidden")
        canary_path = Path(stages["canary"]["manifest_path"])
        canary = validate_canary_manifest(canary_path)
        self.ledger.append(
            "canary_gate_decision",
            decision="PASS",
            manifest=canary_path.relative_to(self.run_dir).as_posix(),
            manifest_sha256=sha256_file(canary_path),
            canary_status=canary.get("status"),
        )

        self._must_succeed("m1", stages["m1"])
        source_lock = Path(stages["m1"]["lock_manifest_path"])
        campaign_lock = self.create_id_lock(source_lock)

        if stop_after == "id_lock":
            return self._pause_at_id_lock(campaign_lock)

        return self._ood_and_report(campaign_lock)

    def _pause_at_id_lock(self, campaign_lock: Path) -> str:
        """Stop with the ID lock in place and OOD still sealed.

        The multi-seed contract requires every seed to hold an ID-only
        checkpoint lock before ANY seed is allowed to touch Decoupled or
        Compositional, so the campaign is deliberately run in two phases.
        """

        pause_path = self.run_dir / "ID_LOCK_STOP.json"
        write_json_exclusive(
            pause_path,
            {
                "schema_version": CAMPAIGN_SCHEMA,
                "stopped_at_utc": utc_now(),
                "reason": "id_lock_complete_ood_sealed",
                "seed": self.seed,
                "lock_manifest": campaign_lock.relative_to(self.run_dir).as_posix(),
                "lock_manifest_sha256": sha256_file(campaign_lock),
                "gpu": self.gpu,
                "ood_opened": False,
                "h100_launched": False,
            },
        )
        self.ledger.append(
            "campaign_paused_at_id_lock",
            reason="all seeds must be locked before any OOD attempt",
            seed=self.seed,
            lock_manifest=campaign_lock.relative_to(self.run_dir).as_posix(),
            lock_manifest_sha256=sha256_file(campaign_lock),
            pause_marker=pause_path.relative_to(self.run_dir).as_posix(),
            pause_marker_sha256=sha256_file(pause_path),
            ood_opened=False,
            h100_launched=False,
        )
        return "id_lock_complete_ood_sealed"

    def resume_ood(self) -> str:
        """Second phase: one-shot OOD plus report against an existing ID lock."""

        campaign_lock = self.run_dir / "id_lock" / "lock_manifest.json"
        if campaign_lock.is_symlink() or not campaign_lock.is_file():
            raise CampaignError(f"cannot resume: missing ID lock manifest: {campaign_lock}")
        if not (self.run_dir / "ID_LOCK_STOP.json").is_file():
            raise CampaignError("cannot resume: the ID phase did not stop cleanly at its lock")
        if (self.run_dir / "ood").exists() or (self.run_dir / "STOP.json").exists():
            raise CampaignError("cannot resume: this run already consumed its OOD attempt")
        for row in read_ledger_events(self.run_dir / "decision_ledger.jsonl"):
            if row.get("event") in {"ood_attempt_started", "ood_attempt_finished"}:
                raise CampaignError("cannot resume: the decision ledger already records an OOD attempt")
        self.ledger.append(
            "ood_phase_started",
            schema_version=CAMPAIGN_SCHEMA,
            seed=self.seed,
            lock_manifest_sha256=sha256_file(campaign_lock),
            attempt_id=self.attempt_id,
        )
        self.gpu_preflight()
        return self._ood_and_report(campaign_lock)

    def _ood_and_report(self, campaign_lock: Path) -> str:
        stages = self.plan["stages"]
        ood = self.one_shot_ood(stages["ood"], campaign_lock)
        self.ledger.append(
            "m1_gate_decision",
            decision=ood["decision"],
            ood_result=Path(stages["ood"]["result_path"]).relative_to(self.run_dir).as_posix(),
            ood_result_sha256=sha256_file(Path(stages["ood"]["result_path"])),
        )

        self._must_succeed("report", stages["report"])
        report_path = Path(stages["report"]["report_path"])
        validate_report(report_path)
        report_hash = sha256_file(report_path)
        self.ledger.append(
            "report_validated",
            report_path=report_path.relative_to(self.run_dir).as_posix(),
            report_sha256=report_hash,
            required_sections=len(REPORT_MARKERS),
            explicit_no_h100_statement=True,
        )
        stop_path = self.run_dir / "STOP.json"
        write_json_exclusive(
            stop_path,
            {
                "schema_version": CAMPAIGN_SCHEMA,
                "stopped_at_utc": utc_now(),
                "reason": "manual_review_required",
                "m1_decision": ood["decision"],
                "report_path": report_path.relative_to(self.run_dir).as_posix(),
                "report_sha256": report_hash,
                "gpu": self.gpu,
                "h100_launched": False,
                "next_stage_launched": False,
            },
        )
        self.ledger.append(
            "campaign_stopped",
            reason="manual_review_required",
            stop_marker=stop_path.relative_to(self.run_dir).as_posix(),
            stop_marker_sha256=sha256_file(stop_path),
            h100_launched=False,
            next_stage_launched=False,
        )
        return "manual_review_required"


def read_ledger_events(path: Path) -> list[dict[str, Any]]:
    """Read an existing decision ledger; used to refuse a second OOD attempt."""

    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"missing or unsafe decision ledger: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CampaignError(f"decision ledger line {number} is not JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise CampaignError(f"decision ledger line {number} is not an object")
        rows.append(row)
    return rows


def _load_plan(path: Path | None, seed: int = 0) -> dict[str, Any]:
    if path is None:
        return default_plan(seed)
    return load_json_object(path, "command plan")


def _default_run_dir(
    repo_root: Path,
    now: datetime | None = None,
    *,
    seed: int = 0,
    gpu_label: str | None = None,
) -> Path:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    label = gpu_label or GPU_LABEL
    return (
        repo_root
        / "data"
        / "results"
        / "metacog_alignment"
        / f"{stamp}_{MODEL_LABEL}_{label}_seed{seed}"
    )


def _write_ledger_sidecar(ledger: DecisionLedger, *, phase: str = "") -> Path:
    suffix = f".{phase}.sha256" if phase else ".sha256"
    sidecar = ledger.path.with_suffix(ledger.path.suffix + suffix)
    if sidecar.exists() or sidecar.is_symlink():
        raise CampaignError(f"refusing to overwrite ledger hash: {sidecar}")
    digest = sha256_file(ledger.path)
    with sidecar.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {ledger.path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-revision",
        required=True,
        help=f"immutable 40-hex Hugging Face commit for {EXPECTED_MODEL}",
    )
    parser.add_argument(
        "--tokenizer-revision",
        required=True,
        help="immutable 40-hex tokenizer commit (may equal the model revision)",
    )
    parser.add_argument("--run-dir", type=Path, help="fresh output directory; default includes UTC date")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        choices=ALLOWED_SEEDS,
        help="optimisation seed for this campaign run (split seed stays 0)",
    )
    parser.add_argument(
        "--stop-after",
        choices=("id_lock", "report"),
        default="report",
        help=(
            "'id_lock' stops with Decoupled/Compositional still sealed so every "
            "seed can be locked before any OOD attempt; resume later with --resume-ood"
        ),
    )
    parser.add_argument(
        "--resume-ood",
        action="store_true",
        help="run the one-shot OOD and report phases in an existing run directory that stopped at its ID lock",
    )
    parser.add_argument("--plan", type=Path, help="optional JSON command plan implementing the fixed contracts")
    parser.add_argument(
        "--unsafe-test-plan",
        action="store_true",
        help="allow --plan execution for synthetic contract tests; never use for a scientific run",
    )
    parser.add_argument("--nvidia-smi", default="nvidia-smi", help="nvidia-smi binary (test override)")
    parser.add_argument(
        "--gpu-index",
        help=(
            "physical A5000 index to reserve; if omitted, choose the idle eligible "
            "A5000 with the most free memory"
        ),
    )
    parser.add_argument(
        "--min-free-memory-mib",
        type=int,
        default=DEFAULT_MIN_FREE_MIB,
        help=f"minimum free A5000 memory (default: {DEFAULT_MIN_FREE_MIB})",
    )
    parser.add_argument(
        "--allow-existing-processes-under-mib",
        type=int,
        default=0,
        help=(
            "explicitly allow small resident GPU processes up to this total MiB "
            "on the selected card (default 0; maximum 512)"
        ),
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="render and print the plan without creating files or querying a GPU",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not PINNED_REVISION_RE.fullmatch(args.model_revision):
        parser.error("--model-revision must be an immutable 40-hex commit")
    if not PINNED_REVISION_RE.fullmatch(args.tokenizer_revision):
        parser.error("--tokenizer-revision must be an immutable 40-hex commit")
    if args.min_free_memory_mib < DEFAULT_MIN_FREE_MIB:
        parser.error(f"--min-free-memory-mib cannot be below {DEFAULT_MIN_FREE_MIB}")
    if not 0 <= args.allow_existing_processes_under_mib <= 512:
        parser.error("--allow-existing-processes-under-mib must be in [0, 512]")
    if args.gpu_index is not None and not args.gpu_index.isdigit():
        parser.error("--gpu-index must be a non-negative physical GPU index")
    if args.unsafe_test_plan and args.plan is None:
        parser.error("--unsafe-test-plan requires --plan")

    if args.resume_ood and args.stop_after != "report":
        parser.error("--resume-ood always runs through the report; drop --stop-after")

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = (args.run_dir or _default_run_dir(repo_root, seed=args.seed)).resolve()
    attempt_id = uuid.uuid4().hex
    context = {
        "python": sys.executable,
        "repo": str(repo_root),
        "run_dir": str(run_dir),
        "model_revision": args.model_revision.lower(),
        "tokenizer_revision": args.tokenizer_revision.lower(),
        "ood_attempt_id": attempt_id,
    }
    raw_plan = _load_plan(args.plan, args.seed)
    plan = _render(raw_plan, context)
    validate_plan(plan, run_dir)
    if args.print_plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.plan is not None and not args.unsafe_test_plan:
        parser.error(
            "custom plans are test-only; pass --unsafe-test-plan explicitly or use --print-plan"
        )

    if args.resume_ood:
        if run_dir.is_symlink() or not run_dir.is_dir():
            print(f"error: --resume-ood needs an existing campaign directory: {run_dir}", file=sys.stderr)
            return 1
    else:
        if run_dir.exists() or run_dir.is_symlink():
            print(f"error: refusing to overwrite existing campaign directory: {run_dir}", file=sys.stderr)
            return 1
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir()

    runner: CampaignRunner | None = None
    exit_code = 1
    try:
        runner = CampaignRunner(
            repo_root=repo_root,
            run_dir=run_dir,
            plan=plan,
            model_revision=args.model_revision.lower(),
            tokenizer_revision=args.tokenizer_revision.lower(),
            nvidia_smi=args.nvidia_smi,
            min_free_mib=args.min_free_memory_mib,
            requested_gpu_index=args.gpu_index,
            allowed_existing_process_mib=args.allow_existing_processes_under_mib,
            attempt_id=attempt_id,
            seed=args.seed,
            resume=args.resume_ood,
        )
        plan_name = "campaign_plan_ood.json" if args.resume_ood else "campaign_plan.json"
        plan_path = run_dir / plan_name
        write_json_exclusive(plan_path, plan)
        runner.ledger.append(
            "campaign_plan_locked",
            plan_path=plan_path.relative_to(run_dir).as_posix(),
            plan_sha256=sha256_file(plan_path),
            phase="ood" if args.resume_ood else "id",
        )
        if args.resume_ood:
            reason = runner.resume_ood()
        else:
            reason = runner.run(stop_after=args.stop_after)
        print(f"STOP: {reason}; no H100 job was launched. Artifacts: {run_dir}")
        exit_code = 0
    except ControlledStop as exc:
        if runner is not None:
            runner.ledger.append(
                "campaign_stopped",
                reason=str(exc),
                h100_launched=False,
                next_stage_launched=False,
            )
        print(f"STOP: {exc}. No H100 job was launched.", file=sys.stderr)
        exit_code = 3
    except (CampaignError, OSError) as exc:
        if runner is not None:
            runner.ledger.append(
                "campaign_failed_closed",
                error_type=type(exc).__name__,
                reason=str(exc),
                h100_launched=False,
                next_stage_launched=False,
            )
        print(f"error: {exc}\nSTOP: no H100 job was launched.", file=sys.stderr)
        exit_code = 1
    finally:
        if runner is not None:
            try:
                sidecar = _write_ledger_sidecar(
                    runner.ledger, phase="ood" if args.resume_ood else ""
                )
                print(f"decision ledger hash: {sidecar}")
            except CampaignError as exc:
                print(f"error: could not seal decision ledger: {exc}", file=sys.stderr)
                exit_code = 1
            finally:
                if runner.gpu_lock_handle is not None:
                    runner.gpu_lock_handle.close()
                    runner.gpu_lock_handle = None
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
