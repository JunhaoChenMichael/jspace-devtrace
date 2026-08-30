#!/usr/bin/env python3
"""Collect the environment/hardware/data provenance both A100 handoffs require."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
BATTERIES = (
    "battery_v1_final.json",
    "battery_v2_final.json",
    "battery_v2_g2.json",
    "battery_v3d.json",
    "battery_v4_final.json",
)
SOURCES = (
    "src/experiments/measure.py",
    "src/experiments/train_metacog_m1.py",
    "src/experiments/train_memory_rl.py",
    "src/experiments/preflight_qa_reward.py",
    "src/experiments/evaluate_memory_rl.py",
    "src/analysis/evaluate_metacog_m1_ood.py",
    "src/analysis/gate_metacog_m0.py",
    "src/analysis/report_metacog_m1.py",
    "src/analysis/memory_rl_gates.py",
    "src/jlens.py",
    "src/memory_rl/recall.py",
    "scripts/run_metacog_alignment_campaign.py",
    "scripts/lock_rlqa_checkpoints.py",
    "scripts/report_metacog_three_seed.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str]) -> str | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "scikit-learn", "scipy", "datasets"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    payload = {
        "collected_on_host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "packages": packages,
        "expected_gpu": os.environ.get("METACOG_EXPECTED_GPU"),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                           "--format=csv,noheader"]),
        "slurm_partitions": run(["sinfo", "-o", "%P %D %G %c %m %l"]),
        "git_commit": run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_dirty": (run(["git", "-C", str(REPO), "status", "--porcelain"]) or "").splitlines(),
        "model": {
            "repo_id": "Qwen/Qwen3-8B",
            "revision": MODEL_REVISION,
            "cache_snapshot": str(
                Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
                / "hub/models--Qwen--Qwen3-8B/snapshots" / MODEL_REVISION
            ),
        },
        "battery_sha256": {
            name: sha256(REPO / "data" / "benchmarks" / name) for name in BATTERIES
        },
        "source_sha256": {
            name: sha256(REPO / name) for name in SOURCES if (REPO / name).is_file()
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
