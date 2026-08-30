#!/usr/bin/env python3
"""Emit the preregistration manifest and environment record for RL-QA 32B."""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-32B"
REV = "9216db5781bf21249d130ec9da846c4624c16137"
OUT = REPO / "data/results/rlqa32_a100"

BATTERIES = ("battery_v1_final", "battery_v2_final", "battery_v2_g2",
             "battery_v3d", "battery_v4_final")
SOURCES = ("src/experiments/train_memory_rl.py", "src/experiments/preflight_qa_reward.py",
           "src/experiments/evaluate_memory_rl.py", "src/experiments/measure.py",
           "src/memory_rl/data.py", "src/memory_rl/modeling.py", "src/memory_rl/recall.py",
           "src/memory_rl/objectives.py", "src/analysis/memory_rl_gates.py",
           "scripts/lock_rlqa_checkpoints.py", "scripts/audit_admission_prompts.py",
           "scripts/report_rlqa32_seed0.py")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return None


def main() -> int:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    split = json.loads((OUT / "dryrun_s0/split_manifest.json").read_text())
    leak = json.loads((OUT / "preflight/prompt_leak_audit.json").read_text())
    chat_template = None
    cfg = Path("/rodata/azradonc_dev/m253405/cache/hub/models--Qwen--Qwen3-32B/snapshots") / REV
    tmpl = cfg / "tokenizer_config.json"
    if tmpl.is_file():
        data = json.loads(tmpl.read_text())
        if isinstance(data.get("chat_template"), str):
            chat_template = hashlib.sha256(data["chat_template"].encode()).hexdigest()

    prereg = {
        "schema_version": "rlqa-32b-preregistration/v1",
        "written_at_utc": now,
        "written_after": [
            "32B teacher measurements (results_v{1f,2f,2g2}_qwen3-32B.json)",
            "data-only dry run",
            "prompt leak audit",
            "Stage B0 preflight launch",
        ],
        "written_before": ["recipe freeze", "canary", "formal seed-0 training",
                           "ID lock", "one-shot OOD"],
        "honesty_note": (
            "This manifest was written after the ID-only measurement and preflight "
            "stages had started, not before them. Every artifact those stages "
            "produced is hashed here, so the record is complete even though the "
            "ordering deviates from the plan's 'before any preflight' wording. "
            "Nothing sealed was touched at any point."
        ),
        "model": {"repo": MODEL, "revision": REV, "tokenizer_revision": REV,
                  "chat_template_sha256": chat_template,
                  "enable_thinking": "template default for V/W and training; "
                                     "False only for full-context QA generation, "
                                     "byte-for-byte as in the completed 8B campaign"},
        "authorised_scope": {
            "method": "rl-qa", "seeds": [0], "budget": 2,
            "seed_expansion": "requires a human release note; not authorised here",
            "forbidden": ["rl-w", "rl-hybrid", "PPO/DPO", "new reward", "new budget",
                          "other model families", "70B", "full fine-tuning",
                          "OOD-driven coefficient search"],
        },
        "recipe": {"lambda_qa": 1.0, "lambda_w": 0.0, "group_size": 8, "grpo_epochs": 2,
                   "lora_rank": 32, "learning_rate": 1e-6, "beta": 0.03, "max_steps": 300,
                   "temperature": "5.0, locked only if Stage B0 passes; searching is forbidden",
                   "max_length": 2048, "answer_tokens": 64, "dtype": "bfloat16",
                   "eval_steps": [0, 100, 200, 300],
                   "selection_scope": "id_validation_only",
                   "selection_metric": "frozen_recall_QA",
                   "selection_rule": "strict_first_maximum",
                   "selection_tie_break": "earliest_step"},
        "data": {"split_seed": 0, "train_episodes": 175, "validation_episodes": 45,
                 "split_manifest_sha256": split["manifest_sha256"],
                 "training_sources": ["explicit", "evoked", "evoked_g2"],
                 "sealed_until_lock": ["decoupled", "compositional"],
                 "admission_prompt_audit": {
                     "decision": leak["decision"], "counts": leak["counts"],
                     "aggregate_prompt_sha256": leak["aggregate_prompt_sha256"],
                     "template_sha256": leak["admission_prompt_template_sha256"]}},
        "statistics": {"bootstrap_draws": 4000, "bootstrap_seed": 0,
                       "unit": "episode_cluster", "mcnemar": "exact two-sided",
                       "pooling": "seed x episode pooling is forbidden"},
        "ood": {"attempt_limit_per_seed": 1,
                "conditions": ["decoupled", "compositional"],
                "primary": "decoupled"},
        "gate": {"pass_min_qa_delta_pp": 5.0, "pass_requires_auc_ci_above_zero": True,
                 "max_full_context_qa_drop_pp": 2.0, "max_w_rr_drop": 0.03},
        "battery_sha256": {b: sha(REPO / "data/benchmarks" / f"{b}.json") for b in BATTERIES},
        "teacher_sha256": {f"results_{t}_qwen3-32B.json":
                           sha(REPO / "data/results" / f"results_{t}_qwen3-32B.json")
                           for t in ("v1f", "v2f", "v2g2")},
        "source_sha256": {s: sha(REPO / s) for s in SOURCES if (REPO / s).is_file()},
    }

    env = {
        "schema_version": "rlqa-32b-environment/v1",
        "collected_at_utc": now,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {p: (md.version(p) if _has(p) else None) for p in
                     ("torch", "transformers", "peft", "accelerate", "numpy",
                      "scikit-learn", "scipy", "datasets")},
        "git_commit": run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "gpu_inventory": run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                              "--format=csv,noheader"]),
        "slurm_partitions": run(["sinfo", "-o", "%P %D %G %c %m %l"]),
    }

    for name, payload in (("preregistration_manifest.json", prereg),
                          ("environment_and_hashes.json", env)):
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(f"exists, keeping: {path}")
            continue
        with path.open("x", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote {path}")
    return 0


def _has(pkg: str) -> bool:
    try:
        md.version(pkg)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
