#!/usr/bin/env python3
"""Phase 0: classify every consumer of the verbal score, and say what can be re-run here.

The recovery plan gates all further work on this inventory. It records not only
which components the defect touched, but which of them have their inputs present
in this repository at all -- several historical campaigns are deliberately not
tracked in git, so their corrections cannot be produced on this machine.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/results/a100_next_boundary_campaign/reports"

CLASSES = (
    "UNAFFECTED", "RESCORE_ONLY", "RESELECT_AND_REGENERATE_QA",
    "RELOCK_EXISTING_CHECKPOINTS", "RETRAIN_REQUIRED",
    "INPUTS_ABSENT_FROM_THIS_REPOSITORY", "NOT_PRESENT", "DONE",
)

# component -> (classification, why, evidence path or None)
COMPONENTS = {
    "measure.py verbal_salience / verbal_salience_raw": (
        "DONE", "the defect itself; corrected to a log-space ratio with no guard epsilon",
        "src/experiments/measure.py"),
    "locomo_gate / longmemeval_gate / vprobe_robust / measure_vlm verbal scores": (
        "DONE", "same defect, corrected through the shared helper", None),
    "vrating_baseline 1-10 rating": (
        "DONE", "same defect class: the digit mass sat under its own guard; renormalised in log space",
        "src/experiments/vrating_baseline.py"),
    "W_rr workspace readout": (
        "UNAFFECTED", "reciprocal rank over layers; no epsilon anywhere in the path",
        "src/experiments/measure.py"),
    "frozen teacher labels": (
        "UNAFFECTED", "derived from W_rr, never from the verbal score", None),
    "RL admission policy scoring": (
        "UNAFFECTED", "binary_action_logits normalises in log space over two actions",
        "src/memory_rl/modeling.py"),
    "metacognitive ID checkpoint selection": (
        "UNAFFECTED", "train_metacog_m1.py scores through binary_action_logits, so the "
                      "locked checkpoints were selected on an unaffected quantity",
        "src/experiments/train_metacog_m1.py"),
    "Qwen3-8B metacognitive three-seed report": (
        "DONE", "re-measured from the locked adapters: mean delta V +0.273 -> +0.213, all seeds still pass",
        "data/results/a100_next_boundary_campaign/qwen3_8b_metacog_v3/reports/CORRECTED_8B_METACOG_THREE_SEED_REPORT.md"),
    "Qwen3-32B metacognitive M0 gate and seed 0": (
        "DONE", "gate reversed to MISALIGNMENT_REGIME; the re-run trained and returned AMBER",
        "data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0"),
    "Qwen3 scale sweep": (
        "DONE", "re-measured under v3 across five dense sizes plus the sparse diagnostic",
        "data/results/a100_next_boundary_campaign/shared/CORRECTED_SCALE_SWEEP_REPORT.md"),
    "Qwen3-8B and Qwen3-32B RL-QA Original arms": (
        "DONE", "Original arm re-scored and its budget-2 sets regenerated; 8B survives, "
                "32B reclassified from FAIL to ADMISSION_POSITIVE_QA_UNRESOLVED",
        "data/results/a100_next_boundary_campaign/qwen3_rlqa_v3"),
    "cross-family predictive grid (Qwen2.5, OLMo-2, Mistral, GPT-2)": (
        "RESCORE_ONLY", "measurement only, and re-runnable from scratch: the batteries are "
                        "tracked, the checkpoints are public. The ORIGINAL result files are "
                        "not in this repository, so this is a fresh measurement rather than a "
                        "correction of a stored one", None),
    "Qwen3-14B Binary metacognitive seed 0": (
        "RETRAIN_REQUIRED", "never run; the corrected gate decides whether it may train", None),
    "verbal-gated downstream policies (binary-verbal gating, top-k containment, "
    "mixed-pool, case studies, naturalistic streams)": (
        "INPUTS_ABSENT_FROM_THIS_REPOSITORY",
        "requires the original downstream run artifacts, which are project outputs excluded "
        "from git by repository policy", None),
    "Qwen2.5-7B RL-QA Original arm": (
        "INPUTS_ABSENT_FROM_THIS_REPOSITORY",
        "the completed Qwen2.5-7B campaign directory is not tracked here", None),
    "metacognitive objective study (Binary / Soft / Pairwise / Listwise)": (
        "NOT_PRESENT", "no such study exists in this repository; mixed_pool.py is a different "
                       "analysis and does not use the verbal probe for winner classification", None),
    "paper tables and figures": (
        "INPUTS_ABSENT_FROM_THIS_REPOSITORY",
        "paper/ is a separate git repository and is not checked out here", None),
    "multimodal verbal probing": (
        "RESCORE_ONLY", "measure_vlm is corrected, but no stored VLM result files exist here "
                        "to compare against; a fresh measurement is possible", None),
}


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    consumers = subprocess.run(
        ["grep", "-rl", "-e", 'row\\["V"\\]', "-e", "verbal_salience", "-e", "v_ref",
         "--include=*.py", "src", "scripts"],
        cwd=REPO, capture_output=True, text=True).stdout.split()

    entries = []
    for name, (klass, why, evidence) in COMPONENTS.items():
        assert klass in CLASSES, klass
        path = REPO / evidence if evidence else None
        entries.append({
            "component": name, "classification": klass, "reason": why,
            "evidence": evidence,
            "evidence_sha256": sha(path) if path and path.is_file() else None,
            "evidence_exists": bool(path and path.exists()),
        })

    counts: dict[str, int] = {}
    for e in entries:
        counts[e["classification"]] = counts.get(e["classification"], 0) + 1

    payload = {
        "schema_version": "v3-impact-inventory/v1",
        "verbal_score_schema_required": "workspace_measurement_metadata.v3",
        "source_files_referencing_the_verbal_score": sorted(consumers),
        "counts": counts,
        "components": entries,
        "validation": {
            "every_component_classified": all(e["classification"] in CLASSES for e in entries),
            "no_unknown_left": all(e["classification"] != "UNKNOWN" for e in entries),
            "blocked_components": [e["component"] for e in entries
                                   if e["classification"] in
                                   ("INPUTS_ABSENT_FROM_THIS_REPOSITORY", "NOT_PRESENT")],
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "V3_IMPACT_INVENTORY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")

    L = ["# v3 impact inventory", "",
         "Every consumer of the verbal score, classified. The recovery plan gates further "
         "work on this document.", "",
         "| Component | Classification | Why |", "|---|---|---|"]
    for e in entries:
        L.append(f"| {e['component']} | `{e['classification']}` | {e['reason']} |")
    L += ["", "## Counts", ""] + [f"- `{k}`: {v}" for k, v in sorted(counts.items())]
    L += ["", "## What cannot be produced on this machine", "",
          "Several components are blocked not by the defect but by absent inputs. This "
          "repository deliberately excludes trained artifacts and completed campaign "
          "directories, and `paper/` is a separate repository. The following need those "
          "inputs restored from the originating server before they can be corrected:", ""]
    L += [f"- {c}" for c in payload["validation"]["blocked_components"]]
    L += ["", "Nothing here is a judgement that those results are fine. They are unverified.", ""]
    (OUT_DIR / "V3_IMPACT_INVENTORY.md").write_text("\n".join(L))
    print(f"wrote inventory: {json.dumps(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
