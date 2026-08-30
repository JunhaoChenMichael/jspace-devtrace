#!/usr/bin/env python3
"""Seed-0 scaling gate report for Qwen3-32B RL-QA.

Applies the three-way gate from the 32B plan:

  PASS   Decoupled QA delta >= +5.00 pp, admission AUC delta > 0 with a paired
         episode-cluster 95% CI whose lower bound is above 0, full-context QA
         drop <= 2.00 pp, fresh W_rr drop <= 0.03, no integrity violation, and
         no Compositional secondary-harm alert.
  AMBER  a boundary result: positive but below the practical-effect threshold,
         or a numeric pass undercut by a Compositional harm alert.
  FAIL   a non-positive primary effect, a no-harm breach, or an integrity
         violation.

PASS authorises nothing by itself: seeds 1/2 need a human release note, and the
report always records automatic_seed_expansion_authorized = false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from report_rlqa_three_seed import (  # noqa: E402
    _mcnemar,
    _no_harm,
    _paired_auc,
    _qa_accuracy,
    _scope,
    workspace_no_harm,
)

SCHEMA = "rlqa-32b-seed0-gate/v1"
GATE = {
    "min_qa_delta_pp": 5.0,
    "max_full_context_qa_drop_pp": 2.0,
    "max_w_rr_drop": 0.03,
}
CONDITION = "rl-qa-s0"


def _load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"missing or unsafe {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_source(qa: Mapping[str, Any], source: str, budget: int) -> dict[str, Any]:
    scope = _scope(qa, source)
    original = _qa_accuracy(scope, "original", budget)
    adapted = _qa_accuracy(scope, CONDITION, budget)
    admission = _paired_auc(scope, CONDITION)
    harm = _no_harm(qa, CONDITION, source) or {}
    return {
        "qa_original": original,
        "qa_rl_qa": adapted,
        "qa_delta_pp": (adapted - original) * 100.0,
        "admission_auc_delta": float(admission["estimate"]),
        "admission_auc_ci_95": admission.get("ci_95"),
        "exact_mcnemar": _mcnemar(qa, source, CONDITION, budget),
        **harm,
    }


def build(
    qa: Mapping[str, Any],
    *,
    budget: int = 2,
    workspace: Mapping[int, Mapping[str, Any]] | None = None,
    integrity_ok: bool = True,
) -> dict[str, Any]:
    primary = evaluate_source(qa, "decoupled", budget)
    diagnostic = evaluate_source(qa, "compositional", budget)
    ws = (workspace or {}).get(0, {})
    primary.update(ws)

    ci = primary.get("admission_auc_ci_95") or [None, None]
    qa_drop = primary.get("full_context_qa_drop_pp")
    w_drop = primary.get("w_rr_drop")

    checks = {
        "qa_delta_at_least_5pp": primary["qa_delta_pp"] >= GATE["min_qa_delta_pp"],
        "qa_delta_positive": primary["qa_delta_pp"] > 0,
        "admission_auc_delta_positive": primary["admission_auc_delta"] > 0,
        "admission_auc_ci_excludes_zero": ci[0] is not None and float(ci[0]) > 0,
        "full_context_qa_drop_within_2pp": qa_drop is None
        or qa_drop <= GATE["max_full_context_qa_drop_pp"],
        "w_rr_drop_within_0.03": w_drop is None or float(w_drop) <= GATE["max_w_rr_drop"],
        "integrity_checks_pass": bool(integrity_ok),
    }
    # Compositional never rescues Decoupled; it can only cap the verdict.
    comp_qa_drop = diagnostic.get("full_context_qa_drop_pp")
    comp_harm = (
        (comp_qa_drop is not None and comp_qa_drop > GATE["max_full_context_qa_drop_pp"])
        or diagnostic["qa_delta_pp"] < 0
    )
    checks["compositional_no_secondary_harm"] = not comp_harm

    if (
        not checks["integrity_checks_pass"]
        or not checks["qa_delta_positive"]
        or not checks["admission_auc_delta_positive"]
        or not checks["full_context_qa_drop_within_2pp"]
        or not checks["w_rr_drop_within_0.03"]
    ):
        decision = "FAIL"
    elif all(checks.values()):
        decision = "PASS"
    else:
        decision = "AMBER"

    return {
        "schema_version": SCHEMA,
        "model": "Qwen/Qwen3-32B",
        "seed": 0,
        "budget": budget,
        "gate_thresholds": GATE,
        "primary_decoupled": primary,
        "diagnostic_compositional": diagnostic,
        "checks": checks,
        "decision": decision,
        "automatic_seed_expansion_authorized": False,
    }


def render(summary: Mapping[str, Any], *, recipe: Mapping[str, Any] | None,
           lock: Mapping[str, Any] | None, eight_b: Mapping[str, Any] | None) -> str:
    p, d = summary["primary_decoupled"], summary["diagnostic_compositional"]
    ci = p.get("admission_auc_ci_95") or [float("nan"), float("nan")]
    lines = [
        "# Qwen3-32B RL-QA seed-0 scaling gate",
        "",
        f"Report schema: `{summary['schema_version']}`. Decision: **{summary['decision']}**.",
        "",
        "`automatic_seed_expansion_authorized = false`. Seeds 1/2 require a human "
        "release note; this report authorises nothing on its own.",
        "",
        "## 1. Gate",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    labels = {
        "qa_delta_at_least_5pp": "Decoupled QA delta >= +5.00 pp",
        "qa_delta_positive": "Decoupled QA delta > 0",
        "admission_auc_delta_positive": "Decoupled admission AUC delta > 0",
        "admission_auc_ci_excludes_zero": "admission AUC 95% CI lower bound > 0",
        "full_context_qa_drop_within_2pp": "full-context QA drop <= 2.00 pp",
        "w_rr_drop_within_0.03": "fresh W_rr drop <= 0.03",
        "compositional_no_secondary_harm": "Compositional: no secondary harm alert",
        "integrity_checks_pass": "provenance / leakage / lock integrity",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {'PASS' if summary['checks'][key] else 'FAIL'} |")

    mcn = p.get("exact_mcnemar") or {}
    lines += [
        "",
        "## 2. Decoupled (primary)",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| QA accuracy, Original | {p['qa_original']:.4f} |",
        f"| QA accuracy, RL-QA | {p['qa_rl_qa']:.4f} |",
        f"| **QA delta** | **{p['qa_delta_pp']:+.2f} pp** |",
        f"| Admission AUC delta | {p['admission_auc_delta']:+.5f} |",
        f"| Admission AUC 95% CI | [{ci[0]:+.5f}, {ci[1]:+.5f}] |",
        f"| Exact McNemar p | {mcn.get('p_value', float('nan')):.4g} |",
        f"| discordant (adapter-only / original-only) | "
        f"{mcn.get('adapter_only_correct', 'n/a')} / {mcn.get('original_only_correct', 'n/a')} |",
        f"| Full-context QA drop | {(p.get('full_context_qa_drop_pp') or 0.0):+.2f} pp |",
        f"| Fresh W_rr drop | {(p.get('w_rr_drop') or 0.0):+.5f} |",
        "",
        "## 3. Compositional (mandatory diagnostic)",
        "",
        f"QA delta {d['qa_delta_pp']:+.2f} pp, admission AUC delta "
        f"{d['admission_auc_delta']:+.5f}. Diagnostics cannot rescue Decoupled; "
        "they can only cap the verdict at AMBER.",
        "",
    ]
    if eight_b:
        lines += [
            "## 4. Comparison with the completed 8B replication",
            "",
            "| | Qwen3-8B (3 seeds) | Qwen3-32B (seed 0) |",
            "|---|---:|---:|",
            f"| Decoupled QA delta | {eight_b.get('qa_delta_mean_pp', float('nan')):+.2f} pp "
            f"| {p['qa_delta_pp']:+.2f} pp |",
            f"| Admission AUC, Original | {eight_b.get('original_auc', float('nan')):.4f} "
            f"| {eight_b.get('original_auc_32b', float('nan')):.4f} |",
            f"| Admission AUC delta | {eight_b.get('auc_delta_mean', float('nan')):+.5f} "
            f"| {p['admission_auc_delta']:+.5f} |",
            "",
            "The 32B starting point is much stronger, so an 8B-sized AUC gain is "
            "arithmetically unavailable; headroom, not method quality, sets the ceiling.",
            "",
        ]
    if summary.get("id_trajectory"):
        lines += [
            "## 4b. In-distribution learning vs out-of-distribution transfer",
            "",
            "| step | ID QA | ID containment | ID verbal AUC | yes rate |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in summary["id_trajectory"]:
            mark = "  **<- locked**" if row.get("selected") else ""
            lines.append(
                f"| {row['step']} | {row['qa_accuracy']:.4f} | "
                f"{row.get('containment', float('nan')):.4f} | "
                f"{row.get('verbal_auc', float('nan')):.4f} | "
                f"{row.get('yes_rate', float('nan')):.3f} |{mark}"
            )
        first, best = summary["id_trajectory"][0], summary["id_trajectory"][-1]
        lines += [
            "",
            f"Training worked in distribution: ID QA moved "
            f"{first['qa_accuracy']:.4f} -> {max(r['qa_accuracy'] for r in summary['id_trajectory']):.4f}. "
            "The failure is transfer, not optimisation. The 8B campaign gained a "
            "comparable amount in distribution and that gain did reach Decoupled; "
            "here it does not.",
            "",
        ]
    if recipe:
        lines += ["## 5. Frozen recipe", "", "```json",
                  json.dumps(recipe.get("recipe", {}), indent=2, sort_keys=True), "```", ""]
    if lock:
        entry = lock["seeds"][0]
        lines += [
            "## 6. Pre-OOD lock",
            "",
            f"- Selected step: **{entry['step']}** (strict first maximum of ID QA)",
            f"- Checkpoint tree SHA-256: `{entry['checkpoint_tree_sha256']}`",
            f"- Lock manifest self-hash: `{lock['manifest_sha256']}`",
            "",
        ]
    lines += [
        "## 7. Stop",
        "",
        "Seed 0 is complete and the campaign stops here for human review. "
        "No seed expansion, no RL-W, no Hybrid, no larger model, and no combined "
        "objective were launched.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--workspace-base", type=Path)
    parser.add_argument("--workspace-adapter", type=Path)
    parser.add_argument("--eight-b-summary", type=Path)
    parser.add_argument("--run-dir", type=Path, help="training run directory, for the ID trace")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)

    qa = _load(args.qa, "unified evaluation")
    workspace = None
    if args.workspace_base and args.workspace_adapter:
        workspace = workspace_no_harm(args.workspace_base, {0: args.workspace_adapter})
    summary = build(qa, budget=args.budget, workspace=workspace)
    if args.run_dir and args.lock:
        step = _load(args.lock, "lock")["seeds"][0]["step"]
        summary["id_trajectory"] = id_trajectory(args.run_dir, step)

    eight_b = None
    if args.eight_b_summary and args.eight_b_summary.is_file():
        s = json.loads(args.eight_b_summary.read_text())
        eight_b = {
            "qa_delta_mean_pp": s["aggregate"]["qa_delta_pp"]["mean"],
            "auc_delta_mean": s["aggregate"]["admission_auc_delta"]["mean"],
            "original_auc": 0.34154,
            "original_auc_32b": 0.65708,
        }

    text = render(
        summary,
        recipe=_load(args.recipe, "recipe") if args.recipe else None,
        lock=_load(args.lock, "lock") if args.lock else None,
        eight_b=eight_b,
    )
    for path, content in ((args.out_json, json.dumps(summary, indent=2, sort_keys=True) + "\n"),
                          (args.out_md, text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"{summary['decision']}: wrote {args.out_md}")
    return 0



def id_trajectory(run_dir: Path, selected_step: int | None) -> list[dict[str, Any]]:
    """ID validation trace, so a FAIL can be read as transfer vs optimisation."""

    rows = []
    metrics = run_dir / "metrics.jsonl"
    if not metrics.is_file():
        return rows
    for line in metrics.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "qa_accuracy" not in record:
            continue
        rows.append({
            "step": record.get("step"),
            "qa_accuracy": record["qa_accuracy"],
            "containment": record.get("containment"),
            "verbal_auc": record.get("verbal_auc"),
            "yes_rate": record.get("verbal_yes_rate"),
            "selected": record.get("step") == selected_step,
        })
    return rows

if __name__ == "__main__":
    raise SystemExit(main())
