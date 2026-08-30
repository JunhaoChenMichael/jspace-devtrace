#!/usr/bin/env python3
"""Seed-0 report for the Qwen3-32B Binary Metacognitive Alignment scaling gate.

The campaign stopped at M0: the plan gates a new scale point on the existence
of a repairable reporting gap (Decoupled W_before - V_before >= 0.10), and
Qwen3-32B does not have one. No training was run and no OOD battery was opened.

The report pairs that decision with the five-size measurement sweep that
explains it, so the stop is legible as a result rather than as a failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "metacog-32b-scale-boundary/v1"
CONDITIONS = ("explicit", "evoked", "decoupled", "compositional")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render(gate: Mapping[str, Any], trend: Mapping[str, Any] | None,
           provenance: Mapping[str, Any] | None, run_dir: str,
           eight_b: Mapping[str, Any] | None) -> str:
    g = gate["gate"]
    lines = [
        "# Qwen3-32B Binary Metacognitive Alignment: seed-0 scaling gate",
        "",
        f"Report schema: `{SCHEMA}`. M0 decision: **{gate['decision']}**.",
        "",
        "The plan gates a new scale point on a repairable reporting gap, not on "
        "reproducing the 8B numbers. Qwen3-32B does not have one, so training was "
        "forbidden and no OOD battery was opened. `automatic_seed_expansion_authorized = false`.",
        "",
        "## 1. The gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Decoupled V (before) | {g['observed']['V']:.5f} |",
        f"| Decoupled W_rr (before) | {g['observed']['W_rr']:.5f} |",
        f"| **Reporting gap (W - V)** | **{g['reporting_gap']:+.5f}** |",
        f"| Required gap | {g['min_reporting_gap']:.2f} |",
        f"| Decision | {gate['decision']} |",
        "",
        "The historical 8B/paper values (V 0.337, W_rr 0.654) appear in the gate "
        "record as context only; a different model has no obligation to match them.",
        "",
        "## 2. M0 measurement, all four conditions",
        "",
        "| Condition | V | W_rr | gap |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        block = gate["conditions"][condition]["pooled_auc"]
        lines.append(
            f"| {condition} | {block['V']:.4f} | {block['W_rr']:.4f} | "
            f"{block['W_rr'] - block['V']:+.4f} |"
        )
    if eight_b:
        lines += [
            "",
            "Against the completed 8B campaign, on the same 335 Decoupled candidates:",
            "",
            "| | Qwen3-8B | Qwen3-32B |",
            "|---|---:|---:|",
            f"| Decoupled V | {eight_b['V']:.4f} | {g['observed']['V']:.4f} |",
            f"| Decoupled W_rr | {eight_b['W_rr']:.4f} | {g['observed']['W_rr']:.4f} |",
            f"| gap | {eight_b['W_rr'] - eight_b['V']:+.4f} | {g['reporting_gap']:+.4f} |",
            "",
            "At 8B the verbal report is *below chance*: it is anti-correlated with "
            "utility, not merely uninformative. At 32B it is well above chance while "
            "the workspace readout barely moves.",
            "",
        ]
    if trend:
        lines += [
            "## 3. Why: the reporter jumps between 14B and 32B",
            "",
            "A five-size measurement sweep (1.7B / 4B / 8B / 14B / 32B, no training) "
            "locates the transition. Adjacent-scale deltas use shared episode draws, "
            "so the difference is paired.",
            "",
            "| Condition | 14B->32B delta V (chat) | 95% CI | 14B->32B delta V_raw | 95% CI |",
            "|---|---:|---|---:|---|",
        ]
        for condition in ("evoked", "decoupled", "compositional"):
            steps = trend["trends"][condition]["adjacent_scale_paired_deltas"]
            last = steps[-1] if steps else None
            if not last:
                continue
            v, raw = last.get("V"), last.get("V_raw")
            lines.append(
                f"| {condition} | {v['delta']:+.4f} | "
                f"[{v['ci_95'][0]:+.4f}, {v['ci_95'][1]:+.4f}] | "
                f"{raw['delta']:+.4f} | [{raw['ci_95'][0]:+.4f}, {raw['ci_95'][1]:+.4f}] |"
            )
        lines += [
            "",
            "Every other adjacent step is small and mostly not distinguishable from "
            "zero, so this is a transition between 14B and 32B rather than a smooth "
            "trend. The jump is several times larger in the chat-template channel "
            "than in the template-free `V_raw` channel, which places it in the "
            "instruct pathway rather than in the underlying next-token computation.",
            "",
            "## 4. Why the small models score below chance",
            "",
            "| Condition | load-bearing literally in context | negatives | literal-mention AUC |",
            "|---|---:|---:|---:|",
        ]
        for condition in CONDITIONS:
            s = trend["trends"][condition]["surface_baseline"]
            lines.append(
                f"| {condition} | {s['load_bearing_literally_present']:.0%} | "
                f"{s['negatives_literally_present']:.0%} | {s['literal_presence_auc']:.4f} |"
            )
        lines += [
            "",
            "The benchmark makes the load-bearing concept an unstated bridge, so "
            "'does this word appear in the passage' is an almost perfect ANTI-predictor "
            "of utility on the non-Explicit conditions. A model answering from surface "
            "prominence therefore scores below chance, which is what the small models do. "
            "On Explicit, where the feature is uninformative, V sits near 0.5 at every "
            "size - the control that rules out a pipeline artefact.",
            "",
            "This is an association, not a demonstrated mechanism: it shows where the "
            "small models' scores land relative to a surface baseline, not that they "
            "compute that feature.",
            "",
        ]
    lines += [
        "## 5. What this means for the claim",
        "",
        "The 8B dissociation survives: the workspace readout tracks utility while the "
        "verbal report does not. What the sweep adds is a boundary. The deficit that "
        "Binary Metacognitive Alignment repairs is not a fixed property of the "
        "architecture; it disappears between 14B and 32B without any intervention. "
        "Trained 8B reporters reach V about 0.57-0.69 on Decoupled; untrained 32B is "
        "0.657 - the intervention buys roughly what this scale step gives for free.",
        "",
        "That favours reading the effect as a capability limit in the verbal channel "
        "rather than a persistent access barrier, though the present evidence does not "
        "settle it: the sizes differ in post-training as well as in parameters.",
        "",
        "## 6. Artifacts",
        "",
        f"- Run directory: `{run_dir}`",
        f"- M0 gate: `{run_dir}/m0/gate.json`, `{run_dir}/m0/gate.md`",
        f"- Decision ledger: `{run_dir}/decision_ledger.jsonl`",
        "- Scale sweep: `data/results/scale_sweep/`, analysis in `SCALE_TREND.md`",
        "",
        "## 7. Stop",
        "",
        "No training, no adapter, no OOD attempt was consumed. Reopening this scale "
        "point requires a pre-registration amendment, not a rerun: the plan forbids "
        "manufacturing a gap by changing prompts, thinking mode, token scoring or the "
        "workspace readout.",
        "",
    ]
    if provenance:
        lines += [
            "## 8. Environment",
            "",
            f"- Repository commit: `{provenance.get('git_commit')}`",
            "- Packages: "
            + ", ".join(f"{k} {v}" for k, v in sorted((provenance.get("packages") or {}).items()) if v),
            "",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--trend", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--eight-b-gate", type=Path)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)

    gate = _load(args.gate)
    eight_b = None
    if args.eight_b_gate and args.eight_b_gate.is_file():
        pooled = _load(args.eight_b_gate)["conditions"]["decoupled"]["pooled_auc"]
        eight_b = {"V": pooled["V"], "W_rr": pooled["W_rr"]}
    text = render(
        gate,
        _load(args.trend) if args.trend and args.trend.is_file() else None,
        _load(args.provenance) if args.provenance and args.provenance.is_file() else None,
        args.run_dir,
        eight_b,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(text, encoding="utf-8")
    print(f"{gate['decision']}: wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
