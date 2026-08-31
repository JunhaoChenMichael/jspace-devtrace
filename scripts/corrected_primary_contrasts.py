#!/usr/bin/env python3
"""Phase 2: paired W-V contrasts, scale-gap intervals, and chat-versus-raw contrasts.

Everything is paired at the episode level and resampled as whole episodes, so a
candidate is never treated as an independent unit of uncertainty. Contrasts that
compare two quantities on the same candidates share their draws.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ("explicit", "evoked", "decoupled", "compositional")
DRAWS = 4000


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    pairs = sorted(zip(scores, labels))
    if not pairs:
        return None
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(l for _, l in pairs)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    return (sum(r for r, (_, l) in zip(ranks, pairs) if l) - pos * (pos + 1) / 2) / (pos * neg)


def by_episode(rows: Sequence[Mapping[str, Any]]) -> dict[Any, list[Mapping[str, Any]]]:
    out: dict[Any, list[Mapping[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["episode"], []).append(r)
    return out


def paired_contrast(rows, key_a, key_b, *, draws=DRAWS, seed=0) -> dict[str, Any] | None:
    """CI on AUC(a) - AUC(b) over the same candidates, sharing episode draws."""
    if not rows or key_a not in rows[0] or key_b not in rows[0]:
        return None
    groups = by_episode(rows)
    episodes = sorted(groups)

    def diff(sample):
        picked = [r for e in sample for r in groups[e]]
        labels = [int(r["label"] == "load_bearing") for r in picked]
        a = auc([float(r[key_a]) for r in picked], labels)
        b = auc([float(r[key_b]) for r in picked], labels)
        return None if a is None or b is None else a - b

    point = diff(episodes)
    if point is None:
        return None
    rng = random.Random(seed)
    est = []
    for _ in range(draws):
        s = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        d = diff(s)
        if d is not None:
            est.append(d)
    est.sort()
    lo, hi = est[int(0.025 * (len(est) - 1))], est[int(0.975 * (len(est) - 1))]
    return {"estimate": point, "ci_95": [lo, hi], "excludes_zero": lo > 0 or hi < 0,
            "draws_effective": len(est)}


def cross_model_gap_delta(rows_a, rows_b, *, draws=DRAWS, seed=0) -> dict[str, Any] | None:
    """CI on gap(b) - gap(a) where gap = AUC(W) - AUC(V), sharing episode draws.

    Two model sizes see identical episodes, so the draw is shared and the
    difference is paired; independent intervals would overstate the uncertainty.
    """
    ga, gb = by_episode(rows_a), by_episode(rows_b)
    episodes = sorted(set(ga) & set(gb))
    if not episodes:
        return None

    def gap(groups, sample):
        picked = [r for e in sample for r in groups[e]]
        labels = [int(r["label"] == "load_bearing") for r in picked]
        w = auc([float(r["W_rr"]) for r in picked], labels)
        v = auc([float(r["V"]) for r in picked], labels)
        return None if w is None or v is None else w - v

    point = None
    pa, pb = gap(ga, episodes), gap(gb, episodes)
    if pa is not None and pb is not None:
        point = pb - pa
    if point is None:
        return None
    rng = random.Random(seed)
    est = []
    for _ in range(draws):
        s = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        x, y = gap(ga, s), gap(gb, s)
        if x is not None and y is not None:
            est.append(y - x)
    est.sort()
    lo, hi = est[int(0.025 * (len(est) - 1))], est[int(0.975 * (len(est) - 1))]
    return {"estimate": point, "ci_95": [lo, hi], "excludes_zero": lo > 0 or hi < 0}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid-root", type=Path,
                    default=REPO / "data/results/a100_next_boundary_campaign/predictive_grid_v3")
    ap.add_argument("--sweep-root", type=Path,
                    default=REPO / "data/results/scale_sweep_v3")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args(argv)

    models: dict[str, dict[str, Any]] = {}
    for root in (args.grid_root, args.sweep_root):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            name = d.name.replace("_", "/", 1) if "_" in d.name and root == args.grid_root else d.name
            entry = models.setdefault(name, {})
            for cond in CONDITIONS:
                f = d / f"{cond}.json"
                if not f.is_file():
                    continue
                rows = json.loads(f.read_text())
                labels = [int(r["label"] == "load_bearing") for r in rows]
                block: dict[str, Any] = {
                    "n_candidates": len(rows),
                    "V": auc([float(r["V"]) for r in rows], labels) if "V" in rows[0] else None,
                    "W_rr": auc([float(r["W_rr"]) for r in rows], labels),
                    "V_raw": auc([float(r["V_raw"]) for r in rows], labels) if "V_raw" in rows[0] else None,
                    "yes_rate": (sum(float(r["V"]) >= 0.5 for r in rows) / len(rows)) if "V" in rows[0] else None,
                }
                if block["V"] is not None:
                    block["gap"] = block["W_rr"] - block["V"]
                    block["W_minus_V_paired"] = paired_contrast(rows, "W_rr", "V")
                    # metacognitive efficiency is only defined where the workspace is informative
                    block["Ms"] = (block["V"] / block["W_rr"]) if block["W_rr"] >= 0.55 else None
                if block["V"] is not None and block["V_raw"] is not None:
                    block["V_chat_minus_V_raw_paired"] = paired_contrast(rows, "V", "V_raw")
                entry[cond] = block

    # Qwen3 dense scale ladder, gap deltas between adjacent sizes and 8B->32B
    ladder = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]
    ladder = [m for m in ladder if m in models]
    scale = {}
    for cond in CONDITIONS:
        pairs = [(ladder[i], ladder[i + 1]) for i in range(len(ladder) - 1)]
        if "Qwen3-8B" in ladder and "Qwen3-32B" in ladder:
            pairs.append(("Qwen3-8B", "Qwen3-32B"))
        steps = []
        for a, b in pairs:
            fa = args.sweep_root / a / f"{cond}.json"
            fb = args.sweep_root / b / f"{cond}.json"
            if not (fa.is_file() and fb.is_file()):
                continue
            d = cross_model_gap_delta(json.loads(fa.read_text()), json.loads(fb.read_text()))
            if d:
                steps.append({"from": a, "to": b, **d})
        scale[cond] = steps

    payload = {"schema_version": "corrected-primary-contrasts/v1",
               "verbal_score_schema": "workspace_measurement_metadata.v3",
               "bootstrap": {"draws": DRAWS, "unit": "episode_cluster", "paired": True},
               "models": models, "qwen3_scale_gap_deltas": scale}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    L = ["# Corrected primary contrasts and scale analysis", "",
         "All intervals are 4,000-draw whole-episode cluster bootstraps. Contrasts that "
         "compare two quantities on the same candidates share their draws, so the "
         "difference is paired. `Ms` is reported only where `W_rr >= 0.55`.", ""]
    for cond in CONDITIONS:
        L += [f"## {cond}", "",
              "| Model | V | W_rr | gap | 95% CI on W−V | V_raw | Ms |",
              "|---|---:|---:|---:|---|---:|---:|"]
        for name in sorted(models):
            b = models[name].get(cond)
            if not b or b.get("V") is None:
                continue
            p = b.get("W_minus_V_paired") or {}
            ci = p.get("ci_95")
            star = "*" if p.get("excludes_zero") else ""
            L.append(
                f"| {name} | {b['V']:.4f} | {b['W_rr']:.4f} | {b['gap']:+.4f}{star} | "
                + (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "n/a") + " | "
                + (f"{b['V_raw']:.4f}" if b.get("V_raw") is not None else "n/a") + " | "
                + (f"{b['Ms']:.3f}" if b.get("Ms") is not None else "n/a") + " |")
        L.append("")
        if scale.get(cond):
            L += ["Paired gap change across Qwen3 dense scales (shared episode draws):", "",
                  "| Step | Δgap | 95% CI |", "|---|---:|---|"]
            for s in scale[cond]:
                st = "*" if s["excludes_zero"] else ""
                L.append(f"| {s['from'].replace('Qwen3-','')} → {s['to'].replace('Qwen3-','')} | "
                         f"{s['estimate']:+.4f}{st} | [{s['ci_95'][0]:+.4f}, {s['ci_95'][1]:+.4f}] |")
            L.append("")
    L += ["`*` marks an interval that excludes zero.", ""]
    args.out_md.write_text("\n".join(L))
    print(f"wrote {args.out_md} over {len(models)} checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
