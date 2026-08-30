#!/usr/bin/env python3
"""Scale trend for the metacognitive reporting gap across Qwen3 sizes.

For every model size and battery condition this recomputes, from raw
measurement rows only:

  V       verbal report AUC (chat template)
  V_raw   verbal report AUC (template-free), separating instruct/template
          behaviour from base capability
  W_rr    workspace readout AUC
  gap     W_rr - V, the quantity Binary Metacognitive Alignment repairs

and fits gap and V against log10(parameters).  Five sizes from one family is a
scale TREND, not a scaling law in the Kaplan/Hoffmann sense: the points are not
compute-matched and the sizes may differ in training recipe, so the fit is
reported with that caveat attached rather than extrapolated.

It also reports the surface-presence baseline: whether a candidate concept
appears literally in the context.  In this benchmark that feature is an almost
perfect ANTI-predictor of utility on the non-Explicit conditions, which is the
hypothesis for why small models score below chance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "explicit": "battery_v1_final",
    "evoked": "battery_v2_final",
    "decoupled": "battery_v4_final",
    "compositional": "battery_v3d",
}
SIGNALS = ("V", "V_raw", "W_rr")


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
        average = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def within_episode_auc(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    by_episode: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    values = []
    for episode_rows in by_episode.values():
        labels = [int(r["label"] == "load_bearing") for r in episode_rows]
        value = auc([float(r[key]) for r in episode_rows], labels)
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def parameter_count(config: Mapping[str, Any]) -> int:
    """Parameter count from the architecture, so no model-card lookup.

    This counts ONE MLP per layer, so for a mixture-of-experts config it returns
    the dense (per-token) path rather than the total parameter count. Callers
    must read `parameter_basis` before labelling an MoE point.
    """

    v = int(config["vocab_size"])
    h = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    inter = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    head_dim = int(config.get("head_dim", h // heads))
    attn = h * heads * head_dim + 2 * h * kv_heads * head_dim + heads * head_dim * h
    mlp = 3 * h * inter
    norms = 2 * h + 2 * heads * head_dim  # rms norms incl. q/k norm in Qwen3
    per_layer = attn + mlp + norms
    embeddings = v * h if config.get("tie_word_embeddings", True) else 2 * v * h
    return embeddings + layers * per_layer + h


def surface_baseline(condition: str) -> dict[str, Any]:
    battery = json.loads(
        (REPO_ROOT / "data" / "benchmarks" / f"{CONDITIONS[condition]}.json").read_text()
    )
    scores, labels = [], []
    load_bearing_present = load_bearing = negatives_present = negatives = 0
    for episode in battery:
        context = episode["context"].lower()
        for item in episode["items"]:
            present = int(item["concept"].lower() in context)
            positive = int(item["label"] == "load_bearing")
            scores.append(present)
            labels.append(positive)
            if positive:
                load_bearing += 1
                load_bearing_present += present
            else:
                negatives += 1
                negatives_present += present
    return {
        "literal_presence_auc": auc(scores, labels),
        "load_bearing_literally_present": load_bearing_present / load_bearing,
        "negatives_literally_present": negatives_present / negatives,
    }


def episode_bootstrap_auc(
    rows: Sequence[Mapping[str, Any]], key: str, *, draws: int = 4000, seed: int = 0
) -> dict[str, Any]:
    """Episode-cluster bootstrap CI for one model/condition/signal."""

    import random

    by_episode: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    episodes = sorted(by_episode)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        picked = [r for e in sample for r in by_episode[e]]
        value = auc(
            [float(r[key]) for r in picked],
            [int(r["label"] == "load_bearing") for r in picked],
        )
        if value is not None:
            estimates.append(value)
    estimates.sort()
    if not estimates:
        return {}
    lo = estimates[int(0.025 * (len(estimates) - 1))]
    hi = estimates[int(0.975 * (len(estimates) - 1))]
    return {"ci_95": [lo, hi], "draws_effective": len(estimates)}


def paired_scale_delta(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    key: str,
    *,
    draws: int = 4000,
    seed: int = 0,
) -> dict[str, Any] | None:
    """CI on AUC(b) - AUC(a) using SHARED episode draws.

    Two model sizes see exactly the same episodes, so the draw can be shared and
    the difference is paired; treating the two AUCs as independent would inflate
    the interval and hide a real jump.
    """

    import random

    def group(rows):
        out: dict[Any, list[Mapping[str, Any]]] = {}
        for row in rows:
            out.setdefault(row["episode"], []).append(row)
        return out

    ga, gb = group(rows_a), group(rows_b)
    episodes = sorted(set(ga) & set(gb))
    if not episodes:
        return None

    def pooled(g, sample):
        picked = [r for e in sample for r in g[e]]
        return auc(
            [float(r[key]) for r in picked],
            [int(r["label"] == "load_bearing") for r in picked],
        )

    point = (pooled(gb, episodes) or 0.0) - (pooled(ga, episodes) or 0.0)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        va, vb = pooled(ga, sample), pooled(gb, sample)
        if va is not None and vb is not None:
            estimates.append(vb - va)
    estimates.sort()
    if not estimates:
        return None
    return {
        "delta": point,
        "ci_95": [
            estimates[int(0.025 * (len(estimates) - 1))],
            estimates[int(0.975 * (len(estimates) - 1))],
        ],
        "excludes_zero": estimates[int(0.025 * (len(estimates) - 1))] > 0
        or estimates[int(0.975 * (len(estimates) - 1))] < 0,
        "draws_effective": len(estimates),
    }


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float] | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return {
        "slope_per_decade": slope,
        "intercept": intercept,
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_points": n,
    }


def collect(sweep_root: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    raw_rows: dict[str, dict[str, Any]] = {}
    for model_dir in sorted(sweep_root.iterdir()):
        if not model_dir.is_dir():
            continue
        config_path = None
        cache = Path(REPO_ROOT).parent / "cache" / "hub" / f"models--Qwen--{model_dir.name}"
        refs = cache / "refs" / "main"
        if refs.is_file():
            config_path = cache / "snapshots" / refs.read_text().strip() / "config.json"
        raw_config = (
            json.loads(config_path.read_text())
            if config_path and config_path.is_file()
            else {}
        )
        is_moe = bool(raw_config.get("num_experts"))
        entry: dict[str, Any] = {
            "parameter_basis": "dense_per_token_path" if is_moe else "total",
            "mixture_of_experts": (
                {"experts": raw_config.get("num_experts"),
                 "experts_per_token": raw_config.get("num_experts_per_tok")}
                if is_moe else None
            ),
            "revision": refs.read_text().strip() if refs.is_file() else None,
            "parameters": (
                parameter_count(json.loads(config_path.read_text()))
                if config_path and config_path.is_file()
                else None
            ),
            "conditions": {},
        }
        for condition in CONDITIONS:
            path = model_dir / f"{condition}.json"
            if not path.is_file():
                continue
            rows = json.loads(path.read_text())
            labels = [int(r["label"] == "load_bearing") for r in rows]
            block: dict[str, Any] = {"n_rows": len(rows), "n_positive": sum(labels)}
            raw_rows.setdefault(model_dir.name, {})[condition] = rows
            for signal in SIGNALS:
                if not rows or signal not in rows[0]:
                    block[signal] = None
                    continue
                block[signal] = {
                    "pooled_auc": auc([float(r[signal]) for r in rows], labels),
                    "within_episode_auc": within_episode_auc(rows, signal),
                    **episode_bootstrap_auc(rows, signal),
                }
            if block.get("W_rr") and block.get("V"):
                block["gap_pooled"] = block["W_rr"]["pooled_auc"] - block["V"]["pooled_auc"]
                block["gap_within_episode"] = (
                    block["W_rr"]["within_episode_auc"] - block["V"]["within_episode_auc"]
                )
            entry["conditions"][condition] = block
        models[model_dir.name] = entry

    ordered = [m for m in models if models[m]["parameters"]]
    ordered.sort(key=lambda m: models[m]["parameters"])
    trends: dict[str, Any] = {}
    for condition in CONDITIONS:
        xs, v_ys, gap_ys, w_ys = [], [], [], []
        for name in ordered:
            block = models[name]["conditions"].get(condition)
            if not block or not block.get("V"):
                continue
            xs.append(math.log10(models[name]["parameters"]))
            v_ys.append(block["V"]["pooled_auc"])
            w_ys.append(block["W_rr"]["pooled_auc"])
            gap_ys.append(block["gap_pooled"])
        steps = []
        for earlier, later in zip(ordered, ordered[1:]):
            rows_a = raw_rows.get(earlier, {}).get(condition)
            rows_b = raw_rows.get(later, {}).get(condition)
            if not rows_a or not rows_b:
                continue
            entry = {"from": earlier, "to": later}
            for signal in SIGNALS:
                if signal not in rows_a[0] or signal not in rows_b[0]:
                    continue
                entry[signal] = paired_scale_delta(rows_a, rows_b, signal)
            steps.append(entry)
        trends[condition] = {
            "adjacent_scale_paired_deltas": steps,
            "log10_parameters": xs,
            "V_fit": linear_fit(xs, v_ys),
            "W_rr_fit": linear_fit(xs, w_ys),
            "gap_fit": linear_fit(xs, gap_ys),
            "surface_baseline": surface_baseline(condition),
        }
    return {
        "schema_version": "metacog-scale-trend/v1",
        "models": models,
        "model_order": ordered,
        "trends": trends,
        "caveats": [
            "Five sizes from one model family; not compute-matched, so this is a "
            "scale trend rather than a scaling law, and it is not extrapolated.",
            "Sizes may differ in training data mix and post-training recipe; "
            "'scale' here is a proxy for everything that changes between releases.",
            "Measurement only: no training, no adapters, no OOD-driven choices.",
        ],
    }


def render(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Metacognitive reporting gap across Qwen3 scale",
        "",
        "Measurement only, no training. `V` is the verbal report AUC, `W_rr` the "
        "workspace readout AUC, and `gap = W_rr - V` is what Binary Metacognitive "
        "Alignment repairs. `V_raw` is the template-free readout.",
        "",
    ]
    order = summary["model_order"]
    for condition in CONDITIONS:
        lines += [
            f"## {condition}",
            "",
            "| Model | params | V | V_raw | W_rr | gap (W-V) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in order:
            block = summary["models"][name]["conditions"].get(condition)
            if not block or not block.get("V"):
                continue
            params = summary["models"][name]["parameters"] / 1e9
            raw = block.get("V_raw")
            lines.append(
                f"| {name} | {params:.1f}B | {block['V']['pooled_auc']:.4f} | "
                f"{(raw['pooled_auc'] if raw else float('nan')):.4f} | "
                f"{block['W_rr']['pooled_auc']:.4f} | {block['gap_pooled']:+.4f} |"
            )
        trend = summary["trends"][condition]
        surface = trend["surface_baseline"]
        lines += [
            "",
            f"Surface-presence baseline: literal-mention AUC "
            f"{surface['literal_presence_auc']:.4f} "
            f"(load-bearing literally present {surface['load_bearing_literally_present']:.0%}, "
            f"negatives {surface['negatives_literally_present']:.0%}).",
            "",
        ]
        steps = trend.get("adjacent_scale_paired_deltas") or []
        if steps:
            lines += [
                "Adjacent-scale paired deltas (shared episode draws, 4,000):",
                "",
                "| Step | delta V | 95% CI | delta V_raw | 95% CI |",
                "|---|---:|---|---:|---|",
            ]
            for step in steps:
                v, vr = step.get("V"), step.get("V_raw")
                def fmt(d):
                    if not d:
                        return "n/a", "n/a"
                    star = " *" if d["excludes_zero"] else ""
                    return (f"{d['delta']:+.4f}{star}",
                            f"[{d['ci_95'][0]:+.4f}, {d['ci_95'][1]:+.4f}]")
                vd, vc = fmt(v)
                rd, rc = fmt(vr)
                lines.append(
                    f"| {step['from'].replace('Qwen3-','')} -> {step['to'].replace('Qwen3-','')} "
                    f"| {vd} | {vc} | {rd} | {rc} |"
                )
            lines += ["", "`*` marks an interval that excludes zero.", ""]
        for key, label in (("V_fit", "V"), ("gap_fit", "gap")):
            fit = trend.get(key)
            if fit:
                lines.append(
                    f"- {label} vs log10(params): slope {fit['slope_per_decade']:+.4f} "
                    f"per decade, R² {fit['r_squared']:.3f} over {fit['n_points']} sizes"
                )
        lines.append("")
    diagnostics = summary.get("diagnostic_models") or {}
    if diagnostics:
        lines += [
            "## Diagnostic: sparse (MoE) model, reported separately",
            "",
            "Qwen3-30B-A3B activates 3B of 30B parameters per token. It is NOT a "
            "substitute for the dense 32B scale point -- both campaign plans forbid "
            "that substitution -- and it is excluded from every fit above. It is "
            "here to separate total parameters from active compute.",
            "",
            "| Model | Condition | V | V_raw | W_rr | gap (W-V) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name, entry in diagnostics.items():
            for condition in CONDITIONS:
                block = entry["conditions"].get(condition)
                if not block or not block.get("V"):
                    continue
                raw = block.get("V_raw")
                lines.append(
                    f"| {name} | {condition} | {block['V']['pooled_auc']:.4f} | "
                    f"{(raw['pooled_auc'] if raw else float('nan')):.4f} | "
                    f"{block['W_rr']['pooled_auc']:.4f} | {block['gap_pooled']:+.4f} |"
                )
        lines += [
            "",
            "On Decoupled the sparse model reports worse than every dense model "
            "measured, including 1.7B, while its workspace readout matches the 8B "
            "dense model. Total parameter count does not carry the transition; "
            "active compute tracks it. The sparse model also differs in post-training, "
            "so this isolates the comparison rather than settling it -- but the "
            "within-model dissociation (workspace fine, report poor) does not depend "
            "on cross-model comparability.",
            "",
        ]
    lines += ["## Caveats", ""] + [f"- {c}" for c in summary["caveats"]] + [""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=REPO_ROOT / "data/results/scale_sweep")
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=REPO_ROOT / "data/results/scale_sweep_diagnostic",
        help="models reported separately from the dense sweep (e.g. MoE), never mixed into the fit",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = collect(args.sweep_root)
    if args.diagnostic_root.is_dir():
        summary["diagnostic_models"] = collect(args.diagnostic_root)["models"]
    for path, text in (
        (args.out_json, json.dumps(summary, indent=2, sort_keys=True) + "\n"),
        (args.out_md, render(summary)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"wrote {args.out_md} over {len(summary['model_order'])} sizes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
