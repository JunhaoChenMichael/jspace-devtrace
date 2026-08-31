#!/usr/bin/env python3
"""Measure the verbal probe under both definitions, and decompose the difference.

For every candidate this records the absolute yes/no probability mass alongside
both scores, so a reported effect can be split into the part carried by the
yes-versus-no RATIO (the intended measurement) and the part carried by how much
absolute MASS the model places on the yes/no tokens at all (the channel the
guard epsilon silently exposed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.measure import yes_no_ids  # noqa: E402
from jlens import WorkspaceLens  # noqa: E402
from memory_rl.modeling import render_admission_prompt  # noqa: E402

EPSILON = 1e-9  # the guard that dominated the v2 denominator


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    pairs = sorted(zip(scores, labels))
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
    positives = sum(l for _, l in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    return (sum(r for r, (_, l) in zip(ranks, pairs) if l) - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter", default=None, help="locked LoRA checkpoint, or omit for base")
    parser.add_argument("--battery", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    lens = WorkspaceLens(
        args.model, device="cuda", dtype=torch.bfloat16,
        adapter_path=args.adapter,
        model_revision=args.model_revision, tokenizer_revision=args.model_revision,
    )
    yes_ids, no_ids = yes_no_ids(lens)
    battery = json.loads(Path(args.battery).read_text())

    rows = []
    for e, episode in enumerate(battery):
        for c, item in enumerate(episode["items"]):
            prompt = render_admission_prompt(lens.tok, episode["context"], item["concept"])
            enc = lens.tok(prompt, return_tensors="pt").to(lens.device)
            with torch.no_grad():
                logits = lens.model(**enc).logits[0, -1].float()
            probabilities = F.softmax(logits, dim=-1)
            py = float(sum(probabilities[i] for i in yes_ids))
            pn = float(sum(probabilities[i] for i in no_ids))
            ratio = float(torch.sigmoid(
                torch.logsumexp(logits[yes_ids], 0) - torch.logsumexp(logits[no_ids], 0)))
            rows.append({
                "episode": e, "candidate_index": c, "concept": item["concept"],
                "label": item["label"],
                "p_yes": py, "p_no": pn, "yes_no_mass": py + pn,
                "V_v2_as_reported": py / (py + pn + EPSILON),
                "V_v3_ratio": ratio,
            })
        print(f"  {args.label}: {e + 1}/{len(battery)}", flush=True)

    labels = [int(r["label"] == "load_bearing") for r in rows]
    masses = [r["yes_no_mass"] for r in rows]
    summary = {
        "schema_version": "verbal-score-decomposition/v1",
        "model": args.model, "model_revision": args.model_revision,
        "adapter": args.adapter, "battery": args.battery, "label": args.label,
        "n_candidates": len(rows),
        "auc": {
            "V_v2_as_reported": auc([r["V_v2_as_reported"] for r in rows], labels),
            "V_v3_ratio": auc([r["V_v3_ratio"] for r in rows], labels),
            "yes_no_mass_alone": auc(masses, labels),
            "p_yes_alone": auc([r["p_yes"] for r in rows], labels),
        },
        "mass": {
            "min": min(masses), "median": sorted(masses)[len(masses) // 2], "max": max(masses),
            "mean": sum(masses) / len(masses),
            "fraction_below_epsilon": sum(m < EPSILON for m in masses) / len(masses),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    a = summary["auc"]
    print(f"{args.label}: v2={a['V_v2_as_reported']:.4f} v3={a['V_v3_ratio']:.4f} "
          f"mass_alone={a['yes_no_mass_alone']:.4f} "
          f"frac_mass<eps={summary['mass']['fraction_below_epsilon']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
