"""
absent_foils.py — within-absent discrimination test, answering the strongest
benchmark-validity critique: on the inferred-regime benchmarks the bridge is
nearly the only context-absent candidate, so a trivial "keep absent strings"
rule matches any channel at the episode level. The real question is whether
the workspace readout discriminates AMONG absent concepts: does it rank the
bridge this context evokes above equally-absent foils it does not?

For each episode, foils are the load-bearing bridges of OTHER episodes in the
same benchmark, filtered to be absent from this context and distinct from all
of this episode's candidates. Every (bridge + foils) set is scored in the
episode's own context:
  W mode: one encoding pass; case-variant peak reciprocal rank for all tokens.
  V mode: the yes/no importance probe per concept (base or fine-tuned model),
          testing whether a post-alignment reporter says yes to any absent
          string (shortcut) or only to the evoked bridge (calibration).

Reports within-absent AUC (bridge vs. foils) with an episode-cluster
bootstrap CI.
"""
import os, sys, json, argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jlens import WorkspaceLens
from measure import (concept_token_ids, workspace_salience, yes_no_ids,
                     verbal_salience)


def build_foil_sets(battery, k_foils, seed_offset=1):
    bridges = []
    for ep in battery:
        lb = [it["concept"] for it in ep["items"] if it["label"] == "load_bearing"]
        bridges.append(lb[0] if lb else None)
    out = []
    n = len(battery)
    for i, ep in enumerate(battery):
        ctx = ep["context"].lower()
        own = {it["concept"].lower() for it in ep["items"]}
        foils = []
        j = (i + seed_offset) % n
        while len(foils) < k_foils and j != i:
            c = bridges[j]
            if (c and c.lower() not in own and ctx.rfind(c.lower()) < 0
                    and c.lower() not in [f.lower() for f in foils]):
                foils.append(c)
            j = (j + 1) % n
        out.append(foils)
    return bridges, out


def cluster_auc_ci(rows, B=4000, seed=0):
    """rows: list of (episode, score, y). Bootstrap over episodes."""
    from sklearn.metrics import roc_auc_score
    eps = sorted({r[0] for r in rows})
    by_ep = {e: [(s, y) for (e2, s, y) in rows if e2 == e] for e in eps}
    y = [r[2] for r in rows]; s = [r[1] for r in rows]
    a = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.choice(len(eps), len(eps))
        ss, yy = [], []
        for i in pick:
            for (sv, yv) in by_ep[eps[i]]:
                ss.append(sv); yy.append(yv)
        if 0 < sum(yy) < len(yy):
            vals.append(roc_auc_score(yy, ss))
    return a, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--mode", choices=["W", "V"], default="W")
    ap.add_argument("--k-foils", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    battery = json.load(open(args.battery))
    bridges, foil_sets = build_foil_sets(battery, args.k_foils)
    lens = WorkspaceLens(args.model, dtype=getattr(torch, args.dtype))
    print(f"model={args.model} battery={args.battery} mode={args.mode} "
          f"episodes={len(battery)} foils/ep={args.k_foils}", flush=True)

    rows = []
    if args.mode == "W":
        for i, ep in enumerate(battery):
            if not bridges[i]:
                continue
            concepts = [bridges[i]] + foil_sets[i]
            ids_of = {c: concept_token_ids(lens, c) for c in concepts}
            flat = sorted({t for ids in ids_of.values() for t in ids})
            _, _, rr = workspace_salience(lens, ep["context"], flat, end_only=True)
            for c in concepts:
                score = max((rr[t] for t in ids_of[c]), default=0.0)
                rows.append({"episode": i, "concept": c,
                             "score": score, "is_bridge": c == bridges[i]})
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(battery)}", flush=True)
    else:
        yes_ids, no_ids = yes_no_ids(lens)
        for i, ep in enumerate(battery):
            if not bridges[i]:
                continue
            for c in [bridges[i]] + foil_sets[i][:5]:
                v = verbal_salience(lens, ep["context"], c, yes_ids, no_ids)
                rows.append({"episode": i, "concept": c,
                             "score": v, "is_bridge": c == bridges[i]})
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(battery)}", flush=True)

    trip = [(r["episode"], r["score"], 1 if r["is_bridge"] else 0) for r in rows]
    a, lo, hi = cluster_auc_ci(trip)
    print(f"\nwithin-absent AUC (bridge vs {args.k_foils if args.mode=='W' else 5} "
          f"absent foils): {a:.3f} [{lo:.3f}, {hi:.3f}]")
    json.dump({"model": args.model, "battery": args.battery, "mode": args.mode,
               "auc": a, "ci": [lo, hi], "rows": rows}, open(args.out, "w"))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
