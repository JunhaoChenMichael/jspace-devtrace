"""
mixed_pool.py — evaluate write-admission policies on the MIXED-provenance pool
(Explicit ∪ Evoked ∪ Decoupled-L), the deployment-realistic setting where any
single-channel policy fails on its weak regime. Pools per-episode records from
existing downstream runs (identical episodes and generation budgets across
policies within each benchmark) and compares policies pairwise with exact
McNemar tests.
"""
import os, sys, json, argparse
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "data", "results")

POOL = [
    ("Explicit", "downstream_v1f_{s}-Instruct.json", "downstream_routedpr_v1f_{s}.json"),
    ("Evoked", "downstream_v2f_{s}-Instruct.json", "downstream_routedpr_v2f_{s}.json"),
    ("Decoupled-L", "downstream_v4xl_{s}-Instruct.json", "downstream_routedpr_v4xl_{s}.json"),
]
POLICIES = ["workspace", "verbal", "embedding", "recency", "oracle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="0.5B,7B")
    ap.add_argument("--budgets", default="2,3")
    args = ap.parse_args()

    for s in args.sizes.split(","):
        for k in args.budgets.split(","):
            per = {p: [] for p in POLICIES + ["routed"]}
            for bench, base_f, routed_f in POOL:
                base = json.load(open(os.path.join(R, base_f.format(s=s))))
                routed = json.load(open(os.path.join(R, routed_f.format(s=s))))
                for p in POLICIES:
                    per[p] += [r["correct"] for r in base["per_episode"][f"{p}@{k}"]]
                per["routed"] += [r["correct"] for r in routed["per_episode"][f"routed@{k}"]]
            n = len(per["routed"])
            print(f"\n== Mixed pool ({n} episodes), {s}, k={k}")
            for p in ["routed"] + POLICIES:
                print(f"  {p:10s} acc={sum(per[p])/n:.3f}")
            for p in POLICIES:
                a, b = per["routed"], per[p]
                b01 = sum(1 for x, y in zip(a, b) if x and not y)
                b10 = sum(1 for x, y in zip(a, b) if y and not x)
                pv = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
                print(f"  routed vs {p:10s}: +{b01}/-{b10}  p={pv:.4f}")


if __name__ == "__main__":
    main()
