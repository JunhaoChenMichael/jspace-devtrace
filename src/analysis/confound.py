"""
confound.py — is the workspace AUC real, or a surface-presence artifact?

The worry: if load_bearing concepts appear literally in the context more often
than distractors, then W (which can linger on recently-seen tokens) could be
detecting "did this word appear" rather than "is it held in the workspace".

Checks:
  1. How strongly does literal appearance correlate with the load_bearing label?
     (severity of the potential confound)
  2. AUC of a DUMB surface baseline: appearance(0/1) predicting load_bearing.
  3. AUC of W restricted to concepts that DO NOT appear literally (pure latent
     workspace signal — the clean test). If W still discriminates here, the
     signal is not a surface artifact.
"""
import sys, json, re
import numpy as np
from sklearn.metrics import roc_auc_score


def appears(concept, context):
    return re.search(re.escape(concept.lower()), context.lower()) is not None


def main(results_path, battery_path="data/benchmarks/battery.json"):
    rows = json.load(open(results_path))
    battery = json.load(open(battery_path))
    # map (episode, concept) -> context
    ctx = {i: ep["context"] for i, ep in enumerate(battery)}

    lb, app, Wrr = [], [], []
    for r in rows:
        lb.append(1 if r["label"] == "load_bearing" else 0)
        app.append(1 if appears(r["concept"], ctx[r["episode"]]) else 0)
        Wrr.append(r.get("W_rr", r["W_end"]))
    lb, app, Wrr = np.array(lb), np.array(app), np.array(Wrr)

    print(f"\n==== confound check: {results_path} ====")
    n = len(lb)
    print(f"items={n}  load_bearing={lb.sum()}")
    print(f"literal-appearance rate: load_bearing={app[lb==1].mean():.2f}  "
          f"non-load={app[lb==0].mean():.2f}   (gap = confound severity)")

    # dumb surface baseline
    if 0 < app.sum() < n:
        print(f"AUC(appearance -> load_bearing) [dumb surface baseline] = {roc_auc_score(lb, app):.3f}")
    print(f"AUC(W_rr -> load_bearing) [all items]                   = {roc_auc_score(lb, Wrr):.3f}")

    # clean test: concepts that never appear literally (pure latent)
    mask = app == 0
    if lb[mask].sum() >= 3 and (1 - lb[mask]).sum() >= 3:
        print(f"AUC(W_rr) on NON-appearing concepts only (n={mask.sum()}, "
              f"load={lb[mask].sum()}) = {roc_auc_score(lb[mask], Wrr[mask]):.3f}"
              f"   <-- pure latent workspace signal")
    else:
        print(f"(too few non-appearing load_bearing items for a clean subset test: "
              f"n={mask.sum()}, load={lb[mask].sum()})")

    # also: among appearing concepts, does W still separate? (controls for appearance)
    mask2 = app == 1
    if lb[mask2].sum() >= 3 and (1 - lb[mask2]).sum() >= 3:
        print(f"AUC(W_rr) on APPEARING concepts only (n={mask2.sum()}) = "
              f"{roc_auc_score(lb[mask2], Wrr[mask2]):.3f}   <-- W beyond mere presence")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
