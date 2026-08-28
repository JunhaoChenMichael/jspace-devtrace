"""
analyze.py — the go/no-go readout.

Questions:
  Q1  Does workspace salience (W) predict ground-truth utility (load_bearing)
      above chance?  -> AUC(W)
  Q2  Does W beat the model's own verbal reflection (V)?  -> AUC(W) vs AUC(V)
  Q3  (make-or-break) Do W and V actually DIVERGE, and when they disagree, which
      one is right?  -> corr(W,V) and disagreement-resolved-by-ground-truth
Also runs the harder load_bearing-vs-distractor contrast (drops filler), since
distractors are the items verbal reflection is designed to be fooled by.
"""
import sys, json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def load(path):
    with open(path) as f:
        return json.load(f)


def auc(scores, pos_mask):
    y = np.array(pos_mask, dtype=int)
    s = np.array(scores, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return roc_auc_score(y, s)


def auc_ci(scores, pos_mask, B=2000, seed=0):
    """AUC with a bootstrap 95% CI (resample items)."""
    y = np.array(pos_mask, dtype=int)
    s = np.array(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy, ss = y[idx], s[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        boots.append(roc_auc_score(yy, ss))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc(s, y), lo, hi


def auc_diff_ci(sA, sB, pos_mask, B=2000, seed=0):
    """Paired bootstrap CI for AUC(A) - AUC(B) on the same items."""
    y = np.array(pos_mask, dtype=int)
    a = np.array(sA, dtype=float); b = np.array(sB, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y); diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        diffs.append(roc_auc_score(yy, a[idx]) - roc_auc_score(yy, b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return auc(a, y) - auc(b, y), lo, hi, float(np.mean(np.array(diffs) > 0))


def zscore(x):
    x = np.array(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd + 1e-9)


def report(path):
    rows = load(path)
    have_V = all("V" in r for r in rows)
    labels = [r["label"] for r in rows]
    lb = [l == "load_bearing" for l in labels]
    ds = [l == "distractor" for l in labels]
    W = [r["W_end"] for r in rows]
    Wm = [r["W_max"] for r in rows]
    Wrr = [r.get("W_rr", 0.0) for r in rows]
    V = [r.get("V", float("nan")) for r in rows]
    have_rr = any(r.get("W_rr") for r in rows)

    print(f"\n================ {path} ================")
    print(f"items={len(rows)}  load_bearing={sum(lb)}  distractor={sum(ds)}  "
          f"filler={len(rows)-sum(lb)-sum(ds)}  verbal={have_V}")

    # --- Q1/Q2: AUCs, load_bearing vs REST ---
    print("\n[load_bearing vs rest]  AUC [95% bootstrap CI]  (>0.5 = discriminates utility)")
    if have_rr:
        a, lo, hi = auc_ci(Wrr, lb); print(f"  W_rr  (workspace, rank)  : {a:.3f} [{lo:.3f}, {hi:.3f}]")
    a, lo, hi = auc_ci(W, lb);  print(f"  W_end (workspace, prob)  : {a:.3f} [{lo:.3f}, {hi:.3f}]")
    a, lo, hi = auc_ci(Wm, lb); print(f"  W_max (workspace, noisy) : {a:.3f} [{lo:.3f}, {hi:.3f}]")
    have_Vraw = all("V_raw" in r for r in rows)
    Vraw = [r.get("V_raw", float("nan")) for r in rows]
    if have_Vraw:
        a, lo, hi = auc_ci(Vraw, lb); print(f"  V_raw (verbal, no template): {a:.3f} [{lo:.3f}, {hi:.3f}]")
    if have_V:
        a, lo, hi = auc_ci(V, lb); print(f"  V     (verbal reflection): {a:.3f} [{lo:.3f}, {hi:.3f}]")
        # NOTE: use W_rr consistently for the paired diff (an earlier version
        # silently switched to W_end when its point AUC was higher, which mixed
        # signal definitions within one results table).
        best_W = Wrr if have_rr else W
        d, dlo, dhi, pgt = auc_diff_ci(best_W, V, lb)
        print(f"  --> AUC(W_rr) - AUC(V) = {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]  "
              f"P(workspace>V) = {pgt:.2f}")
    if have_Vraw and not have_V:
        d, dlo, dhi, pgt = auc_diff_ci(Wrr if have_rr else W, Vraw, lb)
        print(f"  --> AUC(W_rr) - AUC(V_raw) = {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]  "
              f"P(workspace>V_raw) = {pgt:.2f}")

    # --- harder: load_bearing vs DISTRACTOR only ---
    idx = [i for i in range(len(rows)) if lb[i] or ds[i]]
    lb2 = [lb[i] for i in idx]
    print("\n[load_bearing vs distractor only]  the regime reflection is fooled by")
    if have_rr:
        print(f"  W_rr  (workspace, rank)  : {auc([Wrr[i] for i in idx], lb2):.3f}")
    print(f"  W_end (workspace, prob)  : {auc([W[i] for i in idx], lb2):.3f}")
    if have_V:
        print(f"  V     (verbal reflection): {auc([V[i] for i in idx], lb2):.3f}")

    if not have_V:
        return

    # --- Q3: divergence + who-wins-on-disagreement ---
    zW, zV = zscore(W), zscore(V)
    rho, _ = spearmanr(W, V)
    print(f"\n[divergence]  spearman corr(W,V) = {rho:.3f}  "
          f"(low corr => workspace carries info reflection doesn't)")

    # percentile-rank each signal so they are on a common 0..1 scale, then look
    # at items where they most disagree and ask which one matched ground truth.
    def prank(x):
        order = np.argsort(np.argsort(x))
        return order / (len(x) - 1)
    pW, pV = prank(np.array(W)), prank(np.array(V))
    gap = pW - pV
    thr = np.percentile(np.abs(gap), 60)  # the more-disagreeing 40% of items
    w_hi = np.where(gap > thr)[0]   # workspace ranks it far above reflection
    v_hi = np.where(gap < -thr)[0]  # reflection ranks it far above workspace
    lb_arr = np.array(lb)
    if len(w_hi) and len(v_hi):
        print(f"  on strong disagreements (|rank gap|>{thr:.2f}):")
        print(f"    workspace-favored items: P(load_bearing) = {lb_arr[w_hi].mean():.3f}  n={len(w_hi)}")
        print(f"    reflection-favored items: P(load_bearing) = {lb_arr[v_hi].mean():.3f}  n={len(v_hi)}")

    # concrete examples of the money-quadrant: high-W low-V, load_bearing
    print("\n[examples] used-but-not-flagged  (high W, low V, load_bearing):")
    cand = sorted([i for i in range(len(rows)) if lb[i]], key=lambda i: zW[i] - zV[i], reverse=True)
    for i in cand[:4]:
        print(f"    ep{rows[i]['episode']:2d} '{rows[i]['concept']}'  W={W[i]:.4f} V={V[i]:.3f}")
    print("[examples] flagged-but-useless  (high V, low W, distractor):")
    cand2 = sorted([i for i in range(len(rows)) if ds[i]], key=lambda i: zV[i] - zW[i], reverse=True)
    for i in cand2[:4]:
        print(f"    ep{rows[i]['episode']:2d} '{rows[i]['concept']}'  W={W[i]:.4f} V={V[i]:.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
