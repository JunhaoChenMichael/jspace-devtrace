"""
summary.py — master results table across all measured checkpoints.

Scans results_*.json files given on the command line (or a default glob) and
prints one markdown table per battery version: W_rr AUC [CI], V AUC [CI],
W-V paired diff [CI] + P(W>V). Rows are (model, base/instruct).

Usage: python pilot/summary.py [results files...]
"""
import sys, glob, json, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import auc_ci, auc_diff_ci


def row(path):
    rows = json.load(open(path))
    lb = [r["label"] == "load_bearing" for r in rows]
    Wrr = [r.get("W_rr", 0.0) for r in rows]
    have_V = all("V" in r for r in rows)
    a, lo, hi = auc_ci(Wrr, lb)
    out = {"file": os.path.basename(path), "n": len(rows),
           "W": f"{a:.3f} [{lo:.3f}, {hi:.3f}]", "V": "—", "diff": "—"}
    if have_V:
        V = [r["V"] for r in rows]
        av, lov, hiv = auc_ci(V, lb)
        d, dlo, dhi, pgt = auc_diff_ci(Wrr, V, lb)
        out["V"] = f"{av:.3f} [{lov:.3f}, {hiv:.3f}]"
        out["diff"] = f"{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}] P={pgt:.2f}"
    return out


def main():
    DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "results")
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(DATA, "results_*.json")))
    groups = {}
    for p in paths:
        m = re.match(r"results_(v\d)_(.+)\.json", os.path.basename(p))
        ver = m.group(1) if m else "v1"
        groups.setdefault(ver, []).append(p)
    for ver in sorted(groups):
        print(f"\n### battery {ver}\n")
        print("| results file | n | W_rr [95% CI] | V [95% CI] | W − V [95% CI] |")
        print("|---|---|---|---|---|")
        for p in groups[ver]:
            r = row(p)
            print(f"| {r['file']} | {r['n']} | {r['W']} | {r['V']} | {r['diff']} |")


if __name__ == "__main__":
    main()
