"""
master_table.py — paper-grade statistics in one place (gap-fill 5).

For every key results file it reports W_rr / V AUCs with BOTH item-level and
EPISODE-CLUSTER bootstrap CIs (the cluster CI is the honest default: items within
an episode share a context), the paired W−V diff (cluster CI), and — for the four
pre-registered primary contrasts (v4 W−V at 0.5B/1.5B/3B/7B) — Bonferroni-adjusted
cluster CIs at alpha = 0.05/4. Optionally (--embed) computes bge cosine AUC for
every cell and archives it. Output: pilot/master_table.md
"""
import os, sys, json, glob, argparse
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "data", "results")
BENCH = os.path.join(ROOT, "data", "benchmarks")
HERE = RESULTS


def load(path):
    return json.load(open(path))


def cluster_boot(rows, key_a, key_b=None, B=4000, seed=0, alpha=0.05):
    """Episode-cluster bootstrap. Returns (stat, lo, hi, p_gt0) where stat is
    AUC(key_a) if key_b is None else AUC(key_a) - AUC(key_b)."""
    eps = {}
    for r in rows:
        eps.setdefault(r["episode"], []).append(r)
    ep_ids = list(eps.keys())
    rng = np.random.default_rng(seed)

    def stat_of(sample_rows):
        y = np.array([r["label"] == "load_bearing" for r in sample_rows], dtype=int)
        if y.sum() == 0 or y.sum() == len(y):
            return None
        a = roc_auc_score(y, [r[key_a] for r in sample_rows])
        if key_b is None:
            return a
        return a - roc_auc_score(y, [r[key_b] for r in sample_rows])

    point = stat_of(rows)
    boots = []
    for _ in range(B):
        pick = rng.choice(len(ep_ids), len(ep_ids), replace=True)
        sample = [r for i in pick for r in eps[ep_ids[i]]]
        s = stat_of(sample)
        if s is not None:
            boots.append(s)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi, float(np.mean(boots > 0))


def item_boot(rows, key, B=2000, seed=0):
    y = np.array([r["label"] == "load_bearing" for r in rows], dtype=int)
    s = np.array([r[key] for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() in (0, len(y)):
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return roc_auc_score(y, s), lo, hi


def embed_scores(rows, battery, device="cuda"):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    mdl = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(device).eval()

    @torch.no_grad()
    def emb(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        return F.normalize(mdl(**enc).last_hidden_state[:, 0], dim=-1)

    ctx = {i: emb([ep["context"]]) for i, ep in enumerate(battery)}
    for r in rows:
        r["EMB"] = float((emb([r["concept"]]) @ ctx[r["episode"]].T).item())
    del mdl
    import torch as _t
    _t.cuda.empty_cache()
    return rows


CELLS = [
    # (results file, battery file, label)
    ("results_v4f_{s}-Instruct.json", "battery_v4_final.json", "v4"),
    ("results_v3f_{s}-Instruct.json", "battery_v3d.json", "v3"),
    ("results_v2f_{s}-Instruct.json", "battery_v2_final.json", "v2f"),
    ("results_v1f_{s}-Instruct.json", "battery_v1_final.json", "v1f"),
]
SIZES = ["0.5B", "1.5B", "3B", "7B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="also compute bge AUC per cell (GPU)")
    ap.add_argument("--out", default=os.path.join(HERE, "master_table.md"))
    args = ap.parse_args()

    lines = ["# Master statistics table (episode-cluster bootstrap, B=4000, seed 0)",
             "",
             "Cluster CIs are the honest default (items share episode contexts).",
             "Primary contrasts (pre-registered): v4 W−V at each size, Bonferroni alpha=0.05/4.",
             ""]
    for tmpl, bfile, tag in CELLS:
        lines += [f"## battery {tag}", "",
                  "| size | W_rr [cluster CI] | V [cluster CI] | W−V [cluster CI] P(>0)"
                  + (" | embed AUC | W−embed [cluster CI]" if args.embed else "") + " |",
                  "|" + "---|" * (6 if args.embed else 4)]
        battery = load(os.path.join(BENCH, bfile)) if args.embed else None
        for s in SIZES:
            path = os.path.join(HERE, tmpl.format(s=s))
            if not os.path.exists(path):
                continue
            rows = load(path)
            have_V = all("V" in r for r in rows)
            alpha = 0.05 / 4 if tag == "v4" else 0.05
            w, wlo, whi, _ = cluster_boot(rows, "W_rr", alpha=0.05)
            cell = [f"{s}", f"{w:.3f} [{wlo:.3f}, {whi:.3f}]"]
            if have_V:
                v, vlo, vhi, _ = cluster_boot(rows, "V", alpha=0.05)
                d, dlo, dhi, pgt = cluster_boot(rows, "W_rr", "V", alpha=alpha)
                star = " *" if dlo > 0 else ""
                adj = " (Bonf)" if tag == "v4" else ""
                cell += [f"{v:.3f} [{vlo:.3f}, {vhi:.3f}]",
                         f"{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{adj} P={pgt:.3f}{star}"]
            else:
                cell += ["—", "—"]
            if args.embed:
                rows = embed_scores(rows, battery)
                e, elo, ehi, _ = cluster_boot(rows, "EMB", alpha=0.05)
                de, delo, dehi, pge = cluster_boot(rows, "W_rr", "EMB", alpha=0.05)
                cell += [f"{e:.3f} [{elo:.3f}, {ehi:.3f}]",
                         f"{de:+.3f} [{delo:+.3f}, {dehi:+.3f}] P={pge:.3f}"]
            lines.append("| " + " | ".join(cell) + " |")
        lines.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
