"""
prg_calibration.py — robustness of PRG's rank calibration to the calibration
stream. Two analyses on the mixed pool (Explicit ∪ Evoked ∪ Decoupled-L),
offline from per-item scores:

  A. Calibration-size curve: percentile ranks computed from N sampled
     calibration items per route (N = 10..full), load-bearing containment@k
     averaged over resamples. Answers "how large must the running window be?"

  B. Held-out calibration: CDFs fit on a disjoint episode split (or on the
     other two benchmarks), applied to the held-out episodes. Answers "does
     transductive (same-pool) calibration inflate the result?"

Containment@k = fraction of episodes whose top-k routed set includes the
load-bearing item (the offline selection metric used in App. PRG).
"""
import os, json, bisect, argparse, random

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "data", "results")
B = os.path.join(ROOT, "data", "benchmarks")

POOL = [
    ("Explicit", "battery_v1_final.json", "results_v1f_{s}-Instruct.json"),
    ("Evoked", "battery_v2_final.json", "results_v2f_{s}-Instruct.json"),
    ("Decoupled-L", "battery_v4_xl.json", "results_v4xl_{s}-Instruct.json"),
]


@torch.no_grad()
def add_emb(episodes, device="cuda"):
    from transformers import AutoModel, AutoTokenizer
    name = "BAAI/bge-small-en-v1.5"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).to(device).eval()

    def embed(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        return F.normalize(mdl(**enc).last_hidden_state[:, 0], dim=-1)

    for ep in episodes:
        ctx = embed([ep["context"]])
        con = embed([it["concept"] for it in ep["items"]])
        for it, s in zip(ep["items"], (con @ ctx.T).squeeze(-1).tolist()):
            it["EMB"] = float(s)


def load_pool(size):
    """Episodes with items carrying W_rr, EMB, present flag, load-bearing label."""
    episodes = []
    for bench, bat_f, res_f in POOL:
        bat = json.load(open(os.path.join(B, bat_f)))
        rows = json.load(open(os.path.join(R, res_f.format(s=size))))
        by_ep = {}
        for r in rows:
            by_ep.setdefault(r["episode"], []).append(r)
        for ei, items in by_ep.items():
            ctx = bat[ei]["context"].lower()
            eps_items = []
            for it in items:
                eps_items.append({
                    "concept": it["concept"],
                    "W_rr": it["W_rr"],
                    "present": ctx.rfind(it["concept"].lower()) >= 0,
                    "load": it["label"] == "load_bearing",
                })
            episodes.append({"bench": bench, "context": bat[ei]["context"],
                             "items": eps_items})
    return episodes


def prank(pop, x):
    return bisect.bisect_left(pop, x) / max(1, len(pop) - 1)


def containment(episodes, cal_w, cal_e, k=2):
    cal_w, cal_e = sorted(cal_w), sorted(cal_e)
    hit = 0
    for ep in episodes:
        scored = [(prank(cal_e, it["EMB"]) if it["present"]
                   else prank(cal_w, it["W_rr"]), it["load"]) for it in ep["items"]]
        scored.sort(key=lambda t: -t[0])
        hit += any(load for _, load in scored[:k])
    return hit / len(episodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="0.5B,7B")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--resamples", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = {}
    for s in args.sizes.split(","):
        episodes = load_pool(s)
        add_emb(episodes, args.device)
        w_all = [it["W_rr"] for ep in episodes for it in ep["items"] if not it["present"]]
        e_all = [it["EMB"] for ep in episodes for it in ep["items"] if it["present"]]
        full = containment(episodes, w_all, e_all, args.k)
        print(f"\n== {s}: {len(episodes)} episodes, route pops |W|={len(w_all)} "
              f"|E|={len(e_all)}, full-pool containment@{args.k} = {full:.3f}")
        out[s] = {"full": full, "curve": {}, "heldout": {}}

        # A. calibration-size curve
        rng = random.Random(0)
        for n in [10, 25, 50, 100, 200]:
            accs = []
            for _ in range(args.resamples):
                cw = rng.sample(w_all, min(n, len(w_all)))
                ce = rng.sample(e_all, min(n, len(e_all)))
                accs.append(containment(episodes, cw, ce, args.k))
            m = sum(accs) / len(accs)
            lo = sorted(accs)[int(0.025 * len(accs))]
            hi = sorted(accs)[int(0.975 * len(accs))]
            out[s]["curve"][n] = (m, lo, hi)
            print(f"  N={n:4d} per route: containment {m:.3f} [{lo:.3f}, {hi:.3f}]")

        # B1. split-half by episodes (calibrate on half, evaluate on other half)
        accs = []
        for rep in range(args.resamples):
            rr = random.Random(1000 + rep)
            idx = list(range(len(episodes)))
            rr.shuffle(idx)
            half = len(idx) // 2
            cal_eps = [episodes[i] for i in idx[:half]]
            ev_eps = [episodes[i] for i in idx[half:]]
            cw = [it["W_rr"] for ep in cal_eps for it in ep["items"] if not it["present"]]
            ce = [it["EMB"] for ep in cal_eps for it in ep["items"] if it["present"]]
            accs.append(containment(ev_eps, cw, ce, args.k))
        m = sum(accs) / len(accs)
        out[s]["heldout"]["split_half"] = m
        print(f"  split-half calibration: containment {m:.3f}")

        # B2. cross-benchmark: calibrate on Explicit+Evoked, evaluate Decoupled-L
        cal_eps = [ep for ep in episodes if ep["bench"] != "Decoupled-L"]
        ev_eps = [ep for ep in episodes if ep["bench"] == "Decoupled-L"]
        cw = [it["W_rr"] for ep in cal_eps for it in ep["items"] if not it["present"]]
        ce = [it["EMB"] for ep in cal_eps for it in ep["items"] if it["present"]]
        c_cross = containment(ev_eps, cw, ce, args.k)
        cw_t = [it["W_rr"] for ep in ev_eps for it in ep["items"] if not it["present"]]
        ce_t = [it["EMB"] for ep in ev_eps for it in ep["items"] if it["present"]]
        c_trans = containment(ev_eps, cw_t, ce_t, args.k)
        out[s]["heldout"]["cross_bench"] = (c_cross, c_trans)
        print(f"  cross-benchmark calibration (fit Explicit+Evoked, eval Decoupled-L): "
              f"{c_cross:.3f} vs transductive {c_trans:.3f}")

    with open(os.path.join(R, "prg_calibration.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved data/results/prg_calibration.json")


if __name__ == "__main__":
    main()
