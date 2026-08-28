"""
layer_trace.py — for a chosen episode, dump the reciprocal-rank grid of the
bridge concept's first token across (layer x token position) under the logit
lens. Feeds the mechanistic 'where the workspace lights up' heatmap.
Also dumps, at the END position, the per-layer reciprocal rank of every
candidate (for the ranking-flow figure).
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens
from experiments.measure import concept_token_ids


@torch.no_grad()
def rr_grid(lens, context, concept_ids):
    """Return [n_layers+1, seq] reciprocal rank of the best concept-variant
    token at each (layer, position); plus the token strings."""
    hs, ids = lens.hidden_states(context)           # hs [L+1, seq, d]
    L, seq, _ = hs.shape
    grid = np.zeros((L, seq), dtype=np.float32)
    for li in range(L):
        logits = lens.logitlens(hs[li])             # [seq, vocab]
        ranks = (logits > logits.gather(1, torch.tensor(
            [[concept_ids[0]]] * seq, device=logits.device)).expand_as(logits)).sum(1)
        # best over concept variants
        best = None
        for cid in concept_ids:
            r = (logits > logits[:, cid:cid+1]).sum(1) + 1   # rank (1-based)
            rr = 1.0 / r.float()
            best = rr if best is None else torch.maximum(best, rr)
        grid[li] = best.cpu().numpy()
    toks = [lens.tok.decode([i]) for i in ids]
    return grid, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--episodes", required=True, help="comma-separated indices")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lens = WorkspaceLens(args.model, dtype=torch.bfloat16)
    battery = json.load(open(args.battery))
    out = {}
    for ep in [int(x) for x in args.episodes.split(",")]:
        b = battery[ep]
        bridge = [it["concept"] for it in b["items"]
                  if it["label"] == "load_bearing"][0]
        cids = concept_token_ids(lens, bridge)
        grid, toks = rr_grid(lens, b["context"], cids)
        # per-candidate end-position per-layer rr (for ranking flow)
        cand_rr = {}
        hs, _ = lens.hidden_states(b["context"])
        for it in b["items"]:
            c_ids = concept_token_ids(lens, it["concept"])
            per_layer = []
            for li in range(hs.shape[0]):
                logits = lens.logitlens(hs[li][-1])            # [vocab] end pos
                best = max(1.0 / (int((logits > logits[cid]).sum()) + 1)
                           for cid in c_ids)
                per_layer.append(best)
            cand_rr[it["concept"]] = {"label": it["label"],
                                      "rr_by_layer": per_layer,
                                      "peak_rr": float(max(per_layer))}
        out[str(ep)] = {"bridge": bridge, "question": b["probe_question"],
                        "answer": b["answer"], "tokens": toks,
                        "grid": grid.tolist(), "cand_rr": cand_rr}
        print(f"ep{ep}: bridge={bridge} grid={grid.shape}", flush=True)
    json.dump(out, open(args.out, "w"))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
