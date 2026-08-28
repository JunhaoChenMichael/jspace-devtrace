"""
ignition_readout.py — W_ig, the BROADCAST member of the workspace-availability
family (THEORY.md §1): Global-Neuronal-Workspace-style ignition.

A concept is "in the workspace" under GNW not when it peaks at one (layer,
position) but when it is GLOBALLY AVAILABLE: decodable broadly across positions
after first appearing, rising sharply into a sustained late-layer plateau.

Per concept, over the (layer x position) reciprocal-rank grid of the context:
  breadth     : fraction of positions in the last half of the context whose
                late-layer readout ranks the concept in the top-100
  persistence : mean reciprocal rank over (late layers x last-half positions)
  sharpness   : max layer-to-layer increase of the position-max RR profile
  W_ig = breadth * persistence^(1/2) * (1 + sharpness)   (monotone composite)

Static like W_rr (same forward pass), so it isolates whether the GNW
AGGREGATION — rather than dynamics — closes any gap.
"""
import os, sys, json, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens
from measure import concept_token_ids


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    lens = WorkspaceLens(args.model, dtype=getattr(torch, args.dtype))
    n = lens.n_layers
    late = [L for L in range(1, n + 1) if L >= round(0.6 * n)]
    battery = json.load(open(args.battery))
    print(f"model={args.model} episodes={len(battery)} late_layers={late[0]}..{late[-1]} "
          f"device={lens.device}", flush=True)

    rows = []
    for ei, ep in enumerate(battery):
        hs, ids = lens.hidden_states(ep["context"])
        seq = len(ids)
        half = seq // 2
        cands = {}
        for it in ep["items"]:
            c = concept_token_ids(lens, it["concept"])
            if c:
                cands[it["concept"]] = c

        # rr[c][L] = per-position reciprocal rank vector at layer L (last half)
        prof = {c: {} for c in cands}
        for L in range(1, n + 1):
            logits = lens.logitlens(hs[L, half:]).float()      # [pos, vocab]
            for c, cids in cands.items():
                best = None
                for cid in cids:
                    col = logits[:, cid]
                    ranks = (logits > col.unsqueeze(1)).sum(dim=1) + 1
                    rr = 1.0 / ranks.float()
                    best = rr if best is None else torch.maximum(best, rr)
                prof[c][L] = best

        for it in ep["items"]:
            c = it["concept"]
            if c not in cands:
                continue
            late_stack = torch.stack([prof[c][L] for L in late])   # [lateL, pos]
            breadth = (late_stack.max(dim=0).values > 1.0 / args.topk).float().mean().item()
            persistence = late_stack.mean().item()
            layer_max = torch.tensor([prof[c][L].max() for L in range(1, n + 1)])
            sharpness = torch.diff(layer_max).max().item() if n > 1 else 0.0
            w_ig = breadth * (persistence ** 0.5) * (1.0 + max(0.0, sharpness))
            rows.append({"episode": ei, "concept": c, "label": it["label"],
                         "breadth": round(breadth, 4),
                         "persistence": round(persistence, 6),
                         "sharpness": round(sharpness, 4),
                         "W_ig": w_ig})
        if (ei + 1) % 10 == 0:
            print(f"  {ei+1}/{len(battery)}", flush=True)

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"saved {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
