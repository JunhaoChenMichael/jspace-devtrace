"""
replay_readout.py — W_rep, the REHEARSAL member of the workspace-availability
family (THEORY.md §1): hippocampal-replay-style reactivation.

Probe-blind protocol: encode the RAW context only (no chat template, no probe),
then let the model free-run K sampled continuations ("offline replay / mind-
wandering"). A concept's salience is its REACTIVATION across rollouts:

  W_rep_emit : fraction of rollouts that EMIT the concept (any case-variant
               first token appears among generated ids) — surface reactivation.
  W_rep_dec  : mean over rollouts of the peak case-variant reciprocal rank of
               the concept across CONTINUATION positions and band layers —
               counts silent (decoded but unemitted) reactivation too.

Pre-registered predictions (THEORY.md §6): W_rep > W_rr on two-hop/weakly-evoked
bridges (v3), where every static readout — logit-lens AND trained future-lens —
sits at chance, because replay gives the model tokens of room to actually
perform the composition; non-regression on v2f/v4 evoked bridges.
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
    ap.add_argument("--k", type=int, default=8, help="rollouts per episode")
    ap.add_argument("--n-new", type=int, default=40, help="tokens per rollout")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--n-layers", type=int, default=6)
    args = ap.parse_args()

    lens = WorkspaceLens(args.model, dtype=getattr(torch, args.dtype))
    n = lens.n_layers
    fracs = [0.3, 0.45, 0.6, 0.75, 0.9, 1.0][: args.n_layers]
    layers = sorted({max(1, round(n * f)) for f in fracs})
    battery = json.load(open(args.battery))
    print(f"model={args.model} episodes={len(battery)} K={args.k} N={args.n_new} "
          f"layers={layers} device={lens.device}", flush=True)

    rows = []
    for ei, ep in enumerate(battery):
        torch.manual_seed(1000 + ei)  # reproducible replay per episode
        enc = lens.tok(ep["context"], return_tensors="pt").to(lens.device)
        ctx_len = enc["input_ids"].shape[1]
        gen = lens.model.generate(
            **enc, max_new_tokens=args.n_new, do_sample=True,
            temperature=args.temperature, top_p=args.top_p,
            num_return_sequences=args.k, pad_token_id=lens.tok.pad_token_id)

        cands = {}
        for it in ep["items"]:
            c = concept_token_ids(lens, it["concept"])
            if c:
                cands[it["concept"]] = c

        emit = {c: 0 for c in cands}
        dec = {c: 0.0 for c in cands}
        for r in range(args.k):
            seq = gen[r].unsqueeze(0)
            cont_ids = set(gen[r][ctx_len:].tolist())
            for c, ids in cands.items():
                if any(i in cont_ids for i in ids):
                    emit[c] += 1
            out = lens.model(input_ids=seq, output_hidden_states=True)
            hs = out.hidden_states  # tuple of [1, seq, d]
            best = {c: 0.0 for c in cands}
            for L in layers:
                h = hs[L][0, ctx_len:]                      # continuation positions
                logits = lens.logitlens(h).float()          # [cont, vocab]
                for c, ids in cands.items():
                    for cid in ids:
                        col = logits[:, cid]
                        ranks = (logits > col.unsqueeze(1)).sum(dim=1) + 1
                        rr = (1.0 / ranks.float()).max().item()
                        if rr > best[c]:
                            best[c] = rr
            for c in cands:
                dec[c] += best[c]
            del out, hs

        for it in ep["items"]:
            c = it["concept"]
            if c not in cands:
                continue
            rows.append({"episode": ei, "concept": c, "label": it["label"],
                         "W_rep_emit": emit[c] / args.k,
                         "W_rep_dec": dec[c] / args.k})
        if (ei + 1) % 5 == 0:
            print(f"  {ei+1}/{len(battery)}", flush=True)

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"saved {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
