"""
futurelens_exp.py — the decisive readout experiment (HANDOFF TODO 4).

The 2026-07-09 audit showed the logit-lens W_rr finds one-hop evoked bridges
(v2f 7B AUC 0.63) but is at chance on decoupled two-hop bridges (v3 ~0.50).
The binding constraint is the READOUT, not power. This trains the principled
dual of Anthropic's J-lens: per-layer affine probes h -> vocab logits whose
softmax puts mass on tokens appearing in a FUTURE window — "what is this state
pushing toward saying later" — and evaluates concept salience with it.

Leakage design: probes are trained ONLY on v1_final contexts (a different
battery, different generator run) and evaluated on v3d / v2_final. The probe
never sees labels, probes, answers, or eval contexts.

Output: results_fl_{tag}_{model}.json rows {episode, concept, label, W_fl}
(W_fl = max over trained layers of reciprocal rank of the concept's
case-variant candidate tokens at the END position of the context).
"""
import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens
from measure import concept_token_ids


def collect_train(lens, contexts, layers, window, max_positions):
    """final_norm'd hidden states (selected layers) + future-token targets.
    Features are pre-normed so a probe initialized from the model's own
    unembedding starts exactly at the logit-lens solution ("tuned future lens")."""
    feats = {L: [] for L in layers}
    tgt_rows, mask_rows = [], []
    for ctx in contexts:
        hs, ids = lens.hidden_states(ctx)
        normed = {L: lens.final_norm(hs[L]).detach().float().cpu() for L in layers}
        seq = len(ids)
        for p in range(seq - 1):
            fut = ids[p + 1: min(seq, p + 1 + window)]
            if not fut:
                continue
            row = fut + [0] * (window - len(fut))
            mask = [1.0] * len(fut) + [0.0] * (window - len(fut))
            tgt_rows.append(row); mask_rows.append(mask)
            for L in layers:
                feats[L].append(normed[L][p])
            if len(tgt_rows) >= max_positions:
                break
        if len(tgt_rows) >= max_positions:
            break
    X = {L: torch.stack(feats[L]) for L in layers}          # cpu [n, d]
    T = torch.tensor(tgt_rows, dtype=torch.long)             # [n, w]
    M = torch.tensor(mask_rows, dtype=torch.float32)         # [n, w]
    return X, T, M


def collect_eval(lens, battery, layers):
    """final_norm'd END-position states per episode + candidate token ids."""
    rows, states = [], []
    for ei, ep in enumerate(battery):
        hs, _ = lens.hidden_states(ep["context"])
        states.append({L: lens.final_norm(hs[L, -1]).detach().float().cpu() for L in layers})
        for it in ep["items"]:
            cands = concept_token_ids(lens, it["concept"])
            if cands:
                rows.append({"episode": ei, "concept": it["concept"],
                             "label": it["label"], "cands": cands, "W_fl": 0.0})
    return rows, states


def train_probe(X, T, M, d, vocab, device, steps, lr, batch, W0):
    """Tuned future lens: init from the model's own unembedding (W0) so step 0
    IS the logit-lens; small-lr Adam then shifts it toward future-window mass.
    A from-scratch 545M-param probe on ~16k positions never leaves uniform."""
    probe = nn.Linear(d, vocab).to(device)
    with torch.no_grad():
        probe.weight.copy_(W0)
        probe.bias.zero_()
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n = X.shape[0]
    Xd = X.to(device)
    Td, Md = T.to(device), M.to(device)
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=device)
        lp = F.log_softmax(probe(Xd[idx]), dim=-1)
        pick = lp.gather(1, Td[idx])                        # [b, w]
        loss = -(pick * Md[idx]).sum() / Md[idx].sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    del Xd, Td, Md, opt
    return probe.eval(), float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-battery", default="data/benchmarks/battery_v1_final.json")
    ap.add_argument("--eval", default="data/benchmarks/battery_v3d.json:v3,pilot/battery_v2_final.json:v2f",
                    help="comma list of batteryfile:tag")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-positions", type=int, default=16000)
    ap.add_argument("--n-layers", type=int, default=6)
    args = ap.parse_args()

    lens = WorkspaceLens(args.model, dtype=getattr(torch, args.dtype))
    device = lens.device
    n, d, vocab = lens.n_layers, lens.d_model, lens.vocab
    fracs = [0.3, 0.45, 0.6, 0.75, 0.9, 1.0][: args.n_layers]
    layers = sorted({max(1, round(n * f)) for f in fracs})
    mtag = args.model.split("/")[-1]
    print(f"model={mtag} layers={layers} d={d} vocab={vocab} device={device}", flush=True)

    train_ctxs = [ep["context"] for ep in json.load(open(args.train_battery))]
    X, T, M = collect_train(lens, train_ctxs, layers, args.window, args.max_positions)
    print(f"train positions={T.shape[0]} from {len(train_ctxs)} contexts "
          f"({args.train_battery})", flush=True)

    evals = []
    for spec in args.eval.split(","):
        bfile, tag = spec.split(":")
        battery = json.load(open(bfile))
        rows, states = collect_eval(lens, battery, layers)
        evals.append({"tag": tag, "rows": rows, "states": states})
        print(f"eval {tag}: {len(states)} episodes, {len(rows)} items", flush=True)

    # keep the unembedding for probe init, then free the model's VRAM
    W0 = lens.unembed.weight.detach().float().cpu()
    del lens.model
    torch.cuda.empty_cache()

    for L in layers:
        probe, loss = train_probe(X[L], T, M, d, vocab, device,
                                  args.steps, args.lr, args.batch, W0)
        with torch.no_grad():
            for ev in evals:
                for row in ev["rows"]:
                    h = ev["states"][row["episode"]][L].to(device)
                    logits = probe(h)
                    for cid in row["cands"]:
                        rank = int((logits > logits[cid]).sum().item()) + 1
                        row["W_fl"] = max(row["W_fl"], 1.0 / rank)
        print(f"layer {L}: trained (loss {loss:.3f}) + scored", flush=True)
        del probe
        torch.cuda.empty_cache()

    for ev in evals:
        out = f"data/results/results_fl_{ev['tag']}_{mtag}.json"
        rows = [{k: v for k, v in r.items() if k != "cands"} for r in ev["rows"]]
        json.dump(rows, open(out, "w"), indent=2)
        print(f"saved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
