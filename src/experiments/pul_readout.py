"""
pul_readout.py — W_J, the LEVERAGE member of the workspace-availability family
(THEORY.md §2): the Prospective Utility Lens.

Salience of concept c at the end-of-encoding state = the directional derivative
of expected FUTURE answering utility along c's INPUT-side read-direction:

  S_c = max_{L in band}  cos( E_q[ grad_{h_L} log p(a_q | context, q) ],  v_c^L )

where q ranges over a fixed probe-blind wh-question bank, a_q is the model's own
greedy answer (pathwise gradient, answer held fixed), and v_c^L is the RepE
diff-of-means concept vector (jlens.concept_vector) — the direction our causal
experiments PROVED the network reads (real flips 0.60-0.85 vs sham 0.05-0.35).

This is the tangent at eps=0 of the causal patch we already published (the
secant), marginalized over futures: salience = infinitesimal causal leverage.
Training-free; the gradient is computed ONCE per (episode, question, layer) and
every concept is scored by a dot product — cost independent of #concepts.
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens
from patch import _layer_modules

WH_BANK = [
    "What is the most important fact to take away from this?",
    "Where did this take place?",
    "Who is involved and what defines them?",
    "Why did events unfold the way they did?",
    "What will this person most likely need next?",
    "What language, currency, or customs are relevant here?",
    "What underlying situation explains the details?",
    "What would an expert infer from this that is not stated?",
    "What should be remembered to answer questions later?",
    "What category of thing is central here?",
    "How would you summarize the essential point in one word?",
    "What follows from this?",
]

TEMPLATES = [
    "The key concept here is {c}.",
    "Everything in this story points to {c}.",
    "She kept thinking about {c}.",
    "The answer to the riddle is {c}.",
]
NEUTRAL = ["thing", "stuff", "matter", "topic"]


def concept_vectors(lens, concepts, layers):
    """Input-side RepE directions per concept per layer (unit norm), cached."""
    vecs = {}
    with torch.no_grad():
        neg_states = {}
        for L in layers:
            accs = []
            for w in NEUTRAL:
                for t in TEMPLATES:
                    hs, _ = lens.hidden_states(t.format(c=w))
                    accs.append(hs[L, -1].float())
            neg_states[L] = torch.stack(accs).mean(0)
        for c in concepts:
            states = {L: [] for L in layers}
            for t in TEMPLATES:
                hs, _ = lens.hidden_states(t.format(c=c))
                for L in layers:
                    states[L].append(hs[L, -1].float())
            vecs[c] = {}
            for L in layers:
                v = torch.stack(states[L]).mean(0) - neg_states[L]
                vecs[c][L] = v / (v.norm() + 1e-8)
    return vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-questions", type=int, default=12)
    ap.add_argument("--ans-tokens", type=int, default=6)
    args = ap.parse_args()

    lens = WorkspaceLens(args.model, dtype=getattr(torch, args.dtype))
    n = lens.n_layers
    band = sorted({max(1, round(n * f)) for f in [0.3, 0.45, 0.6, 0.75]})  # causal band
    battery = json.load(open(args.battery))
    Q = WH_BANK[: args.n_questions]
    embed = lens.model.get_input_embeddings()
    print(f"model={args.model} episodes={len(battery)} band={band} |Q|={len(Q)} "
          f"device={lens.device}", flush=True)

    for p in lens.model.parameters():
        p.requires_grad_(False)

    rows = []
    for ei, ep in enumerate(battery):
        concepts = [it["concept"] for it in ep["items"]]
        vecs = concept_vectors(lens, concepts, band)

        # accumulate E_q[ grad_{h_L} log p(a_q | ctx, q) ] at END-OF-CONTEXT position
        grad_sum = {L: None for L in band}
        ctx_ids = lens.tok(ep["context"], return_tensors="pt").to(lens.device)
        ctx_len = ctx_ids["input_ids"].shape[1]
        for q in Q:
            full = ep["context"] + f"\n\nQuestion: {q}\nAnswer:"
            enc = lens.tok(full, return_tensors="pt").to(lens.device)
            with torch.no_grad():
                gen = lens.model.generate(**enc, max_new_tokens=args.ans_tokens,
                                          do_sample=False,
                                          pad_token_id=lens.tok.pad_token_id)
            ans_ids = gen[0][enc["input_ids"].shape[1]:]
            if len(ans_ids) == 0:
                continue
            seq = torch.cat([enc["input_ids"][0], ans_ids]).unsqueeze(0)
            # grad-enabled forward: make the embedding output a graph leaf
            captured = {}
            hooks = []
            def mk_embed_hook():
                def hook(mod, inp, out):
                    out = out.detach().requires_grad_(True)
                    captured["emb"] = out
                    return out
                return hook
            blocks = _layer_modules(lens.model)
            def mk_block_hook(L):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h.retain_grad()
                    captured[L] = h
                    return out
                return hook
            hooks.append(embed.register_forward_hook(mk_embed_hook()))
            for L in band:
                hooks.append(blocks[L - 1].register_forward_hook(mk_block_hook(L)))
            try:
                out = lens.model(input_ids=seq)
                logits = out.logits[0]
                n_ans = len(ans_ids)
                lp = F.log_softmax(logits[-n_ans - 1:-1].float(), dim=-1)
                loss = lp.gather(1, ans_ids.unsqueeze(1)).sum()
                loss.backward()
                for L in band:
                    g = captured[L].grad[0, ctx_len - 1].float()
                    grad_sum[L] = g if grad_sum[L] is None else grad_sum[L] + g
            finally:
                for h in hooks:
                    h.remove()
            lens.model.zero_grad(set_to_none=True)

        for it in ep["items"]:
            c = it["concept"]
            s = max(F.cosine_similarity(grad_sum[L], vecs[c][L], dim=0).item()
                    for L in band if grad_sum[L] is not None)
            rows.append({"episode": ei, "concept": c, "label": it["label"],
                         "W_pul": s})
        if (ei + 1) % 5 == 0:
            print(f"  {ei+1}/{len(battery)}", flush=True)

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"saved {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
