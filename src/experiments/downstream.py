"""
downstream.py — the payoff experiment.

Turns "workspace salience predicts utility better than verbal reflection" into a
hard task metric: a memory-BUDGETED agent must choose which items to keep, then
answer a probe using ONLY what it kept. We compare selection policies.

Setup (reuses the pilot exactly):
  - Each episode has candidate items (concepts) with per-item workspace salience
    W_rr and verbal-reflection salience V (already computed in results_*.json),
    plus a probe_question and gold answer (battery).
  - A policy ranks the items; we keep the top-k (the memory budget).
  - At recall, the model sees ONLY the kept concept words + the probe, and answers.
  - Accuracy = did the answer contain the gold answer.

Policies:
  workspace : rank by W_rr (store what entered the workspace)
  verbal    : rank by V   (store what the model SAYS is important)
  embedding : rank by cosine(concept, context) under a sentence embedder
              (bge-small-en-v1.5) — the "just use RAG relevance" reviewer baseline
  recency   : rank by last literal occurrence in the context (absent -> last) —
              the "just keep the most recent stuff" reviewer baseline
  random    : random k (seeded, averaged)
  oracle    : load_bearing items first (upper bound on selection)
References (budget-independent):
  full_context : answer from the whole passage  (ceiling)
  no_memory    : answer from the probe alone     (floor / prior)

Significance: per-episode correctness is logged for EVERY condition, and an
exact McNemar test (paired, same episodes) is run for workspace vs each rival
policy at each budget, plus each policy vs the no_memory floor.

Prediction: workspace beats verbal on the v2 silent-bridge battery (self-report
is fooled there), especially at small k; they tie on v1 (explicit info).
"""
import os, sys, json, argparse, re
import torch
import torch.nn.functional as F
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens

STOP = set("the a an of to in on at for and or but with without is are was were be been "
           "this that these those it its her his their your my our".split())


@torch.no_grad()
def generate(lens, prompt, cache=None, max_new_tokens=48):
    if cache is not None and prompt in cache:
        return cache[prompt]
    msgs = [{"role": "user", "content": prompt}]
    text = lens.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = lens.tok(text, return_tensors="pt").to(lens.device)
    out = lens.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=lens.tok.pad_token_id)
    gen = out[0][enc["input_ids"].shape[1]:]
    ans = lens.tok.decode(gen, skip_special_tokens=True).strip()
    if cache is not None:
        cache[prompt] = ans
    return ans


def normalize(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def grade(output, gold):
    o, g = normalize(output), normalize(gold)
    if g and g in o:
        return True
    # else: any content word of the gold answer present
    words = [w for w in g.split() if len(w) >= 3 and w not in STOP]
    return any(w in o for w in words) if words else (g in o)


def recall_prompt(concepts, probe):
    mem = ", ".join(concepts) if concepts else "(nothing)"
    return (f"Earlier you read a passage. The only things you remember from it are: "
            f"{mem}.\nUsing only what you remember, answer concisely.\nQuestion: {probe}\nAnswer:")


def full_prompt(context, probe):
    return f"{context}\nAnswer concisely.\nQuestion: {probe}\nAnswer:"


def nomem_prompt(probe):
    return f"Answer this question concisely.\nQuestion: {probe}\nAnswer:"


def rank(items, policy, seed=0):
    """Return items ordered best-first under a policy."""
    if policy == "workspace":
        return sorted(items, key=lambda x: x["W_rr"], reverse=True)
    if policy == "verbal":
        return sorted(items, key=lambda x: x.get("V", 0.0), reverse=True)
    if policy == "embedding":
        return sorted(items, key=lambda x: x.get("EMB", 0.0), reverse=True)
    if policy == "recency":
        return sorted(items, key=lambda x: x.get("REC", -1), reverse=True)
    if policy == "absent":
        # trivial surface heuristic: keep context-absent items first (REC<0),
        # tie-break by embedding relevance
        return sorted(items, key=lambda x: (x.get("REC", -1) < 0,
                                            x.get("EMB", 0.0)), reverse=True)
    if policy == "absentv":
        # absence routing with the verbal probe as tie-break
        return sorted(items, key=lambda x: (x.get("REC", -1) < 0,
                                            x.get("V", 0.0)), reverse=True)
    if policy == "absentw":
        # PRG's inferred route made explicit: provenance narrows to the absent
        # set, the workspace readout discriminates within it
        return sorted(items, key=lambda x: (x.get("REC", -1) < 0,
                                            x["W_rr"]), reverse=True)
    if policy == "ll":
        return sorted(items, key=lambda x: x.get("LL", -99.0), reverse=True)
    if policy == "fusion":
        return sorted(items, key=lambda x: x.get("FUSION", 0.0), reverse=True)
    if policy == "routed":
        return sorted(items, key=lambda x: x.get("ROUTED", -99.0), reverse=True)
    if policy == "oracle":
        return sorted(items, key=lambda x: x["label"] == "load_bearing", reverse=True)
    if policy == "random":
        # NOTE: an earlier hash-based shuffle was an IDENTITY permutation at seed=0,
        # and battery item order is label-correlated (load_bearing often listed
        # first), which contaminated the random baseline with a semi-oracle.
        # random.Random(seed) is deterministic and actually shuffles.
        import random as _random
        idx = list(range(len(items)))
        _random.Random(seed).shuffle(idx)
        return [items[i] for i in idx]
    return items


# ---------- reviewer baselines: cheap salience scores added onto each item ----------

@torch.no_grad()
def add_embedding_scores(by_ep, battery, embed_model, device):
    """EMB = cosine(concept, context) under a sentence embedder (CLS pooling)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(embed_model)
    mdl = AutoModel.from_pretrained(embed_model).to(device).eval()

    def embed(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        out = mdl(**enc).last_hidden_state[:, 0]      # CLS (bge-style)
        return F.normalize(out, dim=-1)

    for ei, items in by_ep.items():
        ctx_emb = embed([battery[ei]["context"]])
        con_emb = embed([it["concept"] for it in items])
        sims = (con_emb @ ctx_emb.T).squeeze(-1).tolist()
        for it, s in zip(items, sims):
            it["EMB"] = float(s)
    del mdl
    if device == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def add_ll_scores(by_ep, battery, lens):
    """LL = length-normalized log p(concept | context) under the subject model
    (one extra forward pass per candidate; the conditional-likelihood baseline)."""
    for ei, items in by_ep.items():
        ctx_ids = lens.tok(battery[ei]["context"],
                           return_tensors="pt").input_ids.to(lens.device)
        for it in items:
            c_ids = lens.tok(" " + it["concept"], add_special_tokens=False,
                             return_tensors="pt").input_ids.to(lens.device)
            full = torch.cat([ctx_ids, c_ids], dim=1)
            logits = lens.model(full).logits[0]
            lp, start = 0.0, ctx_ids.shape[1]
            for j in range(c_ids.shape[1]):
                lp += F.log_softmax(logits[start - 1 + j].float(), dim=-1)[
                    c_ids[0, j]].item()
            it["LL"] = lp / c_ids.shape[1]


def add_recency_scores(by_ep, battery):
    """REC = last occurrence position of the concept in the context; absent -> -1."""
    for ei, items in by_ep.items():
        ctx = battery[ei]["context"].lower()
        for it in items:
            it["REC"] = ctx.rfind(it["concept"].lower())


def add_routed_scores(by_ep, battery):
    """PRG (provenance-routed gating): an item absent from the source text is
    scored by the workspace readout (the only informative channel there); an
    item present in the text is scored by embedding relevance (the stronger
    channel for surface content). Each route is converted to its population
    PERCENTILE RANK over the run, so the two routes have identical uniform
    marginals and are directly comparable (z-scores are not, because the two
    raw distributions differ in shape)."""
    import bisect
    absent_w, present_e = [], []
    for items in by_ep.values():
        for it in items:
            if it.get("REC", -1) < 0:
                absent_w.append(it["W_rr"])
            else:
                present_e.append(it.get("EMB", 0.0))
    absent_w.sort(); present_e.sort()
    for items in by_ep.values():
        for it in items:
            if it.get("REC", -1) < 0:
                it["ROUTED"] = bisect.bisect_left(absent_w, it["W_rr"]) / max(1, len(absent_w) - 1)
            else:
                it["ROUTED"] = bisect.bisect_left(present_e, it.get("EMB", 0.0)) / max(1, len(present_e) - 1)


def add_fusion_scores(by_ep, battery):
    """FUSION = mean within-episode percentile rank of {recency, frequency,
    embedding-relevance} — a surface-feature admission-control proxy in the
    spirit of A-MAC / multi-factor value models (no internals)."""
    for ei, items in by_ep.items():
        ctx = battery[ei]["context"].lower()
        feats = []
        for it in items:
            it["FREQ"] = ctx.count(it["concept"].lower())
            feats.append((it.get("REC", -1), it["FREQ"], it.get("EMB", 0.0)))
        n = len(items)
        for k in range(3):
            order = sorted(range(n), key=lambda i: feats[i][k])
            for rank, i in enumerate(order):
                items[i][f"_pr{k}"] = rank / max(1, n - 1)
        for it in items:
            it["FUSION"] = (it["_pr0"] + it["_pr1"] + it["_pr2"]) / 3


def mcnemar(a, b):
    """Exact McNemar on paired booleans: returns (b01, b10, p).
    b01 = a correct & b wrong (a wins), b10 = a wrong & b correct."""
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    p = binomtest(min(b01, b10), n, 0.5).pvalue if n > 0 else 1.0
    return b01, b10, float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--results", required=True, help="results_*.json with per-item W_rr,V,label")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--budgets", default="1,2")
    ap.add_argument("--policies", default="workspace,verbal,embedding,recency,random,oracle")
    ap.add_argument("--seeds", type=int, default=3, help="random-policy averaging")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max-new-tokens", type=int, default=48,
                    help="generation budget; two-hop recall composition needs >24")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = json.load(open(args.results))
    battery = json.load(open(args.battery))
    have_V = all("V" in r for r in rows)
    budgets = [int(b) for b in args.budgets.split(",")]
    policies = args.policies.split(",")
    if "verbal" in policies and not have_V:
        policies = [p for p in policies if p != "verbal"]

    # group items by episode
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["episode"], []).append(r)

    lens = WorkspaceLens(args.model, dtype=getattr(torch, args.dtype))
    print(f"model={args.model} episodes={len(by_ep)} policies={policies} budgets={budgets} "
          f"verbal={have_V} max_new_tokens={args.max_new_tokens} device={lens.device}", flush=True)

    if "embedding" in policies or "absent" in policies:
        add_embedding_scores(by_ep, battery, args.embed_model, lens.device)
    if "ll" in policies:
        add_ll_scores(by_ep, battery, lens)
    if "recency" in policies or "fusion" in policies or "absent" in policies or "absentw" in policies or "absentv" in policies:
        add_recency_scores(by_ep, battery)
    if "fusion" in policies or "routed" in policies:
        if not all("EMB" in r for rs in by_ep.values() for r in rs):
            add_embedding_scores(by_ep, battery, args.embed_model, lens.device)
        if "fusion" in policies:
            add_fusion_scores(by_ep, battery)
    if "routed" in policies:
        if not all("REC" in r for rs in by_ep.values() for r in rs):
            add_recency_scores(by_ep, battery)
        add_routed_scores(by_ep, battery)

    results = {"model": args.model, "results_file": args.results, "battery": args.battery,
               "per_condition": {}, "refs": {}, "per_episode": {}, "mcnemar": {}}
    cache = {}  # identical prompts across conditions answer identically (greedy decode)

    # references (budget-independent), with per-episode records
    for ref in ["full_context", "no_memory"]:
        recs = []
        for ei, items in sorted(by_ep.items()):
            ep = battery[ei]
            p = full_prompt(ep["context"], ep["probe_question"]) if ref == "full_context" \
                else nomem_prompt(ep["probe_question"])
            ans = generate(lens, p, cache, args.max_new_tokens)
            recs.append({"episode": ei, "answer": ans, "correct": bool(grade(ans, ep["answer"]))})
        results["refs"][ref] = round(sum(r["correct"] for r in recs) / len(recs), 3)
        results["per_episode"][ref] = recs
        print(f"  [ref] {ref:12s} acc={results['refs'][ref]:.3f}", flush=True)

    # policy x budget, with per-episode records (random: seed 0's records kept)
    for policy in policies:
        for k in budgets:
            accs = []
            reps = args.seeds if policy == "random" else 1
            for s in range(reps):
                recs = []
                for ei, items in sorted(by_ep.items()):
                    ep = battery[ei]
                    ordered = rank(items, policy, seed=s)
                    kept = [x["concept"] for x in ordered[:k]]
                    ans = generate(lens, recall_prompt(kept, ep["probe_question"]), cache,
                                   args.max_new_tokens)
                    recs.append({"episode": ei, "kept": kept, "answer": ans,
                                 "correct": bool(grade(ans, ep["answer"]))})
                accs.append(sum(r["correct"] for r in recs) / len(recs))
                if s == 0:
                    results["per_episode"][f"{policy}@{k}"] = recs
            acc = sum(accs) / len(accs)
            results["per_condition"][f"{policy}@{k}"] = round(acc, 3)
            print(f"  {policy:10s} k={k}  acc={acc:.3f}", flush=True)

    # paired significance on the SAME episodes: exact McNemar
    def corr(cond):
        return [r["correct"] for r in results["per_episode"][cond]]
    rivals = [p for p in policies if p not in ("workspace", "random")]
    for k in budgets:
        if "workspace" in policies:
            for rv in rivals:
                b01, b10, p = mcnemar(corr(f"workspace@{k}"), corr(f"{rv}@{k}"))
                results["mcnemar"][f"workspace_vs_{rv}@{k}"] = \
                    {"workspace_only": b01, f"{rv}_only": b10, "p": round(p, 4)}
                print(f"  [mcnemar] workspace vs {rv:9s} k={k}: +{b01}/-{b10}  p={p:.4f}", flush=True)
        for pol in [q for q in policies if q != "random"]:
            b01, b10, p = mcnemar(corr(f"{pol}@{k}"), corr("no_memory"))
            results["mcnemar"][f"{pol}_vs_no_memory@{k}"] = \
                {f"{pol}_only": b01, "no_memory_only": b10, "p": round(p, 4)}

    out = args.out or f"data/results/downstream_{args.model.split('/')[-1]}_{os.path.basename(args.results).replace('results_','').replace('.json','')}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
