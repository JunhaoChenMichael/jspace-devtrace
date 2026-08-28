"""
harvest_gate.py — the Harvest component of provenance-routed gating (PRG).

A verbal memory writer can only store what it thinks to write down; silently
inferred content never becomes a candidate. This experiment simulates that
failure and tests whether decoding the workspace recovers it:

  1. Remove the load-bearing (silent) item from every episode's candidate set.
  2. HARVEST: decode the top-H non-surface, non-stopword concepts from the
     end-of-context residual stream (max reciprocal rank over layers) and add
     them to the candidate pool.
  3. Report bridge recovery@H (does the true silent concept appear among the
     harvested candidates?) and budget-k recall QA with the harvested pool
     under workspace scoring, vs. the crippled pool without harvesting.
"""
import os, sys, json, argparse, re
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jlens import WorkspaceLens
from downstream import generate, grade, recall_prompt

STOPWORDS = set("""the a an of to in on at for and or but with without is are was
were be been this that these those it its her his their your my our from by as
he she they we you i not no yes so if then than when where what who how why
one two three which there here also more most other some any all into over
about after before during under above out up down off very just only even
still such own same about would could should will can may might must said says
""".split())


@torch.no_grad()
def harvest(lens, context, H=10, layers=None, null_cache={}):
    """Top-H whole-word, non-surface, non-stopword concepts by CONTRASTIVE
    logit-lens decoding: score = logit(x) - logit(null context), which removes
    context-independent junk tokens that dominate raw logit-lens top-k."""
    hs, ids = lens.hidden_states(context)
    n = lens.n_layers
    layers = layers or sorted({max(1, round(n * f)) for f in
                               [0.5, 0.65, 0.8, 0.9, 1.0]})
    if "null" not in null_cache:
        null_text = ("The following is a plain paragraph. It contains ordinary "
                     "sentences about everyday matters. Nothing in particular "
                     "happens, and no specific place, person, or object is "
                     "described in detail.")
        nh, _ = lens.hidden_states(null_text)
        null_cache["null"] = {L: lens.logitlens(nh[L, -1]).float() for L in layers}
    surface = set(w.lower() for w in re.findall(r"[a-z]+", context.lower()))
    scores = {}
    for L in layers:
        diff = lens.logitlens(hs[L, -1]).float() - null_cache["null"][L]
        top = torch.topk(diff, 800)
        for rank_pos, tid in enumerate(top.indices.tolist()):
            raw = lens.tok.convert_ids_to_tokens(int(tid))
            if not (raw.startswith("\u0120") or raw.startswith(" ") or raw.startswith("\u2581")):
                continue                       # keep whole-word starts only
            word = lens.tok.decode([tid]).strip().lower()
            if not word.isalpha() or len(word) < 3:
                continue
            if word in STOPWORDS or word in surface:
                continue
            rr = 1.0 / (rank_pos + 1)
            if rr > scores.get(word, 0.0):
                scores[word] = rr
    return sorted(scores.items(), key=lambda kv: -kv[1])[:H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--harvest", type=int, default=8)
    ap.add_argument("--k", type=int, default=3, help="memory budget")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    lens = WorkspaceLens(args.model, dtype=getattr(torch, args.dtype))
    battery = json.load(open(args.battery))
    print(f"model={args.model} episodes={len(battery)} H={args.harvest} "
          f"k={args.k}", flush=True)

    cache = {}
    recovered = 0
    recs = []
    for ei, ep in enumerate(battery):
        bridge = next(it["concept"] for it in ep["items"]
                      if it["label"] == "load_bearing")
        present = [it["concept"] for it in ep["items"]
                   if it["label"] != "load_bearing"]
        harvested = harvest(lens, ep["context"], H=args.harvest)
        hwords = [w for w, _ in harvested]
        hit = bridge.lower() in hwords
        recovered += hit
        # crippled pool: writer missed the bridge; keep top-k of present items
        # (order as given = writer's own ordering proxy)
        kept_cr = present[: args.k]
        ans_cr = generate(lens, recall_prompt(kept_cr, ep["probe_question"]),
                          cache, args.max_new_tokens)
        ok_cr = grade(ans_cr, ep["answer"])
        # harvested pool: harvested concepts (workspace-ranked) get priority
        # slots, then present items fill the rest
        kept_h = (hwords[: max(1, args.k - 1)] + present)[: args.k]
        ans_h = generate(lens, recall_prompt(kept_h, ep["probe_question"]),
                         cache, args.max_new_tokens)
        ok_h = grade(ans_h, ep["answer"])
        recs.append({"episode": ei, "bridge": bridge, "recovered": bool(hit),
                     "harvested": hwords, "crippled_correct": bool(ok_cr),
                     "harvest_correct": bool(ok_h)})
        if (ei + 1) % 10 == 0:
            print(f"  {ei+1}/{len(battery)} recovery so far "
                  f"{recovered/(ei+1):.2f}", flush=True)

    n = len(recs)
    out = {"model": args.model, "battery": args.battery, "H": args.harvest,
           "k": args.k,
           "bridge_recovery": round(recovered / n, 3),
           "qa_crippled": round(sum(r["crippled_correct"] for r in recs) / n, 3),
           "qa_harvested": round(sum(r["harvest_correct"] for r in recs) / n, 3),
           "records": recs}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"recovery@{args.harvest}={out['bridge_recovery']} "
          f"QA crippled={out['qa_crippled']} harvested={out['qa_harvested']}",
          flush=True)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
