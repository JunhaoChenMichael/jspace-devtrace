"""
locomo_diag.py — diagnose whether the note-level workspace score on LoCoMo
carries signal, separating two hypotheses for the null result:

  H1 (regime): surface-heavy notes are the verbal/embedding channels' home
      regime; a well-measured workspace score simply has no edge there.
  H2 (measurement): max-pooling W over a note's ~10 content words saturates
      (words literally present in the session decode at rank 1 somewhere),
      so note scores tie at 1.0 and top-k degenerates to annotation order.

Checks, per model size, scoring pass only (no generation):
  - distribution of note-level scores under three liftings:
      W_max     = max rr over all content words (what locomo_gate.py used)
      W_mean    = mean rr over content words
      W_absent  = max rr over words NOT present in the session text
                  (the note's inferred/paraphrased part - closest construct
                  to the paper's rho=0 readout)
  - tie statistics: fraction of notes at 1.0; fraction of episodes whose
    top-3 by W_max is decided by ties (selection = annotation order)
  - selection-level evidence containment@k for every policy and lifting
    (does the kept set contain a note whose dia matches question evidence?)
    - generation-free analogue of downstream accuracy.
"""
import os, sys, json, re, argparse, random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/experiments")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens
from measure import concept_token_ids, workspace_salience, yes_no_ids
from locomo_gate import (build_episodes, content_words, score_verbal,
                         score_embedding, V_PROMPT)


@torch.no_grad()
def score_workspace_variants(lens, ep):
    ctx_low = ep["context"].lower()
    all_words = sorted({w for n in ep["notes"] for w in content_words(n["note"])})
    ids_of = {w: concept_token_ids(lens, w.lower()) for w in all_words}
    flat = sorted({i for ids in ids_of.values() for i in ids})
    _, _, rr = workspace_salience(lens, ep["context"], flat, end_only=True)
    for n in ep["notes"]:
        words = content_words(n["note"])
        vals = {w: max((rr[i] for i in ids_of[w]), default=0.0) for w in words}
        absent = [w for w in words if ctx_low.rfind(w.lower()) < 0]
        n["W_max"] = max(vals.values(), default=0.0)
        n["W_mean"] = sum(vals.values()) / len(vals) if vals else 0.0
        n["W_absent"] = max((vals[w] for w in absent), default=0.0)
        n["n_words"] = len(words)
        n["n_absent"] = len(absent)


def containment(episodes, key, k=3):
    hit = tot = 0
    for ep in episodes:
        order = sorted(ep["notes"], key=lambda n: n[key], reverse=True)[:k]
        kept = set()
        for n in order:
            kept |= set(n.get("dias", []))
        for q in ep["questions"]:
            tot += 1
            hit += bool(kept & set(q["evidence"]))
    return hit / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default="data/benchmarks/locomo/locomo10.json")
    ap.add_argument("--n-questions", type=int, default=120)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = json.load(open(args.data))
    episodes = build_episodes(data, args.n_questions)
    print(f"model={args.model} episodes={len(episodes)}", flush=True)

    score_embedding(episodes, args.device)
    lens = WorkspaceLens(args.model, device=args.device, dtype=torch.bfloat16)
    yes_ids, no_ids = yes_no_ids(lens)
    for i, ep in enumerate(episodes):
        score_workspace_variants(lens, ep)
        score_verbal(lens, ep, yes_ids, no_ids)
        turn = lambda n: max((int(d.split(":")[-1]) for d in n.get("dias", [])
                              if ":" in d), default=0)
        for n in ep["notes"]:
            n["R"] = turn(n)
        if (i + 1) % 20 == 0:
            print(f"  scored {i+1}/{len(episodes)}", flush=True)

    notes = [n for ep in episodes for n in ep["notes"]]
    at1 = sum(1 for n in notes if n["W_max"] >= 0.999) / len(notes)
    ge_half = sum(1 for n in notes if n["W_max"] >= 0.5) / len(notes)
    print(f"\n== saturation: {len(notes)} notes; W_max==1.0 for {at1:.1%}, "
          f">=0.5 for {ge_half:.1%}")
    tied = 0
    for ep in episodes:
        srt = sorted(ep["notes"], key=lambda n: n["W_max"], reverse=True)
        if len(srt) > args.k and srt[args.k - 1]["W_max"] == srt[args.k]["W_max"]:
            tied += 1
    print(f"   episodes with tie across the top-{args.k} boundary: "
          f"{tied}/{len(episodes)} = {tied/len(episodes):.1%}")
    absent_any = sum(1 for n in notes if n["n_absent"] > 0) / len(notes)
    print(f"   notes with >=1 absent (paraphrase) word: {absent_any:.1%}")

    print(f"\n== evidence containment@{args.k} (selection-level, no generation)")
    rng = random.Random(0)
    for ep in episodes:
        for n in ep["notes"]:
            n["RAND"] = rng.random()
    for key, label in [("W_max", "workspace (max, as run)"),
                       ("W_mean", "workspace (mean)"),
                       ("W_absent", "workspace (absent words)"),
                       ("V", "verbal"), ("E", "embedding"),
                       ("R", "recency"), ("RAND", "random")]:
        print(f"  {label:26s} {containment(episodes, key, args.k):.3f}")

    out = f"data/results/locomo_diag_{args.model.split('/')[-1]}.json"
    slim = [{k: n[k] for k in ("W_max", "W_mean", "W_absent", "V", "E", "R",
                               "n_words", "n_absent", "dias")}
            for n in notes]
    json.dump(slim, open(out, "w"))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
