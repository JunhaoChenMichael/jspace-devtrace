"""
longmemeval_gate.py — turn-level write-admission on LongMemEval (oracle
variant; Wu et al., 2025). Counterpart to locomo_gate.py with a stricter
property: candidates are the session's own turns, i.e. verbatim substrings of
the context, so every candidate is surface-present (rho=1) by construction —
the silently-inferred regime that separates the paper's channels cannot occur.

  episode   = the single evidence session of a single-session question
  candidates = the session's turns ("user: ..." / "assistant: ...")
  probe     = the LongMemEval question
  policies  = workspace (mean-rr lifting; turns are verbatim so an
              absent-words lifting is undefined) / verbal / embedding /
              recency / random, question-blind top-k; has_answer oracle;
              no-memory and full-context references.

Responses are saved for token-F1 and external LLM-judge grading
(src/analysis/llm_judge.py).
"""
import os, sys, json, re, argparse, random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jlens import WorkspaceLens
from measure import concept_token_ids, workspace_salience, yes_no_ids
from locomo_gate import content_words, score_embedding, f1, answer, QA_PROMPT, QA_FULL
from experiments.measure import _yes_vs_no

V_PROMPT = ("{ctx}\n\nBased only on the conversation above, is the following "
            "exchange one of the most important things to remember in order "
            "to answer possible future questions?\nExchange: \"{note}\"\n"
            "Answer with a single word: yes or no.")

MAX_NOTE_WORDS = 60


def turn_note(t):
    words = str(t.get("content", "")).split()
    body = " ".join(words[:MAX_NOTE_WORDS]) + (" ..." if len(words) > MAX_NOTE_WORDS else "")
    return f"{t['role']}: {body}"


def build_episodes(data, n_questions, seed=0):
    rng = random.Random(seed)
    usable = [q for q in data
              if q["question_type"] in ("single-session-user", "single-session-assistant")
              and len(q["answer_session_ids"]) == 1]
    rng.shuffle(usable)
    episodes = []
    for q in usable[:n_questions]:
        sid = q["answer_session_ids"][0]
        sess = None
        for s_id, s in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            if s_id == sid:
                sess = s
                break
        if not sess or len(sess) < 4:
            continue
        notes = [{"note": turn_note(t), "idx": i,
                  "oracle": bool(t.get("has_answer"))}
                 for i, t in enumerate(sess)]
        ctx = "\n".join(f"{t['role']}: {t['content']}" for t in sess)
        episodes.append({"context": ctx, "notes": notes,
                         "questions": [{"question": q["question"],
                                        "answer": str(q["answer"]),
                                        "qid": q["question_id"]}]})
    return episodes


@torch.no_grad()
def score_workspace_mean(lens, ep):
    all_words = sorted({w for n in ep["notes"] for w in content_words(n["note"])})
    ids_of = {w: concept_token_ids(lens, w.lower()) for w in all_words}
    flat = sorted({i for ids in ids_of.values() for i in ids})
    _, _, rr = workspace_salience(lens, ep["context"], flat, end_only=True)
    for n in ep["notes"]:
        vals = [max((rr[i] for i in ids_of[w]), default=0.0)
                for w in content_words(n["note"])]
        n["W"] = sum(vals) / len(vals) if vals else 0.0


@torch.no_grad()
def score_verbal(lens, ep, yes_ids, no_ids):
    for n in ep["notes"]:
        q = V_PROMPT.format(ctx=ep["context"], note=n["note"])
        msgs = [{"role": "user", "content": q}]
        prompt = lens.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        enc = lens.tok(prompt, return_tensors="pt").to(lens.device)
        logits = lens.model(**enc).logits[0, -1].float()
        n["V"] = _yes_vs_no(logits, yes_ids, no_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/benchmarks/longmemeval/longmemeval_oracle")
    ap.add_argument("--n-questions", type=int, default=126)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = json.load(open(args.data))
    episodes = build_episodes(data, args.n_questions)
    nq = sum(len(ep["questions"]) for ep in episodes)
    nobs = sum(len(ep["notes"]) for ep in episodes)
    print(f"model={args.model} episodes={len(episodes)} questions={nq} "
          f"turns={nobs} k={args.k}", flush=True)

    score_embedding(episodes, args.device)
    lens = WorkspaceLens(args.model, device=args.device, dtype=torch.bfloat16)
    yes_ids, no_ids = yes_no_ids(lens)

    per_q = {p: [] for p in ["workspace", "verbal", "embedding", "recency",
                             "random", "oracle", "no_memory", "full_context"]}
    for i, ep in enumerate(episodes):
        score_workspace_mean(lens, ep)
        score_verbal(lens, ep, yes_ids, no_ids)
        notes = ep["notes"]
        for n in notes:
            n["R"] = n["idx"]

        def top(key):
            return [n["note"] for n in sorted(notes, key=lambda n: n[key],
                                              reverse=True)[:args.k]]

        idx = list(range(len(notes)))
        random.Random(i).shuffle(idx)
        sets = {"workspace": top("W"), "verbal": top("V"),
                "embedding": top("E"), "recency": top("R"),
                "random": [notes[j]["note"] for j in idx[:args.k]]}
        hits = [n["note"] for n in notes if n["oracle"]]
        rest = [n["note"] for n in notes if not n["oracle"]]
        sets["oracle"] = (hits + rest)[:args.k]

        for q in ep["questions"]:
            gold = q["answer"]
            for pol, kept in sets.items():
                resp = answer(lens, QA_PROMPT.format(
                    notes="\n".join(f"- {n}" for n in kept), q=q["question"]))
                per_q[pol].append({"q": q["question"], "gold": gold, "resp": resp,
                                   "correct": f1(resp, gold) >= 0.5})
            resp = answer(lens, QA_PROMPT.format(notes="(no notes)", q=q["question"]))
            per_q["no_memory"].append({"q": q["question"], "gold": gold, "resp": resp,
                                       "correct": f1(resp, gold) >= 0.5})
            resp = answer(lens, QA_FULL.format(ctx=ep["context"], q=q["question"]))
            per_q["full_context"].append({"q": q["question"], "gold": gold, "resp": resp,
                                          "correct": f1(resp, gold) >= 0.5})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(episodes)} episodes", flush=True)

    print("\n== accuracy (token-F1 grading)")
    for p, rows in per_q.items():
        print(f"  {p:12s} {sum(r['correct'] for r in rows)/len(rows):.3f}")
    from scipy.stats import binomtest
    print("== McNemar vs workspace")
    for p in ["verbal", "embedding", "recency", "random"]:
        a = [r["correct"] for r in per_q["workspace"]]
        b = [r["correct"] for r in per_q[p]]
        b01 = sum(1 for x, y in zip(a, b) if x and not y)
        b10 = sum(1 for x, y in zip(a, b) if y and not x)
        pv = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        print(f"  workspace vs {p:10s}: +{b01}/-{b10}  p={pv:.4f}")

    out = args.out or f"data/results/longmemeval_gate_{args.model.split('/')[-1]}_k{args.k}.json"
    json.dump({"model": args.model, "k": args.k, "per_question": per_q},
              open(out, "w"))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
