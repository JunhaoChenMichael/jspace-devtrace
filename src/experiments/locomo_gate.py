"""
locomo_gate.py — naturalistic-stream validation of write-admission policies on
LoCoMo (Maharana et al., 2024). Deployment-shaped counterpart to downstream.py:

  episode   = one conversation session (context x = the session transcript)
  candidates = LoCoMo's per-session observation annotations (the factual notes
               a Generative-Agents-style writer would propose)
  probe     = a single-hop QA whose evidence turns all lie in that session
  policies  = workspace / verbal / embedding / recency / random (question-blind,
              top-k observations) + evidence oracle (upper bound) + no-memory /
              full-context references

Workspace score of an observation = max W_rr over its content words, read from
the end-of-session residual stream (same readout as measure.py, extended from
single concepts to note sentences). Verbal score = P(yes) to the importance
probe on the note. Grading: token-F1 >= 0.5 against the gold answer.
"""
import os, sys, json, re, argparse, random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jlens import WorkspaceLens
from measure import concept_token_ids, workspace_salience, yes_no_ids

STOP = set("""a an the and or but if then than so of to in on at for from by with
about as is are was were be been being do does did done have has had having i
you he she it we they them his her its our your their this that these those not
no yes what when where who whom which how why will would can could should shall
may might must there here just also very really said says say get got go went
going make made like one two too own same other others while during""".split())

V_PROMPT = ("{ctx}\n\nBased only on the conversation above, is the following "
            "note one of the most important things to remember in order to "
            "answer possible future questions?\nNote: \"{note}\"\n"
            "Answer with a single word: yes or no.")

QA_PROMPT = ("You previously had a conversation, but you can only remember "
             "these notes:\n{notes}\n\nQuestion: {q}\nAnswer briefly, in a few "
             "words.")

QA_FULL = ("Here is a conversation:\n{ctx}\n\nQuestion: {q}\nAnswer briefly, "
           "in a few words.")


def content_words(text):
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
    return [w for w in words if w.lower() not in STOP]


def session_text(conv, sid):
    date = conv.get(f"session_{sid}_date_time", "")
    lines = [f"(Conversation on {date})"] if date else []
    for t in conv[f"session_{sid}"]:
        lines.append(f"{t['speaker']}: {t['text']}")
    return "\n".join(lines)


def build_episodes(data, n_questions, seed=0):
    """One episode per (sample, session) that owns >=1 usable question."""
    rng = random.Random(seed)
    pool = []
    for smp in data:
        conv = smp["conversation"]
        obs_by_sess = {}
        for key, per_speaker in smp["observation"].items():
            m = re.match(r"session_(\d+)_observation", key)
            if not m or not isinstance(per_speaker, dict):
                continue
            notes = []
            for speaker, rows in per_speaker.items():
                for row in rows:
                    if isinstance(row, list) and len(row) >= 2:
                        dias = row[1] if isinstance(row[1], list) else [row[1]]
                        notes.append({"note": row[0],
                                      "dias": [str(d) for d in dias]})
            if notes:
                obs_by_sess[m.group(1)] = notes
        for q in smp["qa"]:
            if str(q.get("category")) != "4":
                continue
            ev = re.findall(r"D(\d+):\d+", str(q.get("evidence", "")))
            if not ev or len(set(ev)) != 1:
                continue
            sid = ev[0]
            if sid not in obs_by_sess or f"session_{sid}" not in conv:
                continue
            pool.append({"sample": smp["sample_id"], "sid": sid,
                         "question": q["question"], "answer": str(q["answer"]),
                         "evidence": re.findall(r"D\d+:\d+", str(q["evidence"])),
                         "conv": conv, "notes": obs_by_sess[sid]})
    rng.shuffle(pool)
    pool = pool[:n_questions]
    episodes = {}
    for q in pool:
        key = (q["sample"], q["sid"])
        ep = episodes.setdefault(key, {
            "context": session_text(q["conv"], q["sid"]),
            "notes": q["notes"], "questions": []})
        ep["questions"].append({k: q[k] for k in ("question", "answer", "evidence")})
    return list(episodes.values())


@torch.no_grad()
def score_workspace(lens, ep, lifting="max"):
    """Note-level workspace score at the end-of-session state.
    lifting='max'    : max W_rr over all content words (naive; ties heavily,
                       below-random selection - kept as the documented failure)
    lifting='absent' : max W_rr over words NOT present in the session text
                       (the note's paraphrase/inference part - the construct-
                       faithful lifting, mirroring the rho=0 readout)"""
    ctx_low = ep["context"].lower()
    all_words = sorted({w for n in ep["notes"] for w in content_words(n["note"])})
    ids_of = {w: concept_token_ids(lens, w.lower()) for w in all_words}
    flat = sorted({i for ids in ids_of.values() for i in ids})
    _, _, rr = workspace_salience(lens, ep["context"], flat, end_only=True)
    for n in ep["notes"]:
        words = content_words(n["note"])
        if lifting == "absent":
            words = [w for w in words if ctx_low.rfind(w.lower()) < 0]
        ws = [rr[i] for w in words for i in ids_of[w]]
        n["W"] = max(ws) if ws else 0.0


@torch.no_grad()
def score_verbal(lens, ep, yes_ids, no_ids):
    for n in ep["notes"]:
        q = V_PROMPT.format(ctx=ep["context"], note=n["note"])
        msgs = [{"role": "user", "content": q}]
        prompt = lens.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        enc = lens.tok(prompt, return_tensors="pt").to(lens.device)
        p = F.softmax(lens.model(**enc).logits[0, -1].float(), dim=-1)
        py = sum(p[i].item() for i in yes_ids)
        pn = sum(p[i].item() for i in no_ids)
        n["V"] = py / (py + pn + 1e-9)


@torch.no_grad()
def score_embedding(episodes, device):
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
        for i in range(0, len(ep["notes"]), 32):
            chunk = ep["notes"][i:i + 32]
            sims = (embed([n["note"] for n in chunk]) @ ctx.T).squeeze(-1).tolist()
            for n, s in zip(chunk, sims):
                n["E"] = float(s)
    del mdl


def keep_sets(ep, k, seed):
    """Question-blind top-k per policy; oracle is per-question (upper bound)."""
    notes = ep["notes"]

    def top(key, rev=True):
        return [n["note"] for n in sorted(notes, key=lambda n: n[key],
                                          reverse=rev)[:k]]

    def turn(n):
        ts = [int(d.split(":")[-1]) for d in n.get("dias", []) if ":" in d]
        return max(ts) if ts else 0
    for n in notes:
        n["R"] = turn(n)
    sets = {"workspace": top("W"), "verbal": top("V"), "embedding": top("E"),
            "recency": top("R")}
    idx = list(range(len(notes)))
    random.Random(seed).shuffle(idx)
    sets["random"] = [notes[i]["note"] for i in idx[:k]]
    return sets


def oracle_set(ep, question, k):
    ev = set(question["evidence"])
    hits = [n["note"] for n in ep["notes"] if ev & set(n.get("dias", []))]
    rest = [n["note"] for n in ep["notes"] if not ev & set(n.get("dias", []))]
    return (hits + rest)[:k]


def f1(pred, gold):
    pt = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", pred)]
    gt = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", gold)]
    if not pt or not gt:
        return 0.0
    common = {}
    for w in gt:
        common[w] = min(pt.count(w), gt.count(w))
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p, r = n_common / len(pt), n_common / len(gt)
    return 2 * p * r / (p + r)


@torch.no_grad()
def answer(lens, prompt_text, max_new=40):
    msgs = [{"role": "user", "content": prompt_text}]
    prompt = lens.tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
    enc = lens.tok(prompt, return_tensors="pt").to(lens.device)
    out = lens.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=lens.tok.pad_token_id or lens.tok.eos_token_id)
    return lens.tok.decode(out[0, enc["input_ids"].shape[1]:],
                           skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/benchmarks/locomo/locomo10.json")
    ap.add_argument("--n-questions", type=int, default=120)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--w-lifting", default="max", choices=["max", "absent"])
    args = ap.parse_args()

    data = json.load(open(args.data))
    episodes = build_episodes(data, args.n_questions)
    nq = sum(len(ep["questions"]) for ep in episodes)
    nobs = sum(len(ep["notes"]) for ep in episodes)
    print(f"model={args.model} episodes={len(episodes)} questions={nq} "
          f"notes={nobs} k={args.k}", flush=True)

    score_embedding(episodes, args.device)
    lens = WorkspaceLens(args.model, device=args.device, dtype=torch.bfloat16)
    yes_ids, no_ids = yes_no_ids(lens)

    per_q = {p: [] for p in ["workspace", "verbal", "embedding", "recency",
                             "random", "oracle", "no_memory", "full_context"]}
    for i, ep in enumerate(episodes):
        score_workspace(lens, ep, lifting=args.w_lifting)
        score_verbal(lens, ep, yes_ids, no_ids)
        sets = keep_sets(ep, args.k, seed=i)
        for q in ep["questions"]:
            gold = q["answer"]
            for pol, notes in sets.items():
                resp = answer(lens, QA_PROMPT.format(
                    notes="\n".join(f"- {n}" for n in notes), q=q["question"]))
                per_q[pol].append({"q": q["question"], "gold": gold,
                                   "resp": resp, "correct": f1(resp, gold) >= 0.5})
            resp = answer(lens, QA_PROMPT.format(
                notes="\n".join(f"- {n}" for n in oracle_set(ep, q, args.k)),
                q=q["question"]))
            per_q["oracle"].append({"q": q["question"], "gold": gold,
                                    "resp": resp, "correct": f1(resp, gold) >= 0.5})
            resp = answer(lens, QA_PROMPT.format(notes="(no notes)", q=q["question"]))
            per_q["no_memory"].append({"q": q["question"], "gold": gold,
                                       "resp": resp, "correct": f1(resp, gold) >= 0.5})
            resp = answer(lens, QA_FULL.format(ctx=ep["context"], q=q["question"]))
            per_q["full_context"].append({"q": q["question"], "gold": gold,
                                          "resp": resp, "correct": f1(resp, gold) >= 0.5})
        if (i + 1) % 10 == 0:
            done = len(per_q["workspace"])
            print(f"  {i+1}/{len(episodes)} episodes, {done} questions", flush=True)

    print("\n== accuracy")
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

    suffix = "" if args.w_lifting == "max" else f"_{args.w_lifting}"
    out = args.out or f"data/results/locomo_gate_{args.model.split('/')[-1]}_k{args.k}{suffix}.json"
    json.dump({"model": args.model, "k": args.k, "per_question": per_q},
              open(out, "w"))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
