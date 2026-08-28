"""
llm_judge.py — regrade saved naturalistic-gating runs (locomo_gate /
longmemeval_gate JSONs) with an LLM judge, because token-F1 marks
paraphrased-but-correct answers wrong (and occasionally vice versa).

For every record in per_question[policy], adds "correct_judge" (bool) and
prints accuracy + exact McNemar (workspace vs each policy) under both graders,
plus the F1-vs-judge disagreement rate. Judgments are cached by
(question, gold, response) so shared records cost one call.

Judge: OPENAI_JUDGE_MODEL (default gpt-4.1-mini), temperature 0, yes/no.
Requires OPENAI_API_KEY (source server_env.sh).
"""
import os, sys, json, re, argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

JUDGE_MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4.1-mini")

PROMPT = """You are grading a memory-QA system. Decide whether the response \
correctly answers the question, using the gold answer as reference. The \
response is correct if it contains or entails the gold answer's content \
(paraphrase, reformatting, extra detail are fine); it is incorrect if it \
contradicts the gold answer, omits its key content, or refuses.

Question: {q}
Gold answer: {gold}
Response: {resp}

Reply with exactly one word: yes (correct) or no (incorrect)."""


def judge_one(item):
    q, gold, resp = item
    body = {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 3,
            "messages": [{"role": "user",
                          "content": PROMPT.format(q=q, gold=gold, resp=resp[:800])}]}
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=60).read())
            ans = out["choices"][0]["message"]["content"].strip().lower()
            return ans.startswith("y")
        except Exception as e:
            if attempt == 3:
                print(f"  judge error (kept F1 grade): {e}", file=sys.stderr)
                return None
            import time
            time.sleep(2 ** attempt)


def mcnemar(a, b):
    from math import comb
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n, m = b01 + b10, min(b01, b10)
    p = min(1.0, sum(comb(n, i) for i in range(m + 1)) * 2 / 2 ** n) if n else 1.0
    return b01, b10, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--field", default="correct_judge",
                    help="record field to write (use correct_judge2 for a second judge)")
    args = ap.parse_args()

    for path in args.files:
        d = json.load(open(path))
        per_q = d["per_question"]
        keys, seen = [], {}
        for pol, rows in per_q.items():
            for r in rows:
                k = (r["q"], r["gold"], r["resp"])
                if k not in seen:
                    seen[k] = None
                    keys.append(k)
        print(f"\n### {os.path.basename(path)}: {len(keys)} unique judgments")
        with ThreadPoolExecutor(args.workers) as ex:
            for k, v in zip(keys, ex.map(judge_one, keys)):
                seen[k] = v

        dis = tot = 0
        for pol, rows in per_q.items():
            for r in rows:
                v = seen[(r["q"], r["gold"], r["resp"])]
                r[args.field] = r["correct"] if v is None else v
                tot += 1
                dis += (r[args.field] != r["correct"])
        print(f"F1-vs-judge disagreement: {dis}/{tot} = {dis/tot:.1%}")

        print(f"{'policy':14s} {'F1':>6s} {'judge':>6s}")
        for pol, rows in per_q.items():
            f = sum(r["correct"] for r in rows) / len(rows)
            j = sum(r[args.field] for r in rows) / len(rows)
            print(f"{pol:14s} {f:6.3f} {j:6.3f}")
        if "workspace" in per_q:
            print("McNemar vs workspace (judge grading):")
            a = [r[args.field] for r in per_q["workspace"]]
            for pol in ["verbal", "embedding", "recency", "random"]:
                if pol not in per_q:
                    continue
                b01, b10, p = mcnemar(a, [r[args.field] for r in per_q[pol]])
                print(f"  workspace vs {pol:10s}: +{b01}/-{b10}  p={p:.4f}")
        json.dump(d, open(path, "w"))
        print(f"updated {path} ({args.field} added)")


if __name__ == "__main__":
    main()
