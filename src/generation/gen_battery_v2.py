"""
gen_battery_v2.py — a SILENT-BRIDGE battery.

v1 found the workspace signal is real (not surface), but 98% of concepts appeared
literally, so it barely tested the regime that matters most: concepts the model
must USE but never SEES or SAYS. v2 fixes that.

Each episode's load_bearing concept is a BRIDGE the model must infer and that does
NOT appear as a literal word in the context (e.g. context mentions "Oslo" and the
probe needs the country -> load_bearing concept "norway", never written).
Distractors are vivid words that DO appear literally. This maximizes the gap
between "what's in the workspace" (should hold the silent bridge) and "what the
text contains / what the model says is important" (should be fooled by the vivid
present words).
"""
import os, json, argparse
from openai import OpenAI

SYSTEM = "You are a careful dataset designer for a cognitive-science experiment on LLM memory."

INSTR = """Design {k} DIVERSE test episodes about SILENT BRIDGE reasoning for a memory experiment.

Each episode:
- "context": 4-8 sentences an agent reads. Vary the domain widely.
- "probe_question": asked LATER; answering REQUIRES inferring a bridge fact that is NOT stated.
- "answer": the short correct answer (this is the bridge concept).
- "items": 5-7 candidates, each {{"concept": <one common lowercase word>, "label": <"load_bearing"|"distractor"|"filler">, "role": <short reason>}}.

HARD RULES (this is what makes the experiment valid):
- Exactly 1 item is "load_bearing": it is the BRIDGE concept required to answer the probe, and its concept word MUST NOT appear anywhere in the context. It must be inferable (e.g. context says "Oslo", load_bearing concept = "norway"; context says "she is a cardiologist", load_bearing = "heart"; context says "the Great Barrier Reef", load_bearing = "australia"). The word itself is never written.
- 2-3 items are "distractor": vivid, emotional, or surprising words that DO appear literally in the context but are USELESS for the probe.
- 1-2 items are "filler": neutral words that appear in the context.
- Every concept is a single common English lowercase word, simple to tokenize.
- The load_bearing concept must genuinely NOT be a substring of the context. Double-check this.
- Do not reuse concepts across episodes.

Return ONLY JSON: {{"episodes": [ {{...}}, ... ]}}."""


def gen_batch(client, model, k):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": INSTR.format(k=k)}],
        response_format={"type": "json_object"},
        temperature=1.0,
    )
    return json.loads(r.choices[0].message.content)["episodes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--out", default="data/benchmarks/battery_v2.json")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    episodes = []
    while len(episodes) < args.n:
        k = min(args.batch, args.n - len(episodes))
        try:
            episodes.extend(gen_batch(client, args.model, k))
            print(f"  generated {len(episodes)}/{args.n}")
        except Exception as e:
            print(f"  batch error: {type(e).__name__}: {str(e)[:160]}")
            break

    clean = []
    dropped = 0
    for ep in episodes:
        if not all(x in ep for x in ("context", "probe_question", "answer", "items")):
            continue
        # GPT occasionally returns a list of sentences / non-str fields; coerce
        for fld in ("context", "probe_question", "answer"):
            if isinstance(ep[fld], list):
                ep[fld] = " ".join(str(x) for x in ep[fld])
            ep[fld] = str(ep[fld])
        if not isinstance(ep["items"], list):
            continue
        ctxl = ep["context"].lower()
        for it in ep["items"]:
            it["concept"] = str(it.get("concept", "")).strip().lower().split()[0] if it.get("concept") else ""
        ep["items"] = [it for it in ep["items"] if it["concept"] and it.get("label")]
        # enforce: load_bearing concept must NOT appear literally
        good = []
        for it in ep["items"]:
            if it["label"] == "load_bearing" and it["concept"] in ctxl:
                dropped += 1
                continue  # violates silent-bridge rule; drop this item
            good.append(it)
        ep["items"] = good
        if any(it["label"] == "load_bearing" for it in ep["items"]):
            clean.append(ep)
    with open(args.out, "w") as f:
        json.dump(clean, f, indent=2)
    n_items = sum(len(e["items"]) for e in clean)
    n_lb = sum(1 for e in clean for it in e["items"] if it["label"] == "load_bearing")
    print(f"saved {len(clean)} episodes, {n_items} items ({n_lb} silent load_bearing; "
          f"dropped {dropped} non-silent load_bearing) -> {args.out}")


if __name__ == "__main__":
    main()
