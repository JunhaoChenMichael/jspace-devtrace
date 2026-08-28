"""
gen_battery_v3.py — silent bridge with a DECOUPLED answer (construct-validity fix).

The v2 downstream flaw (found in the 2026-07-09 claim audit): the load_bearing
concept IS the gold answer verbatim, so memory-budgeted QA reduces to "did the
policy keep the answer token" (the oracle is answer injection, not memory use).

v3 keeps the silent-bridge structure but forces the probe answer to be a
DIFFERENT word from every stored concept: the agent must keep the silent bridge
concept and then COMPOSE it with general knowledge at recall time.

Example: context mentions "Oslo" (never "norway"); load_bearing = "norway"
(silent bridge); probe = "What currency should Elena exchange for the taxi?";
answer = "krone" (never in context, not an item concept, derivable only via the
bridge). Now storing the bridge is INSTRUMENTAL to the answer, not identical to it.
"""
import os, json, argparse
from openai import OpenAI

SYSTEM = "You are a careful dataset designer for a cognitive-science experiment on LLM memory."

INSTR = """Design {k} DIVERSE test episodes about SILENT BRIDGE reasoning for a memory experiment.

Each episode:
- "context": 4-8 sentences an agent reads. Vary the domain widely.
- "probe_question": asked LATER; answering requires TWO steps: (1) inferring a bridge fact that is NOT stated in the context, then (2) combining that bridge with common general knowledge.
- "answer": the short correct answer (1-3 words).
- "items": 5-7 candidates, each {{"concept": <one common lowercase word>, "label": <"load_bearing"|"distractor"|"filler">, "role": <short reason>}}.

HARD RULES (this is what makes the experiment valid):
- Exactly 1 item is "load_bearing": the BRIDGE concept required for the probe. Its word MUST NOT appear anywhere in the context.
- THE ANSWER MUST BE A DIFFERENT WORD FROM EVERY ITEM CONCEPT, and must NOT appear in the context either. The answer is reached by combining the bridge concept with general knowledge (e.g. context says "Oslo", bridge concept = "norway", probe asks about currency, answer = "krone"; context says "she is a cardiologist", bridge = "heart", probe asks which organ chamber pumps blood to the lungs, answer = "ventricle"; context says "Great Barrier Reef", bridge = "australia", probe asks the capital, answer = "canberra").
- The probe must be UNANSWERABLE without the bridge: someone who only remembers the distractors/fillers cannot answer it.
- 2-3 items are "distractor": vivid, emotional, or surprising words that DO appear literally in the context but are USELESS for the probe.
- 1-2 items are "filler": neutral words that appear in the context.
- Every concept is a single common English lowercase word, simple to tokenize.
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
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--out", default="data/benchmarks/battery_v3.json")
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
    d_silent = d_coupled = d_ep = 0
    for ep in episodes:
        if not all(x in ep for x in ("context", "probe_question", "answer", "items")):
            continue
        for fld in ("context", "probe_question", "answer"):
            if isinstance(ep[fld], list):
                ep[fld] = " ".join(str(x) for x in ep[fld])
            ep[fld] = str(ep[fld])
        if not isinstance(ep["items"], list):
            continue
        ctxl = ep["context"].lower()
        ansl = ep["answer"].strip().lower()
        for it in ep["items"]:
            it["concept"] = str(it.get("concept", "")).strip().lower().split()[0] if it.get("concept") else ""
        ep["items"] = [it for it in ep["items"] if it["concept"] and it.get("label")]
        good = []
        for it in ep["items"]:
            if it["label"] == "load_bearing":
                if it["concept"] in ctxl:          # bridge must be silent
                    d_silent += 1
                    continue
                if it["concept"] in ansl or ansl in it["concept"]:   # answer decoupled
                    d_coupled += 1
                    continue
            good.append(it)
        ep["items"] = good
        # answer itself must be absent from context and from ALL stored concepts
        if ansl and ansl not in ctxl \
           and all(ansl != it["concept"] for it in ep["items"]) \
           and any(it["label"] == "load_bearing" for it in ep["items"]):
            clean.append(ep)
        else:
            d_ep += 1
    with open(args.out, "w") as f:
        json.dump(clean, f, indent=2)
    n_items = sum(len(e["items"]) for e in clean)
    n_lb = sum(1 for e in clean for it in e["items"] if it["label"] == "load_bearing")
    print(f"saved {len(clean)} episodes, {n_items} items ({n_lb} silent decoupled load_bearing; "
          f"dropped {d_silent} non-silent, {d_coupled} answer-coupled, {d_ep} episodes) -> {args.out}")


if __name__ == "__main__":
    main()
