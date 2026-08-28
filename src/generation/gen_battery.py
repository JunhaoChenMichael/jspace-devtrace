"""
gen_battery.py — build a controlled memory-salience test battery with GPT.

Each episode is a short passage an agent "reads", plus a probe question whose
answer REQUIRES certain facts. Every candidate item is labeled by ground-truth
downstream utility:

  load_bearing : the model must use this to answer the probe, but it is written
                 to be cognitively necessary yet NOT flashy (so verbal reflection
                 tends to under-rate it) — often a bridge concept.
  distractor   : vivid / emotional / surprising but irrelevant to the probe
                 (so verbal reflection tends to OVER-rate it).
  filler       : neutral background.

This planting is the whole point: it creates the regime where "what the model
SAYS is important" (verbal reflection) and "what actually MATTERS" (ground truth)
diverge — which is where a workspace signal can beat verbal reflection.

Items' `concept` is a single common word so the workspace-salience readout on the
open model is well defined at the token level.
"""
import os, json, argparse
from openai import OpenAI

SYSTEM = "You are a careful dataset designer for a cognitive-science experiment on LLM memory."

INSTR = """Design {k} DIVERSE test episodes for an experiment on what an AI agent should remember.

Each episode has:
- "context": 4-8 sentences of naturalistic text the agent reads (a scenario, note, dialogue, or report). Vary the domain: logistics, medicine, travel, coding, cooking, finance, science, everyday life.
- "probe_question": a question asked LATER whose correct answer strictly requires one or two facts from the context. Prefer questions that need a BRIDGE inference (the needed concept is implied, not stated as "important").
- "answer": the short correct answer.
- "items": 5-7 candidate things-to-remember. Each item = {{"concept": <one common lowercase word>, "label": <"load_bearing"|"distractor"|"filler">, "role": <short reason>}}.

CRITICAL design rules that make the experiment work:
- 1-2 items are "load_bearing": genuinely required to answer probe_question, but phrased in the context so they DO NOT sound dramatic or obviously important (a mundane detail, a quiet constraint, a bridge entity). Their "concept" word should actually appear or be strongly implied in the context.
- 2-3 items are "distractor": VIVID, emotional, surprising, or attention-grabbing details in the context that are USELESS for the probe_question (e.g., a shocking number, a dramatic event, a strong feeling). These should be tempting to flag as "important" even though they don't help answer the probe.
- 1-2 items are "filler": neutral background concepts.
- Every "concept" must be a single common English word (noun preferred), lowercase, that a small language model would tokenize simply.
- Do NOT reuse the same concepts across episodes.

Return ONLY valid JSON: {{"episodes": [ {{...}}, ... ]}}."""


def gen_batch(client, model, k):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": INSTR.format(k=k)}],
        response_format={"type": "json_object"},
        temperature=1.0,
    )
    return json.loads(r.choices[0].message.content)["episodes"]


SEEDS = [
    {
        "context": "The lab notebook says the reaction was run at 4 degrees Celsius overnight. "
                   "A technician spilled coffee on the bench and everyone panicked for ten minutes. "
                   "The sample was labeled B-7 and stored in the blue rack. "
                   "The fire alarm went off twice during the week for unrelated drills.",
        "probe_question": "At what temperature was the reaction run?",
        "answer": "4 degrees Celsius",
        "items": [
            {"concept": "cold", "label": "load_bearing", "role": "temperature is the answer"},
            {"concept": "coffee", "label": "distractor", "role": "dramatic spill, irrelevant"},
            {"concept": "alarm", "label": "distractor", "role": "vivid, irrelevant"},
            {"concept": "rack", "label": "filler", "role": "neutral storage detail"},
        ],
    },
    {
        "context": "Maria booked a connecting flight through Reykjavik because it was cheaper. "
                   "The airport cafe had an unforgettable, almost frightening lightning storm outside. "
                   "Her final destination was Oslo, arriving Tuesday. She lost her favorite scarf on the way.",
        "probe_question": "In which country will Maria land at her final destination?",
        "answer": "Norway",
        "items": [
            {"concept": "oslo", "label": "load_bearing", "role": "bridge to Norway"},
            {"concept": "lightning", "label": "distractor", "role": "frightening but useless"},
            {"concept": "scarf", "label": "distractor", "role": "emotional loss, useless"},
            {"concept": "cheaper", "label": "filler", "role": "reason for routing"},
        ],
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--out", default="data/benchmarks/battery.json")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    episodes = list(SEEDS)
    while len(episodes) < args.n:
        k = min(args.batch, args.n - len(episodes))
        try:
            batch = gen_batch(client, args.model, k)
            episodes.extend(batch)
            print(f"  generated {len(episodes)}/{args.n}")
        except Exception as e:
            print(f"  batch error: {type(e).__name__}: {str(e)[:160]}")
            break

    # light validation
    clean = []
    for ep in episodes:
        if all(x in ep for x in ("context", "probe_question", "answer", "items")) and ep["items"]:
            for it in ep["items"]:
                it["concept"] = str(it.get("concept", "")).strip().lower().split()[0] if it.get("concept") else ""
            ep["items"] = [it for it in ep["items"] if it["concept"] and it.get("label")]
            if any(it["label"] == "load_bearing" for it in ep["items"]):
                clean.append(ep)
    with open(args.out, "w") as f:
        json.dump(clean, f, indent=2)
    n_items = sum(len(e["items"]) for e in clean)
    n_lb = sum(1 for e in clean for it in e["items"] if it["label"] == "load_bearing")
    n_ds = sum(1 for e in clean for it in e["items"] if it["label"] == "distractor")
    print(f"saved {len(clean)} episodes, {n_items} items "
          f"({n_lb} load_bearing, {n_ds} distractor) -> {args.out}")


if __name__ == "__main__":
    main()
