"""
gen_battery_v4.py — decoupled answer AND anchor-free probe (the final instrument).

v3 fixed answer-coupling but its probes leak the surface anchor ("the city with
the Plaza Mayor"), so a no-memory agent answers from the probe alone (7B floor
0.404 > every policy). v4 keeps the v3 two-step structure and additionally makes
the probe refer to the episode ONLY through an uninformative proper name:
"What language should Elena have brushed up on for her trip?" is unanswerable
without remembering what the episode was about — the stored bridge concept is
the only path to the answer.
"""
import os, json, argparse
from openai import OpenAI

SYSTEM = "You are a careful dataset designer for a cognitive-science experiment on LLM memory."

INSTR = """Design {k} DIVERSE test episodes about SILENT BRIDGE reasoning for a memory experiment.

Each episode:
- "context": 4-8 sentences an agent reads, about a named person (invent a distinctive first name, different in every episode). Vary the domain widely.
- "probe_question": asked LATER, out of context. It refers to the episode ONLY via the person's name and asks something that requires (1) an unstated bridge fact inferable from the context, then (2) common general knowledge.
- "answer": the short correct answer (1-3 words).
- "items": 5-7 candidates, each {{"concept": <one common lowercase word>, "label": <"load_bearing"|"distractor"|"filler">, "role": <short reason>}}.

HARD RULES (each one is essential for validity):
- Exactly 1 item is "load_bearing": the BRIDGE concept. Its word MUST NOT appear anywhere in the context.
- THE PROBE MUST NOT CONTAIN ANY content word from the context except the person's name. Someone who never read the context CANNOT answer it (e.g. context: "Elena wandered the Plaza Mayor..." -> probe: "What language should Elena have brushed up on for her trip?" NOT "...in the city with the Plaza Mayor?"). The person's name must be the ONLY link.
- THE ANSWER MUST BE A DIFFERENT WORD FROM EVERY ITEM CONCEPT, must NOT appear in the context, and is reached by combining the bridge with general knowledge (context "Plaza Mayor" -> bridge "madrid" -> answer "spanish").
- The probe must be answerable by someone who remembers ONLY the bridge concept plus the question (bridge "madrid" + "what language?" -> "spanish").
- 2-3 items are "distractor": vivid, emotional, surprising words that DO appear literally in the context but are USELESS for the probe.
- 1-2 items are "filler": neutral words that appear in the context.
- Every concept is a single common English lowercase word.
- Do not reuse concepts or person names across episodes.

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


STOP = set("the a an of to in on at for and or but with without is are was were be been "
           "this that these those it its her his their your my our what which who where when "
           "how why should would could have has had do does did will can may might must "
           "about into over under after before during between she he they them him you i we "
           "not no nor so if then than as from by most more one two three".split())


def content_words(text):
    import re
    return {w for w in re.sub(r"[^a-z ]", " ", text.lower()).split()
            if len(w) >= 4 and w not in STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--out", default="data/benchmarks/battery_v4.json")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    episodes = []
    while len(episodes) < args.n:
        k = min(args.batch, args.n - len(episodes))
        try:
            episodes.extend(gen_batch(client, args.model, k))
            print(f"  generated {len(episodes)}/{args.n}", flush=True)
        except Exception as e:
            print(f"  batch error: {type(e).__name__}: {str(e)[:160]}")
            break

    clean = []
    d_silent = d_coupled = d_leak = d_ep = 0
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
                if it["concept"] in ctxl:
                    d_silent += 1; continue
                if it["concept"] in ansl or ansl in it["concept"]:
                    d_coupled += 1; continue
            good.append(it)
        ep["items"] = good
        # anchor-freedom: probe shares no content word (>=4 chars) with the context
        # except (approximately) the person's name — allow at most 1 shared word.
        shared = content_words(ep["probe_question"]) & content_words(ep["context"])
        if len(shared) > 1:
            d_leak += 1; continue
        if ansl and ansl not in ctxl \
           and all(ansl != it["concept"] for it in ep["items"]) \
           and any(it["label"] == "load_bearing" for it in ep["items"]) \
           and len(ep["items"]) >= 4:
            clean.append(ep)
        else:
            d_ep += 1
    with open(args.out, "w") as f:
        json.dump(clean, f, indent=2)
    n_items = sum(len(e["items"]) for e in clean)
    n_lb = sum(1 for e in clean for it in e["items"] if it["label"] == "load_bearing")
    print(f"saved {len(clean)} episodes, {n_items} items ({n_lb} anchor-free decoupled "
          f"load_bearing; dropped {d_silent} non-silent, {d_coupled} answer-coupled, "
          f"{d_leak} probe-leak episodes, {d_ep} other) -> {args.out}")


if __name__ == "__main__":
    main()
