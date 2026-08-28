"""
gen_battery_vlm.py — silent VISUAL bridge battery (HANDOFF TODO 7 / README E3).

Design (v4-style, but the bridge enters through the IMAGE):
  - image: a famous landmark photo (wikipedia REST API originalimage).
  - context TEXT: a short scene about a named person AT the pictured place that
    NEVER names the landmark, city, country, or language — the identity of the
    place exists only in the image. Distractor/filler words appear in the text.
  - load_bearing concept: the CITY (single lowercase word), inferable only by
    recognizing the landmark in the image. Never in the text.
  - probe: asked later, references the person only; answer = the local LANGUAGE
    (never among items, never in the text) — bridge + general knowledge.

Question: does visually-acquired information enter the same word-level workspace
that the logit-lens reads, and does verbal reflection miss it there too?
"""
import os, json, argparse, time, urllib.request
from openai import OpenAI

UA = "jspace-devtrace-research/0.1 (academic memory experiment; contact: henryjiang1113@gmail.com)"

SYSTEM = "You are a careful dataset designer for a cognitive-science experiment on multimodal LLM memory."

INSTR = """List {k} world-famous landmarks that are INSTANTLY recognizable from a photo.
For each, give:
- "landmark": its common name
- "wiki": the exact English Wikipedia page title (for the photo)
- "city": the city it is in, as ONE common lowercase word (skip multi-word cities)
- "language": the main local language, one lowercase word, DIFFERENT from the city word
- "context": 3-5 sentences about a person (invent a distinctive first name, different each time)
  spending time at this place. HARD RULES: the text must NOT contain the landmark name,
  the city, the country, the language, or any nationality/demonym — the place must be
  identifiable ONLY from a photo. Include 2 vivid useless details and 1 mundane object.
- "distractors": 2 single lowercase words that DO appear in your context and are vivid but useless
- "fillers": 1 single lowercase word that appears in your context, neutral
- "person": the person's first name
Do not reuse landmarks, cities, languages, or person names across items.
Return ONLY JSON: {{"items": [ {{...}}, ... ]}}."""


def fetch_image(wiki_title, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        return out_path                       # cached from an earlier run
    url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
           + urllib.request.quote(wiki_title.replace(" ", "_")))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    # prefer the thumbnail (smaller, and wikimedia originals often 403 robots)
    src = (data.get("thumbnail") or data.get("originalimage") or {}).get("source")
    if not src:
        raise RuntimeError("no image")
    time.sleep(2.5)                           # wikimedia rate-limit courtesy
    req = urllib.request.Request(src, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    with open(out_path, "wb") as f:
        f.write(blob)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/benchmarks/battery_vlm.json")
    ap.add_argument("--imgdir", default="data/benchmarks/vlm_images")
    args = ap.parse_args()
    os.makedirs(args.imgdir, exist_ok=True)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": INSTR.format(k=args.n)}],
        response_format={"type": "json_object"}, temperature=1.0)
    items = json.loads(r.choices[0].message.content)["items"]
    print(f"generated {len(items)} landmark specs", flush=True)

    battery, seen = [], set()
    d_leak = d_img = d_dup = 0
    for it in items:
        try:
            city = str(it["city"]).strip().lower()
            lang = str(it["language"]).strip().lower()
            ctx = str(it["context"])
            ctxl = ctx.lower()
            person = str(it.get("person", "The visitor")).strip()
            if " " in city or " " in lang or city == lang:
                d_leak += 1; continue
            if city in seen or lang in ctxl or city in ctxl \
               or str(it["landmark"]).lower().split()[0] in ctxl:
                d_dup += 1; continue
            img = os.path.join(args.imgdir, f"{city}.jpg")
            fetch_image(it["wiki"], img)
        except Exception as e:
            print(f"  skip {it.get('landmark','?')}: {type(e).__name__} {str(e)[:80]}", flush=True)
            d_img += 1; continue
        eps_items = [{"concept": city, "label": "load_bearing", "role": "silent visual bridge (city)"}]
        for d in it.get("distractors", [])[:2]:
            d = str(d).strip().lower()
            if d and d in ctxl:
                eps_items.append({"concept": d, "label": "distractor", "role": "vivid in-text"})
        for fw in it.get("fillers", [])[:1] if isinstance(it.get("fillers"), list) else [it.get("fillers")]:
            fw = str(fw).strip().lower()
            if fw and fw in ctxl:
                eps_items.append({"concept": fw, "label": "filler", "role": "neutral in-text"})
        if len(eps_items) < 3:
            d_leak += 1; continue
        seen.add(city)
        battery.append({"context": ctx, "image": img,
                        "probe_question": f"What language should {person} use to chat with the locals?",
                        "answer": lang, "landmark": it["landmark"], "items": eps_items})
    json.dump(battery, open(args.out, "w"), indent=2)
    n_items = sum(len(e["items"]) for e in battery)
    print(f"saved {len(battery)} episodes / {n_items} items -> {args.out} "
          f"(dropped: {d_leak} leak/format, {d_img} image, {d_dup} dup)")


if __name__ == "__main__":
    main()
