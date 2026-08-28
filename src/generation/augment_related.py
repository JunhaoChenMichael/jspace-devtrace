"""
augment_related.py — the hard multi-absent variant: each episode's candidate
set is augmented with K same-category, confusable foils (proposed by an
external model, verified absent from the context). Foils share the bridge's
semantic field, so embedding relevance saturates and only evocation-sensitive
signals can separate the true bridge; any residual weak evocation of a foil
biases against the workspace readout, making the test conservative.
"""
import os, json, argparse
import urllib.request

MODEL = os.environ.get("FOIL_MODEL", "gpt-5.6")

PROMPT = """Concept: "{bridge}"
Context (the concept is implied by this text but never stated):
{ctx}

Propose {k} single-word concepts of the SAME category as "{bridge}" (e.g., if
it is a country, other countries; if a planet, other planets) that a careless
reader might confuse with it, but that are NOT implied by this context. Each
must be one lowercase word, not appearing in the context. Reply with only the
{k} words, comma-separated."""


def ask(ctx, bridge, k):
    body = {"model": MODEL, "max_completion_tokens": 3000,
            "messages": [{"role": "user",
                          "content": PROMPT.format(ctx=ctx, bridge=bridge, k=k)}]}
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=90).read())
            txt = out["choices"][0]["message"]["content"].strip().lower()
            words = [w.strip().strip(".") for w in txt.split(",")]
            return [w for w in words if w and " " not in w][:k]
        except Exception:
            import time
            time.sleep(2 ** attempt)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="data/benchmarks/battery_v4_final.json")
    ap.add_argument("--k-foils", type=int, default=4)
    ap.add_argument("--out", default="data/benchmarks/battery_v4_relabs.json")
    args = ap.parse_args()

    battery = json.load(open(args.battery))
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for ep in battery:
        lb = [it["concept"] for it in ep["items"] if it["label"] == "load_bearing"]
        jobs.append((ep, lb[0] if lb else None))
    with ThreadPoolExecutor(10) as ex:
        foil_lists = list(ex.map(
            lambda j: ask(j[0]["context"], j[1], args.k_foils) if j[1] else [],
            jobs))

    out, added = [], 0
    for (ep, bridge), foils in zip(jobs, foil_lists):
        ctx = ep["context"].lower()
        own = {it["concept"].lower() for it in ep["items"]}
        clean = [f for f in foils
                 if f not in own and ctx.rfind(f) < 0 and f != (bridge or "").lower()]
        w = dict(ep)
        w["items"] = [dict(it) for it in ep["items"]] + [
            {"concept": f, "label": "foil_absent"} for f in clean]
        added += len(clean)
        out.append(w)
    json.dump(out, open(args.out, "w"))
    print(f"saved {len(out)} episodes -> {args.out} "
          f"(+{added/len(out):.1f} related absent foils/episode)")


if __name__ == "__main__":
    main()
