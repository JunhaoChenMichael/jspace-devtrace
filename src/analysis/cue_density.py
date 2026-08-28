"""
cue_density.py — quantify the cue density of Decoupled-design episodes and
test the dose-response with the 7B reporter, turning the 'sparse vs densely
cued' description into a measured variable.

Cue = a distinct context element (entity, fact, or image) that would let a
knowledgeable reader infer the bridge on its own. Counted by an external
annotator model (gpt-5.6, temperature 0) that sees the context and the bridge.

Outputs per benchmark: mean/median cue count; pooled Spearman correlation of
per-episode cue count with 7B verbal success (bridge ranked top-2 by V within
its episode); reporter success rate by cue-count bin.
"""
import os, json, argparse
import urllib.request

MODEL = os.environ.get("CUE_MODEL", "gpt-5.6")

PROMPT = """Context:
{ctx}

Hidden concept: "{bridge}"

Count the DISTINCT cues in the context that would each, on its own, let a
knowledgeable reader infer the hidden concept. A cue is a named entity, fact,
or described detail pointing to the concept (e.g., a landmark, a currency, a
characteristic custom). Do not count vague atmosphere. Reply with only an
integer."""


def ask(ctx, bridge):
    body = {"model": MODEL, "max_completion_tokens": 4000,
            "messages": [{"role": "user",
                          "content": PROMPT.format(ctx=ctx, bridge=bridge)}]}
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=90).read())
            txt = out["choices"][0]["message"]["content"].strip()
            digits = "".join(c for c in txt if c.isdigit())
            return int(digits) if digits else None
        except Exception:
            import time
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=(
        "data/benchmarks/battery_v4_final.json:data/results/results_v4f_7B-Instruct.json:gpt-4.1,"
        "data/benchmarks/battery_v4_xl.json:data/results/results_v4xl_7B-Instruct.json:gpt-4.1,"
        "data/benchmarks/battery_v4_g56.json:data/results/results_v4g56_7B-Instruct.json:gpt-5.6"))
    ap.add_argument("--out", default="data/results/cue_density.json")
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor
    records = []
    for spec in args.sets.split(","):
        bat_f, res_f, gen = spec.split(":")
        bat = json.load(open(bat_f))
        rows = json.load(open(res_f))
        by_ep = {}
        for r in rows:
            by_ep.setdefault(str(r["episode"]), []).append(r)
        jobs = []
        for ei, ep in enumerate(bat):
            items = by_ep.get(str(ei))
            if not items:
                continue
            lb = [r for r in items if r["label"] == "load_bearing"]
            if not lb:
                continue
            ranked = sorted(items, key=lambda r: r["V"], reverse=True)
            top2 = any(r["label"] == "load_bearing" for r in ranked[:2])
            jobs.append((ei, ep["context"], lb[0]["concept"], top2))
        with ThreadPoolExecutor(12) as ex:
            counts = list(ex.map(lambda j: ask(j[1], j[2]), jobs))
        for (ei, _, bridge, top2), c in zip(jobs, counts):
            if c is not None:
                records.append({"gen": gen, "set": bat_f, "episode": ei,
                                "bridge": bridge, "cues": c, "v_top2": top2})
        done = [r for r in records if r["set"] == bat_f]
        mean = sum(r["cues"] for r in done) / len(done)
        rate = sum(r["v_top2"] for r in done) / len(done)
        print(f"{bat_f} ({gen}): {len(done)} eps, mean cues {mean:.2f}, "
              f"7B reporter top-2 rate {rate:.3f}", flush=True)

    import numpy as np
    from scipy.stats import spearmanr
    cues = np.array([r["cues"] for r in records])
    succ = np.array([1 if r["v_top2"] else 0 for r in records])
    rho, p = spearmanr(cues, succ)
    print(f"\npooled ({len(records)} episodes): Spearman rho={rho:.3f}, p={p:.4g}")
    for lo, hi in [(0, 1), (2, 2), (3, 3), (4, 99)]:
        m = (cues >= lo) & (cues <= hi)
        if m.sum():
            print(f"  cues {lo}-{hi if hi<99 else '+'}: n={m.sum()}, "
                  f"reporter success {succ[m].mean():.3f}")
    json.dump(records, open(args.out, "w"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
