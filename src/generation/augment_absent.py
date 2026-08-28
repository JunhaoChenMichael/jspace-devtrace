"""
augment_absent.py — build the multi-absent stress-test variant of a
Decoupled-design benchmark: each episode's candidate set is augmented with K
plausible context-absent foils (other episodes' bridges, verified absent from
this context), so bare absence no longer identifies the load-bearing item and
a write gate must discriminate among absent candidates end-to-end.

Foils are labeled 'foil_absent'; exactly one absent candidate per episode is
load-bearing. Everything else about the episode is unchanged.
"""
import json, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="data/benchmarks/battery_v4_final.json")
    ap.add_argument("--k-foils", type=int, default=4)
    ap.add_argument("--out", default="data/benchmarks/battery_v4_multiabs.json")
    args = ap.parse_args()

    battery = json.load(open(args.battery))
    bridges = []
    for ep in battery:
        lb = [it["concept"] for it in ep["items"] if it["label"] == "load_bearing"]
        bridges.append(lb[0] if lb else None)

    n = len(battery)
    out = []
    added = 0
    for i, ep in enumerate(battery):
        ctx = ep["context"].lower()
        own = {it["concept"].lower() for it in ep["items"]}
        foils, j = [], (i + 1) % n
        while len(foils) < args.k_foils and j != i:
            c = bridges[j]
            if (c and c.lower() not in own and ctx.rfind(c.lower()) < 0
                    and c.lower() not in [f.lower() for f in foils]):
                foils.append(c)
            j = (j + 1) % n
        w = dict(ep)
        w["items"] = [dict(it) for it in ep["items"]] + [
            {"concept": f, "label": "foil_absent"} for f in foils]
        added += len(foils)
        out.append(w)
    json.dump(out, open(args.out, "w"))
    per = added / len(out)
    print(f"saved {len(out)} episodes -> {args.out} "
          f"(+{per:.1f} absent foils/episode; absent candidates per episode now "
          f"~{per + 1.1:.1f})")


if __name__ == "__main__":
    main()
