"""
dedup_battery.py — post-process a generated battery (HANDOFF TODO 3).

Enforces, across the WHOLE battery (the generator only enforces within a batch):
  - no concept reused across episodes (first episode to use a concept keeps it);
  - every kept episode still has >=1 load_bearing item and >=4 items total;
  - (v2 rule, checked for any battery) a load_bearing concept that appears
    literally in its context is dropped ONLY when --silent is set.

Usage: python pilot/dedup_battery.py in.json out.json [--silent]
"""
import json, sys, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--silent", action="store_true",
                    help="enforce the silent-bridge rule (v2 batteries)")
    ap.add_argument("--min-items", type=int, default=4)
    args = ap.parse_args()

    battery = json.load(open(args.inp))
    seen = set()
    kept, d_dup, d_silent, d_ep = [], 0, 0, 0
    for ep in battery:
        ctx = ep["context"].lower()
        items = []
        for it in ep["items"]:
            c = it["concept"].strip().lower()
            if c in seen:
                d_dup += 1
                continue
            if args.silent and it["label"] == "load_bearing" and c in ctx:
                d_silent += 1
                continue
            items.append(it)
        if sum(it["label"] == "load_bearing" for it in items) >= 1 and len(items) >= args.min_items:
            for it in items:
                seen.add(it["concept"].strip().lower())
            ep["items"] = items
            kept.append(ep)
        else:
            d_ep += 1

    json.dump(kept, open(args.out, "w"), indent=2)
    n_items = sum(len(e["items"]) for e in kept)
    n_lb = sum(1 for e in kept for it in e["items"] if it["label"] == "load_bearing")
    n_ds = sum(1 for e in kept for it in e["items"] if it["label"] == "distractor")
    print(f"{args.inp}: {len(battery)} eps -> kept {len(kept)} eps / {n_items} items "
          f"({n_lb} load_bearing, {n_ds} distractor); dropped {d_dup} dup items, "
          f"{d_silent} non-silent load_bearing, {d_ep} episodes -> {args.out}")


if __name__ == "__main__":
    main()
