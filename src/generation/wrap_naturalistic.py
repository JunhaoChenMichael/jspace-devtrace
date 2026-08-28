"""
wrap_naturalistic.py — embed provenance-controlled episodes inside real LoCoMo
dialogue sessions, to test whether the workspace/verbal dissociation is carried
by content provenance or by synthetic surface style.

Each Decoupled episode's context paragraph is attributed to one speaker of a
real LoCoMo session and appended as that session's final turn (the
deployment-realistic position: a write gate scores content when it arrives).
Candidates, labels, and probes are unchanged; only the carrier changes.
"""
import json, argparse, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="data/benchmarks/battery_v4_final.json")
    ap.add_argument("--locomo", default="data/benchmarks/locomo/locomo10.json")
    ap.add_argument("--out", default="data/benchmarks/battery_v4_locomo_wrapped.json")
    ap.add_argument("--min-turns", type=int, default=10)
    ap.add_argument("--position", choices=["end", "mid"], default="end",
                    help="where the episode paragraph is planted in the carrier")
    args = ap.parse_args()

    battery = json.load(open(args.battery))
    data = json.load(open(args.locomo))
    carriers = []
    for smp in data:
        conv = smp["conversation"]
        for key in conv:
            m = re.match(r"session_(\d+)$", key)
            if not m or not isinstance(conv[key], list):
                continue
            if len(conv[key]) >= args.min_turns:
                lines = [f"{t['speaker']}: {t['text']}" for t in conv[key]]
                speaker = conv[key][0]["speaker"]
                carriers.append((lines, speaker))
    print(f"{len(carriers)} carrier sessions with >= {args.min_turns} turns")

    wrapped = []
    for i, ep in enumerate(battery):
        lines, speaker = carriers[i % len(carriers)]
        turn = (f"{speaker}: Oh, that reminds me of a story I read today. "
                f"{ep['context']}")
        if args.position == "end":
            ctx = "\n".join(lines) + "\n" + turn
        else:
            half = len(lines) // 2
            ctx = "\n".join(lines[:half] + [turn] + lines[half:])
        w = dict(ep)
        w["context"] = ctx
        wrapped.append(w)
    json.dump(wrapped, open(args.out, "w"))
    n_tok = sum(len(w["context"].split()) for w in wrapped) / len(wrapped)
    print(f"saved {len(wrapped)} wrapped episodes -> {args.out} "
          f"(mean {n_tok:.0f} words/context)")


if __name__ == "__main__":
    main()
