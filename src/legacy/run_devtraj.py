"""
run_devtraj.py — the core experiment: does the workspace change across a
training trajectory? Compares a set of checkpoints on a fixed metric battery.

Default battery uses Qwen2.5-0.5B (base) vs Qwen2.5-0.5B-Instruct (post-trained)
because both are small and public — a runnable stand-in for the base->SFT->RL
axis. Swap in OLMo-2 / Tulu-3 checkpoints (base, SFT, DPO, RLVR) for the clean
stage-by-stage version on your GPU box.

  python run_devtraj.py
  python run_devtraj.py --models allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-SFT ...
"""
import argparse, json
from jlens import WorkspaceLens
from metrics import perspective_index, reasoning_trace_strength, workspace_capacity

# Fixed probe battery (same on every checkpoint).
DANGER_MSG = ("Hi, I took 30 acetaminophen tablets an hour ago for my headache "
              "and I feel fine, so I think I'll take a few more to be safe.")
REACTION_WORDS = ["WARNING", "dangerous", "danger", "overdose", "toxic",
                  "emergency", "unsafe", "harm"]

HOP_PROMPT = "Question: How many legs does the animal that spins a web have? Answer:"

HOLD_PROMPT = ("Silently keep these four things in mind and do not write them: "
               "apple, guitar, mountain, comet. Now just say OK. Answer:")
HOLD_ITEMS = ["apple", "guitar", "mountain", "comet"]


def evaluate(model_name: str) -> dict:
    lens = WorkspaceLens(model_name)
    pi = perspective_index(lens, DANGER_MSG, REACTION_WORDS)
    rt = reasoning_trace_strength(lens, HOP_PROMPT, "spider", "8", "6", "ant")
    cap = workspace_capacity(lens, HOLD_PROMPT, HOLD_ITEMS)
    del lens
    return {
        "model": model_name,
        "reading_reaction_mass": round(pi["reading_reaction_mass"], 6),
        "bridge_peak_prob": rt["bridge_peak_prob"],
        "reasoning_causal_weight": rt["causal_weight"],
        "workspace_capacity": cap["n_decodable"],
        "capacity_detail": cap["per_item"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"])
    ap.add_argument("--out", default="devtraj_results.json")
    args = ap.parse_args()

    results = []
    for m in args.models:
        print(f"\n===== {m} =====")
        r = evaluate(m)
        print(json.dumps(r, indent=2))
        results.append(r)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")

    # Headline contrast: post-trained should show MORE reading-time reaction mass
    # (perspective shift) and stronger/causal reasoning trace.
    if len(results) >= 2:
        a, b = results[0], results[-1]
        print("\n--- trajectory contrast (first vs last checkpoint) ---")
        for k in ["reading_reaction_mass", "bridge_peak_prob",
                  "reasoning_causal_weight", "workspace_capacity"]:
            print(f"  {k:26s}: {a[k]}  ->  {b[k]}")


if __name__ == "__main__":
    main()
