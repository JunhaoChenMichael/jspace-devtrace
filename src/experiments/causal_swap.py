"""
causal_swap.py — quantitative causal test (HANDOFF TODO 6): memory READS the workspace.

For pairs of v2f episodes (bridge concept == gold answer there, which is exactly
what makes this clean): the agent answers a probe from a tiny memory containing
its own bridge concept. We then steer the residual stream along the input-side
difference-of-means direction (bridge_A -> bridge_B, jlens.concept_vector) while
it answers, and measure whether the answer log-odds flip toward episode B's
answer. Sham control: steering along a direction built from two UNRELATED
bridges (C -> D) must not produce the flip.

Metrics over N pairs:
  flip_rate       : fraction where patched argmax(ans_A, ans_B) becomes ans_B
  mean_delta      : mean shift in log-odds (ans_B - ans_A), patched - baseline
  sham_flip_rate  : same under the sham direction (should stay ~0)
"""
import os, sys, json, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens
from patch import swap_effect_repe

TEMPLATES = [
    "The key concept here is {c}.",
    "Everything in this story points to {c}.",
    "She kept thinking about {c}.",
    "The answer to the riddle is {c}.",
]


def contrast_prompts(concept):
    return [t.format(c=concept) for t in TEMPLATES]


def recall_prompt(concepts, probe):
    mem = ", ".join(concepts)
    return (f"Earlier you read a passage. The only things you remember from it are: "
            f"{mem}.\nUsing only what you remember, answer concisely.\nQuestion: {probe}\nAnswer:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--battery", default="data/benchmarks/battery_v2_final.json")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    battery = json.load(open(args.battery))
    # usable episodes: bridge == answer, single-word answer
    eps = []
    for ep in battery:
        lb = [it["concept"] for it in ep["items"] if it["label"] == "load_bearing"]
        ans = ep["answer"].strip().lower()
        if len(lb) == 1 and " " not in ans and lb[0] == ans:
            eps.append({"bridge": lb[0], "probe": ep["probe_question"], "answer": ans})
    print(f"usable episodes: {len(eps)}", flush=True)

    lens = WorkspaceLens(args.model, dtype=getattr(torch, args.dtype))
    n = lens.n_layers
    layers = list(range(n // 4, n // 4 + n // 3))

    # DISJOINT pairs (audit fix: the earlier sliding window shared episodes across
    # pairs, so flips were not independent). Real pairs consume the front of the
    # list two-at-a-time; sham directions come from a held-out pool at the back.
    N = min(args.pairs, len(eps) // 2 - 2)
    pool = eps[2 * N:]
    if len(pool) < 2:
        N -= 2
        pool = eps[2 * N:]
    results = []
    for i in range(N):
        A, B = eps[2 * i], eps[2 * i + 1]
        C = pool[(2 * i) % len(pool)]
        D = pool[(2 * i + 1) % len(pool)]
        prompt = recall_prompt([A["bridge"]], A["probe"])
        cands = [" " + A["answer"].capitalize(), " " + B["answer"].capitalize()]
        # real swap: A -> B
        r = swap_effect_repe(lens, prompt,
                             pos_prompts=contrast_prompts(B["bridge"]),
                             neg_prompts=contrast_prompts(A["bridge"]),
                             candidates=cands, layers=layers, scale=args.scale)
        # sham swap: unrelated C -> D, same prompt & candidates
        s = swap_effect_repe(lens, prompt,
                             pos_prompts=contrast_prompts(D["bridge"]),
                             neg_prompts=contrast_prompts(C["bridge"]),
                             candidates=cands, layers=layers, scale=args.scale)
        odds = lambda d: d[cands[1]] - d[cands[0]]
        rec = {"pair": f"{A['bridge']}->{B['bridge']}",
               "base_odds": round(odds(r["baseline"]), 3),
               "patched_odds": round(odds(r["patched"]), 3),
               "sham_odds": round(odds(s["patched"]), 3),
               "flip": bool(odds(r["patched"]) > 0 and odds(r["baseline"]) < 0),
               "sham_flip": bool(odds(s["patched"]) > 0 and odds(s["baseline"]) < 0)}
        results.append(rec)
        print(f"  {rec['pair']:24s} base={rec['base_odds']:+.2f} "
              f"patched={rec['patched_odds']:+.2f} sham={rec['sham_odds']:+.2f} "
              f"flip={rec['flip']} shamflip={rec['sham_flip']}", flush=True)

    flips = sum(r["flip"] for r in results)
    sham = sum(r["sham_flip"] for r in results)
    deltas = [r["patched_odds"] - r["base_odds"] for r in results]
    sham_d = [r["sham_odds"] - r["base_odds"] for r in results]
    # paired significance on the SAME (now independent) pairs: exact McNemar,
    # real-direction flip vs sham-direction flip
    from scipy.stats import binomtest, wilcoxon
    b01 = sum(1 for r in results if r["flip"] and not r["sham_flip"])
    b10 = sum(1 for r in results if r["sham_flip"] and not r["flip"])
    p_mcnemar = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    diffs = [d - s for d, s in zip(deltas, sham_d)]
    p_wilcoxon = float(wilcoxon(diffs).pvalue) if any(diffs) else 1.0
    summary = {"model": args.model, "scale": args.scale, "n_pairs": len(results),
               "flip_rate": round(flips / len(results), 3),
               "sham_flip_rate": round(sham / len(results), 3),
               "mcnemar_real_vs_sham": {"real_only": b01, "sham_only": b10,
                                        "p": round(float(p_mcnemar), 5)},
               "wilcoxon_dlogodds_p": round(p_wilcoxon, 5),
               "mean_delta_logodds": round(sum(deltas) / len(deltas), 3),
               "mean_sham_delta": round(sum(sham_d) / len(sham_d), 3),
               "pairs": results}
    out = args.out or f"pilot/causal_{args.model.split('/')[-1]}.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nflip_rate={summary['flip_rate']} sham={summary['sham_flip_rate']} "
          f"mean_dlogodds={summary['mean_delta_logodds']} (sham {summary['mean_sham_delta']})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
