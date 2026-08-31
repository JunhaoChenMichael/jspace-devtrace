#!/usr/bin/env python3
"""Phase 0: are the metacognitive probe and the RL admission policy the same measurement?

The completed reports show a Yes rate of 0.000 for the 32B metacognitive M0 and
0.991 for the 32B RL-QA step-0 reporter. Either the two tracks measure different
things and must be named differently, or one of them is wrong.

This runs BOTH interfaces over the same candidates, in one process, on one model,
and reports prompt hashes, token sets, probabilities, log-odds, thresholded
decisions and within-episode ranks for each. No training, no tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.measure import YES_VARIANTS, NO_VARIANTS, verbal_salience, yes_no_ids  # noqa: E402
from jlens import WorkspaceLens  # noqa: E402
from memory_rl.modeling import (  # noqa: E402
    binary_action_logits,
    render_admission_prompt,
    yes_no_token_ids,
)

SCHEMA = "reporter-interface-equivalence-audit/v1"
TOLERANCE = 1e-4


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(a), rank(b))


def pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


def top_k_overlap(groups: dict[Any, list[int]], a: list[float], b: list[float], k: int) -> float:
    hits = total = 0
    for idxs in groups.values():
        if len(idxs) < k:
            continue
        ta = {i for i in sorted(idxs, key=lambda i: -a[i])[:k]}
        tb = {i for i in sorted(idxs, key=lambda i: -b[i])[:k]}
        hits += len(ta & tb) / k
        total += 1
    return hits / total if total else float("nan")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--battery", help="already-released measurement battery")
    parser.add_argument(
        "--rl-validation-split",
        metavar="TEACHER_TAG",
        help="audit the RL ID-validation episodes instead of a battery: this is the "
             "exact population on which the RL track reported its step-0 Yes rate",
    )
    parser.add_argument("--label", required=True, help="condition name for the record")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.rl_validation_split:
        from memory_rl.data import build_training_bundle
        bundle = build_training_bundle(
            repo_root=REPO_ROOT, teacher_tag=args.rl_validation_split,
            val_fraction=0.2, seed=0, top_k=2,
        )
        battery = [
            {"context": e.context, "source": e.source,
             "items": [{"concept": c.concept, "label": c.label} for c in e.candidates]}
            for e in bundle.validation_episodes
        ]
    else:
        if not args.battery:
            parser.error("pass --battery or --rl-validation-split")
        battery = json.loads(Path(args.battery).read_text())
    if args.limit_episodes:
        battery = battery[: args.limit_episodes]

    lens = WorkspaceLens(
        args.model,
        device="cuda",
        dtype=torch.bfloat16,
        model_revision=args.model_revision,
        tokenizer_revision=args.model_revision,
    )
    tok = lens.tok

    probe_yes, probe_no = yes_no_ids(lens)          # metacognitive probe token set
    rl_no, rl_yes = yes_no_token_ids(tok)           # RL policy token set

    rows: list[dict[str, Any]] = []
    for e_idx, episode in enumerate(battery):
        context = episode["context"]
        for c_idx, item in enumerate(episode["items"]):
            concept = item["concept"]

            v = verbal_salience(lens, context, concept, probe_yes, probe_no)

            prompt = render_admission_prompt(tok, context, concept)
            with torch.no_grad():
                logits = binary_action_logits(
                    lens.model, tok, [prompt], (rl_no, rl_yes), lens.device, args.max_length
                )
            probs = F.softmax(logits, dim=-1)[0]
            p_yes = float(probs[1])
            log_odds = float(logits[0, 1] - logits[0, 0])

            # the metacognitive probe prompt, rendered the same way for hashing
            probe_prompt = tok.apply_chat_template(
                [{"role": "user", "content":
                  f"{context}\n\nBased only on the passage above, is the concept "
                  f"\"{concept}\" one of the most important things to remember in order to "
                  f"answer possible future questions? Answer with a single word: yes or no."}],
                tokenize=False, add_generation_prompt=True)

            rows.append({
                "episode": e_idx, "candidate_index": c_idx,
                "concept": concept, "label": item["label"],
                "probe_prompt_sha256": hashlib.sha256(probe_prompt.encode()).hexdigest(),
                "rl_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "probe_V": v,
                "rl_p_yes": p_yes,
                "rl_log_odds": log_odds,
                "probe_decision_yes": v >= 0.5,
                "rl_decision_yes": p_yes >= 0.5,
                "abs_difference": abs(v - p_yes),
            })
        print(f"  audited episode {e_idx + 1}/{len(battery)}", flush=True)

    v_all = [r["probe_V"] for r in rows]
    p_all = [r["rl_p_yes"] for r in rows]
    groups: dict[Any, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["episode"], []).append(i)
    within = [spearman([v_all[i] for i in idx], [p_all[i] for i in idx])
              for idx in groups.values() if len(idx) > 2]
    within = [x for x in within if x is not None]

    prompts_identical = all(r["probe_prompt_sha256"] == r["rl_prompt_sha256"] for r in rows)
    tokens_identical = (sorted(probe_yes) == sorted(rl_yes)) and (sorted(probe_no) == sorted(rl_no))
    max_abs = max(r["abs_difference"] for r in rows)
    order_agreement = sum(
        1 for idx in groups.values()
        if [i for i in sorted(idx, key=lambda i: -v_all[i])] ==
           [i for i in sorted(idx, key=lambda i: -p_all[i])]
    ) / len(groups)

    if not prompts_identical or not tokens_identical:
        verdict = "INTERFACES_DIFFERENT_BUT_VALID"
    elif max_abs <= TOLERANCE:
        verdict = "INTERFACES_EQUIVALENT"
    else:
        verdict = "IMPLEMENTATION_MISMATCH"

    result = {
        "schema_version": SCHEMA,
        "verdict": verdict,
        "model": args.model, "model_revision": args.model_revision,
        "condition": args.label,
        "population": args.battery or f"rl_validation_split:{args.rl_validation_split}",
        "counts": {"episodes": len(groups), "candidates": len(rows)},
        "token_sets": {
            "probe_yes": probe_yes, "probe_no": probe_no,
            "rl_yes": rl_yes, "rl_no": rl_no,
            "identical": tokens_identical,
            "probe_variants": list(YES_VARIANTS) + list(NO_VARIANTS),
        },
        "prompts_identical": prompts_identical,
        "agreement": {
            "max_abs_probability_difference": max_abs,
            "mean_abs_probability_difference": sum(r["abs_difference"] for r in rows) / len(rows),
            "pearson": pearson(v_all, p_all),
            "spearman": spearman(v_all, p_all),
            "within_episode_spearman_mean": sum(within) / len(within) if within else None,
            "top1_overlap": top_k_overlap(groups, v_all, p_all, 1),
            "top2_overlap": top_k_overlap(groups, v_all, p_all, 2),
            "top3_overlap": top_k_overlap(groups, v_all, p_all, 3),
            "exact_within_episode_order_agreement": order_agreement,
        },
        "thresholded_yes_rate": {
            "probe_V_ge_0.5": sum(r["probe_decision_yes"] for r in rows) / len(rows),
            "rl_p_yes_ge_0.5": sum(r["rl_decision_yes"] for r in rows) / len(rows),
        },
        "score_ranges": {
            "probe_V": [min(v_all), max(v_all)],
            "rl_p_yes": [min(p_all), max(p_all)],
            "rl_log_odds": [min(r["rl_log_odds"] for r in rows),
                            max(r["rl_log_odds"] for r in rows)],
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"{verdict}: max|dV|={max_abs:.3e} spearman={result['agreement']['spearman']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
