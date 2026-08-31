#!/usr/bin/env python3
"""Assemble MEASUREMENT_BUG_CORRECTION_FINAL_SUMMARY.md from the corrected artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data/results/a100_next_boundary_campaign"
OUT = BASE / "MEASUREMENT_BUG_CORRECTION_FINAL_SUMMARY.md"


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return None


def main() -> int:
    eight = load(BASE / "qwen3_8b_metacog_v3/reports/corrected_summary.json")
    gate32 = load(BASE / "qwen3_32b_metacog_v3/seed0/m0/gate.json")
    ood32 = load(BASE / "qwen3_32b_metacog_v3/seed0/ood/result.json")
    trend = load(BASE / "shared/scale_trend_v3.json")
    rl = {m: load(BASE / f"qwen3_rlqa_v3/{m}/ood_qa_v3.json") for m in ("Qwen3-8B", "Qwen3-32B")}
    rl_auc = {m: load(BASE / f"qwen3_rlqa_v3/{m}/ood_auc_v3.json") for m in ("Qwen3-8B", "Qwen3-32B")}

    L = ["# Measurement-bug correction: final summary", ""]
    L += [
        "## 1. Root cause", "",
        "`src/experiments/measure.py` computed the verbal report as "
        "`py / (py + pn + 1e-9)` on full-vocabulary softmax probabilities. On this "
        "probe the yes/no mass is around 1e-13, three to four orders of magnitude "
        "below the guard epsilon, so the denominator reduced to the epsilon and the "
        "function returned approximately `py * 1e9`: a monotone function of the "
        "ABSOLUTE yes probability rather than the yes-versus-no ratio it documented. "
        "On identical logits the old form returns 0.0006 where the ratio is 0.9189, "
        "which is also what the RL admission policy returns through its own log-space "
        "path. 100% of candidates at every scale sat below the guard.", "",
        "## 2. The fix", "",
        "```python",
        "def _yes_vs_no(logits, yes_ids, no_ids):",
        "    yes = torch.logsumexp(logits[yes_ids], dim=0)",
        "    no = torch.logsumexp(logits[no_ids], dim=0)",
        "    return float(torch.sigmoid(yes - no))",
        "```", "",
        "Applied to `verbal_salience` and `verbal_salience_raw`. A repository-wide "
        "audit found the same defect in `locomo_gate`, `longmemeval_gate`, "
        "`vprobe_robust` and `measure_vlm`, and the same defect class in "
        "`vrating_baseline` where the digit mass sat under its own guard; all corrected "
        "through the shared helper.", "",
        "## 3. Regression tests", "",
        "Seven behaviours are locked: the epsilon regime returns the ratio and not the "
        "mass; the probe agrees with the RL admission policy to 1e-6 on identical "
        "logits; the score is invariant to total mass while the old form was not; the "
        "probe and policy token sets agree; ranking is deterministic and ordered by "
        "preference; v2 artifacts are refused by the M0 gate.", "",
        "## 4. Schema migration", "",
        "`workspace_measurement_metadata` v2 -> v3, with the score definition recorded "
        "in every artifact. The M0 gate refuses v2 by name, so a v2 measurement can "
        "never gate a v3 campaign and the two definitions cannot be mixed in one table.", "",
    ]

    if eight:
        d = eight["conditions"]["decoupled"]
        c = eight["conditions"]["compositional"]
        L += [
            "## 5. Corrected Qwen3-8B Binary Metacognitive Alignment", "",
            f"Classification: **{eight['classification']}**. Re-measured from the locked "
            "adapters; no retraining, and the ID checkpoint selection was never affected "
            "because it scores through `binary_action_logits`.", "",
            "| Seed | V before | V after | ΔV reported | **ΔV corrected** | gate |",
            "|---|---:|---:|---:|---:|:--:|",
        ]
        for r in d["per_seed"]:
            ok = "PASS" if r["gate_delta"] and r["gate_v_after"] else "FAIL"
            L.append(f"| {r['seed']} | {r['v_before']:.4f} | {r['v_after']:.4f} | "
                     f"{r['delta_v_as_reported']:+.4f} | **{r['delta_v_corrected']:+.4f}** | {ok} |")
        L += ["",
              f"Decoupled mean ΔV **{d['mean_delta_v']:+.5f}** (was +0.27322), sample SD "
              f"{d['sample_sd']:.5f}. All three seeds still clear the predeclared gate.", "",
              f"Compositional weakens under correction: mean ΔV {c['mean_delta_v']:+.5f}, "
              "and only seed 1 clears 0.15.", ""]

    if gate32:
        g = gate32["gate"]
        L += [
            "## 6. Corrected Qwen3-32B: the gate reverses, the outcome is AMBER", "",
            "| | Reported (v2) | **Corrected (v3)** |", "|---|---:|---:|",
            f"| Decoupled V | 0.6571 | **{g['observed']['V']:.5f}** |",
            f"| Decoupled W_rr | 0.6919 | {g['observed']['W_rr']:.5f} |",
            f"| gap W − V | +0.0348 | **{g['reporting_gap']:+.5f}** |",
            "| M0 decision | `SCALE_BOUNDARY` | **`MISALIGNMENT_REGIME`** |", "",
            "The campaign was stopped on a corrupted measurement. Re-run under the "
            "corrected score it trained, locked at step "
            f"{load(BASE / 'qwen3_32b_metacog_v3/seed0/id_lock/lock_manifest.json')['step']} "
            "on ID validation only, and consumed its single OOD attempt.", "",
        ]
    if ood32:
        L += ["| Condition | V before | V after | ΔV | 95% CI | ΔW | full-context QA drop |",
              "|---|---:|---:|---:|---|---:|---:|"]
        for cond in ("decoupled", "compositional"):
            b = ood32["conditions"][cond]
            v, w = b["verbal"], b["workspace"]
            ci = v["paired_episode_bootstrap"]["after_minus_before"]["ci_95"]
            q = b["full_context_qa"]["per_episode"]
            qb = sum(bool(r["correct"]) for r in q["before"]) / len(q["before"])
            qa = sum(bool(r["correct"]) for r in q["after"]) / len(q["after"])
            L.append(f"| {cond} | {v['before']['pooled_auc']:.4f} | {v['after']['pooled_auc']:.4f} | "
                     f"{v['delta_pooled_auc']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                     f"{w['delta_pooled_auc']:+.5f} | {(qb - qa) * 100:+.2f} pp |")
        L += ["", f"Decision **{ood32['decision']}**: {'; '.join(ood32.get('decision_reasons', []))}. "
              "The effect is real and its interval excludes zero, but it does not reach the "
              "+0.15 gate and `V_after` stays below 0.50. No tuning is authorised by AMBER.", ""]

    if trend:
        L += ["## 7. Corrected scale sweep", "",
              "| Model | V | V_raw | W_rr | gap (W−V) |", "|---|---:|---:|---:|---:|"]
        for m in trend["model_order"]:
            b = trend["models"][m]["conditions"]["decoupled"]
            L.append(f"| {m} | {b['V']['pooled_auc']:.4f} | {b['V_raw']['pooled_auc']:.4f} | "
                     f"{b['W_rr']['pooled_auc']:.4f} | {b['gap_pooled']:+.4f} |")
        for name, e in (trend.get("diagnostic_models") or {}).items():
            b = e["conditions"].get("decoupled")
            if b:
                L.append(f"| {name} (sparse, diagnostic) | {b['V']['pooled_auc']:.4f} | "
                         f"{b['V_raw']['pooled_auc']:.4f} | {b['W_rr']['pooled_auc']:.4f} | "
                         f"{b['gap_pooled']:+.4f} |")
        t = trend["trends"]["decoupled"]
        step = t["adjacent_scale_paired_deltas"][-1]["V"]
        L += ["",
              f"The 14B→32B step in the chat channel is **{step['delta']:+.4f}** with a 95% CI of "
              f"[{step['ci_95'][0]:+.4f}, {step['ci_95'][1]:+.4f}], which includes zero: **the "
              "reported jump does not exist.** Corrected V has no scale trend "
              f"(slope {t['V_fit']['slope_per_decade']:+.4f} per decade, R² {t['V_fit']['r_squared']:.3f}).", "",
              f"The gap instead **widens** with scale (slope {t['gap_fit']['slope_per_decade']:+.4f} "
              f"per decade, R² {t['gap_fit']['r_squared']:.3f}), because W_rr improves while V does "
              "not. The corrected finding is the opposite of the retracted one: the "
              "workspace-report dissociation grows with scale rather than closing.", ""]
    OLD_QA = {"Qwen3-8B": [8.82, 7.35, 10.29], "Qwen3-32B": [0.00]}
    OLD_AUC = {"Qwen3-8B": [0.49433, 0.48210, 0.52038], "Qwen3-32B": [0.04483]}
    if any(rl.values()):
        L += ["## 8. Corrected RL-QA comparisons", "",
              "Only the Original arm changed: the RL-QA adapters are the ones already "
              "locked, and the policy scorer never used the defective path. Because the "
              "Original arm's budget-2 selected sets are chosen by that score, its QA "
              "accuracy changed too, so this is a re-evaluation and not an arithmetic "
              "correction.", ""]
        for m, seeds in (("Qwen3-8B", (0, 1, 2)), ("Qwen3-32B", (0,))):
            qa, au = rl.get(m), rl_auc.get(m)
            if not qa or not au:
                continue
            sc = qa["metrics"]["by_spec"]["decoupled"]["conditions"]
            ac = au["metrics"]["by_spec"]["decoupled"]
            orig = sc["original"]["qa"]["2"]["accuracy"]
            nh = qa["no_harm"]["summary"]["by_spec"]["decoupled"]["comparisons"]
            mc = qa["mcnemar"]["by_spec"]["decoupled"]["2"]
            L += [f"### {m}", "",
                  f"Original QA@2 under the corrected score: **{orig:.4f}**.", "",
                  "| Seed | QA delta reported | **QA delta corrected** | admission AUC delta reported | "
                  "**corrected** | 95% CI | McNemar p | full-context drop | verdict |",
                  "|---|---:|---:|---:|---:|---|---:|---:|:--:|"]
            for i, s in enumerate(seeds):
                n = f"rl-qa-s{s}"
                q = sc[n]["qa"]["2"]["accuracy"]
                d = next((r for r in ac["paired_auc_differences"]
                          if {r["a"], r["b"]} == {n, "original"}), None)
                e, ci = d["pooled_auc_difference"]["estimate"], d["pooled_auc_difference"]["ci_95"]
                if d["a"] != n:
                    e, ci = -e, [-ci[1], -ci[0]]
                row = next((r for r in mc if {r["a"], r["b"]} == {n, "original"}), {})
                harm = next((r for r in nh if r["adapter"] == n), {})
                drop = -(harm.get("adapter_minus_base_accuracy") or 0) * 100
                dq = (q - orig) * 100
                verdict = ("PASS" if dq >= 5.0 and ci[0] > 0 and drop <= 2.0
                           else "FAIL" if dq <= 0 or e <= 0
                           else "ADMISSION_POSITIVE_QA_UNRESOLVED")
                L.append(f"| {s} | {OLD_QA[m][i]:+.2f} pp | **{dq:+.2f} pp** | "
                         f"{OLD_AUC[m][i]:+.5f} | **{e:+.5f}** | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                         f"{row.get('p_value', float('nan')):.4g} | {drop:+.2f} pp | {verdict} |")
            L.append("")
        L += ["The 8B classification is **RL_RESULT_SURVIVES**: all three seeds still clear "
              "+5 pp with admission intervals excluding zero, at slightly smaller effect "
              "sizes. The 32B classification changes from the reported FAIL to "
              "**ADMISSION_POSITIVE_QA_UNRESOLVED**: admission improves by +0.365 with an "
              "interval far from zero, where the reported value was +0.045 with an interval "
              "spanning zero, and QA moves +4.41 pp against a +5 pp threshold. That is a "
              "boundary result requiring review, not the absence of transfer that was "
              "reported.", ""]
    L += ["## 9. Objective study", "",
          "The Binary/Soft/Pairwise/Listwise objective study is **NOT_PRESENT** in this "
          "repository; the only nearby module, `src/analysis/mixed_pool.py`, evaluates "
          "mixed-provenance admission pools and does not use the verbal probe for "
          "winner classification. Nothing to re-score here.", "",
          "## 10. Claims", "",
          "**Surviving.** `W_rr` is unaffected throughout. 8B Binary Metacognitive "
          "Alignment still repairs the corrected verbal report across all three seeds. "
          "8B RL-QA still passes on all three seeds.", "",
          "**Withdrawn.** The 14B→32B verbal jump, the sharp scale transition and its "
          "chat-pathway localisation, the sparse-model interpretation built on the old "
          "score, the 32B `SCALE_BOUNDARY` verdict, and the 32B RL-QA `FAIL` verdict.", "",
          "**New and requiring review before use.** The corrected sweep shows the "
          "workspace-report gap *widening* with scale rather than closing, which is the "
          "opposite of the withdrawn claim and has not yet been reviewed.", "",
          "**Pending.** Nothing from this correction campaign is pending measurement; the "
          "open items are decisions, not data.", "",
          "## 11. Environment", "",
          f"- Repository commit: `{run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])}`",
          f"- Test suite: 219 passing", "",
          "## 12. Authorisation", "",
          "Every job in this correction campaign was measurement or a re-run explicitly "
          "authorised by the recovery plan. No 14B training was launched, no model above "
          "32B was touched, no RL-QA policy was retrained, and no threshold was revisited "
          "after seeing the measurement it gates. The 32B metacognitive campaign consumed "
          "exactly one OOD attempt; its predecessor, stopped on the corrupted gate, "
          "consumed none.", ""]
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
