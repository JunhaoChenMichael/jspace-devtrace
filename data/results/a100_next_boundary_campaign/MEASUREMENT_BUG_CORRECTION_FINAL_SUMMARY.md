# Measurement-bug correction: final summary

## 1. Root cause

`src/experiments/measure.py` computed the verbal report as `py / (py + pn + 1e-9)` on full-vocabulary softmax probabilities. On this probe the yes/no mass is around 1e-13, three to four orders of magnitude below the guard epsilon, so the denominator reduced to the epsilon and the function returned approximately `py * 1e9`: a monotone function of the ABSOLUTE yes probability rather than the yes-versus-no ratio it documented. On identical logits the old form returns 0.0006 where the ratio is 0.9189, which is also what the RL admission policy returns through its own log-space path. 100% of candidates at every scale sat below the guard.

## 2. The fix

```python
def _yes_vs_no(logits, yes_ids, no_ids):
    yes = torch.logsumexp(logits[yes_ids], dim=0)
    no = torch.logsumexp(logits[no_ids], dim=0)
    return float(torch.sigmoid(yes - no))
```

Applied to `verbal_salience` and `verbal_salience_raw`. A repository-wide audit found the same defect in `locomo_gate`, `longmemeval_gate`, `vprobe_robust` and `measure_vlm`, and the same defect class in `vrating_baseline` where the digit mass sat under its own guard; all corrected through the shared helper.

## 3. Regression tests

Seven behaviours are locked: the epsilon regime returns the ratio and not the mass; the probe agrees with the RL admission policy to 1e-6 on identical logits; the score is invariant to total mass while the old form was not; the probe and policy token sets agree; ranking is deterministic and ordered by preference; v2 artifacts are refused by the M0 gate.

## 4. Schema migration

`workspace_measurement_metadata` v2 -> v3, with the score definition recorded in every artifact. The M0 gate refuses v2 by name, so a v2 measurement can never gate a v3 campaign and the two definitions cannot be mixed in one table.

## 5. Corrected Qwen3-8B Binary Metacognitive Alignment

Classification: **CORE_SURVIVES**. Re-measured from the locked adapters; no retraining, and the ID checkpoint selection was never affected because it scores through `binary_action_logits`.

| Seed | V before | V after | ΔV reported | **ΔV corrected** | gate |
|---|---:|---:|---:|---:|:--:|
| 0 | 0.4020 | 0.5662 | +0.2247 | **+0.1642** | PASS |
| 1 | 0.4020 | 0.6889 | +0.3473 | **+0.2868** | PASS |
| 2 | 0.4020 | 0.5893 | +0.2477 | **+0.1873** | PASS |

Decoupled mean ΔV **+0.21277** (was +0.27322), sample SD 0.06518. All three seeds still clear the predeclared gate.

Compositional weakens under correction: mean ΔV +0.08606, and only seed 1 clears 0.15.

## 6. Corrected Qwen3-32B: the gate reverses, the outcome is AMBER

| | Reported (v2) | **Corrected (v3)** |
|---|---:|---:|
| Decoupled V | 0.6571 | **0.33658** |
| Decoupled W_rr | 0.6919 | 0.69186 |
| gap W − V | +0.0348 | **+0.35528** |
| M0 decision | `SCALE_BOUNDARY` | **`MISALIGNMENT_REGIME`** |

The campaign was stopped on a corrupted measurement. Re-run under the corrected score it trained, locked at step 100 on ID validation only, and consumed its single OOD attempt.

| Condition | V before | V after | ΔV | 95% CI | ΔW | full-context QA drop |
|---|---:|---:|---:|---|---:|---:|
| decoupled | 0.3366 | 0.4320 | +0.0954 | [+0.0241, +0.1764] | +0.00116 | +0.00 pp |
| compositional | 0.3315 | 0.3139 | -0.0176 | [-0.0847, +0.0549] | +0.00147 | +0.00 pp |

Decision **AMBER**: directional gain with no-harm passed, below GREEN effect size. The effect is real and its interval excludes zero, but it does not reach the +0.15 gate and `V_after` stays below 0.50. No tuning is authorised by AMBER.

## 7. Corrected scale sweep

| Model | V | V_raw | W_rr | gap (W−V) |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 0.3174 | 0.2833 | 0.5607 | +0.2433 |
| Qwen3-4B | 0.4862 | 0.4405 | 0.5291 | +0.0429 |
| Qwen3-8B | 0.4020 | 0.5002 | 0.6545 | +0.2525 |
| Qwen3-14B | 0.3480 | 0.4474 | 0.6677 | +0.3196 |
| Qwen3-32B | 0.3366 | 0.5261 | 0.6919 | +0.3553 |
| Qwen3-30B-A3B (sparse, diagnostic) | 0.2855 | 0.4133 | 0.6570 | +0.3715 |

The 14B→32B step in the chat channel is **-0.0115** with a 95% CI of [-0.0766, +0.0468], which includes zero: **the reported jump does not exist.** Corrected V has no scale trend (slope -0.0255 per decade, R² 0.034).

The gap instead **widens** with scale (slope +0.1519 per decade, R² 0.386), because W_rr improves while V does not. The corrected finding is the opposite of the retracted one: the workspace-report dissociation grows with scale rather than closing.

## 8. Corrected RL-QA comparisons

Only the Original arm changed: the RL-QA adapters are the ones already locked, and the policy scorer never used the defective path. Because the Original arm's budget-2 selected sets are chosen by that score, its QA accuracy changed too, so this is a re-evaluation and not an arithmetic correction.

### Qwen3-8B

Original QA@2 under the corrected score: **0.0147**.

| Seed | QA delta reported | **QA delta corrected** | admission AUC delta reported | **corrected** | 95% CI | McNemar p | full-context drop | verdict |
|---|---:|---:|---:|---:|---|---:|---:|:--:|
| 0 | +8.82 pp | **+7.35 pp** | +0.49433 | **+0.43385** | [+0.3743, +0.5022] | 0.0625 | +1.47 pp | PASS |
| 1 | +7.35 pp | **+5.88 pp** | +0.48210 | **+0.42162** | [+0.3645, +0.4855] | 0.125 | +0.00 pp | PASS |
| 2 | +10.29 pp | **+8.82 pp** | +0.52038 | **+0.45990** | [+0.4086, +0.5203] | 0.03125 | +1.47 pp | PASS |

### Qwen3-32B

Original QA@2 under the corrected score: **0.0735**.

| Seed | QA delta reported | **QA delta corrected** | admission AUC delta reported | **corrected** | 95% CI | McNemar p | full-context drop | verdict |
|---|---:|---:|---:|---:|---|---:|---:|:--:|
| 0 | +0.00 pp | **+4.41 pp** | +0.04483 | **+0.36533** | [+0.3056, +0.4423] | 0.375 | +0.00 pp | ADMISSION_POSITIVE_QA_UNRESOLVED |

The 8B classification is **RL_RESULT_SURVIVES**: all three seeds still clear +5 pp with admission intervals excluding zero, at slightly smaller effect sizes. The 32B classification changes from the reported FAIL to **ADMISSION_POSITIVE_QA_UNRESOLVED**: admission improves by +0.365 with an interval far from zero, where the reported value was +0.045 with an interval spanning zero, and QA moves +4.41 pp against a +5 pp threshold. That is a boundary result requiring review, not the absence of transfer that was reported.

## 9. Objective study

The Binary/Soft/Pairwise/Listwise objective study is **NOT_PRESENT** in this repository; the only nearby module, `src/analysis/mixed_pool.py`, evaluates mixed-provenance admission pools and does not use the verbal probe for winner classification. Nothing to re-score here.

## 10. Claims

**Surviving.** `W_rr` is unaffected throughout. 8B Binary Metacognitive Alignment still repairs the corrected verbal report across all three seeds. 8B RL-QA still passes on all three seeds.

**Withdrawn.** The 14B→32B verbal jump, the sharp scale transition and its chat-pathway localisation, the sparse-model interpretation built on the old score, the 32B `SCALE_BOUNDARY` verdict, and the 32B RL-QA `FAIL` verdict.

**New and requiring review before use.** The corrected sweep shows the workspace-report gap *widening* with scale rather than closing, which is the opposite of the withdrawn claim and has not yet been reviewed.

**Pending.** Nothing from this correction campaign is pending measurement; the open items are decisions, not data.

## 11. Environment

- Repository commit: `ebf03416c7baf1f2ca5af8a6370868f67fac518f`
- Test suite: 219 passing

## 12. Authorisation

Every job in this correction campaign was measurement or a re-run explicitly authorised by the recovery plan. No 14B training was launched, no model above 32B was touched, no RL-QA policy was retrained, and no threshold was revisited after seeing the measurement it gates. The 32B metacognitive campaign consumed exactly one OOD attempt; its predecessor, stopped on the corrupted gate, consumed none.
