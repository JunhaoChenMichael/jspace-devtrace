# Impact report: the verbal-score guard epsilon

## What the defect is

`src/experiments/measure.py` computed the verbal report as

```python
return py / (py + pn + 1e-9)
```

where `py` and `pn` are the summed **full-vocabulary softmax probabilities** of the
yes and no token variants. The `1e-9` was a division guard.

On this probe the model spends almost all of its probability elsewhere. Measured
directly on Qwen3-32B, a representative candidate has `P(yes) = 5.86e-13` and
`P(no) = 5.18e-14` — three to four orders of magnitude below the guard. The
denominator therefore reduces to the epsilon, and the function returns
approximately `py * 1e9`.

**The quantity we have been reporting as `V` is a monotone function of the
absolute yes-probability, not the yes-versus-no ratio it documents.**

Two independent confirmations:

- On the same logits, `py/(py+pn+1e-9)` returns 0.0006 while the ratio is 0.9189.
  The RL admission policy, which computes the ratio in log space and has no
  epsilon, returns 0.9189.
- Ranking by the reported `V` and ranking by absolute `P(yes)` give AUCs that
  agree to four decimals at every scale (8B 0.3415 / 0.3413, 32B 0.6571 / 0.6571),
  and **100%** of candidates at every scale sit below the epsilon.

## What is affected

| Component | Affected | Why |
|---|---|---|
| Metacognitive `V` (all scales, all conditions) | **Yes** | computed by `verbal_salience` |
| The five-size scale sweep and the MoE diagnostic | **Yes** | same function |
| RL-QA OOD **Original** arm, both 8B and 32B | **Yes** | `original_verbal_source = precomputed_v_ref`, and `v_ref = row["V"]` from the measurement files |
| RL-QA **selected sets** for the Original arm | **Yes** | budget-2 selection ranks on that score, so the sets it admitted differ |
| `W_rr` workspace readout | No | reciprocal rank, no epsilon |
| Frozen teacher labels | No | derived from `W_rr`, not `V` |
| Trained adapters | No | the training objective never consumed `V` |
| RL policy's own admission scores | No | `binary_action_logits`, log space, no epsilon |
| Full-context QA no-harm | No | generation and grading, not `V` |

## Corrected results

### Qwen3-8B Binary Metacognitive Alignment — the conclusion survives

Re-measured with the locked adapters; no retraining was needed.

| Seed | V before | V after | ΔV reported | **ΔV corrected** | gate ΔV ≥ 0.15 | gate V_after > 0.50 |
|---|---:|---:|---:|---:|:--:|:--:|
| 0 | 0.4020 | 0.5662 | +0.2247 | **+0.1642** | PASS | PASS |
| 1 | 0.4020 | 0.6889 | +0.3473 | **+0.2868** | PASS | PASS |
| 2 | 0.4020 | 0.5893 | +0.2477 | **+0.1873** | PASS | PASS |

Mean ΔV moves from **+0.27322 to +0.21277** (sample SD 0.06518, essentially
unchanged because the correction is a constant shift in the shared baseline).
All three seeds remain GREEN; seed 1 remains Strong GREEN.

The decomposition explains why only the baseline moved: before training, 100% of
candidates sit below the epsilon; after training, **0%** do. Alignment training
raises the absolute yes/no mass out of the affected regime, so `V_after` was
already the true ratio (v2 and v3 agree to four decimals for every trained seed).
Only `V_before` was corrupted, and it was shared by all three seeds.

### Qwen3-32B Binary Metacognitive Alignment — the verdict reverses

| | Reported | Corrected |
|---|---:|---:|
| Decoupled V | 0.6571 | **0.3366** |
| Decoupled W_rr | 0.6919 | 0.6919 |
| Reporting gap W − V | +0.0348 | **+0.3553** |
| M0 decision | `SCALE_BOUNDARY` | **`MISALIGNMENT_REGIME`** |

The gap is far above the 0.10 threshold. **The campaign should have trained, and
was stopped on a corrupted measurement.**

### The scale transition — the headline was an artifact

Corrected Decoupled V for the untrained models:

| Model | V as reported | **V corrected** |
|---|---:|---:|
| Qwen3-8B | 0.3415 | **0.4020** |
| Qwen3-14B | 0.3946 | **0.3480** |
| Qwen3-32B | 0.6571 | **0.3366** |

**There is no 14B→32B jump.** Corrected values are flat and below chance across
the three scales, drifting slightly downward. The reported +0.263 step, the
"chat-channel-only" narrowing and the sparse-model contrast were all computed on
the epsilon-dominated quantity, and none of them survive as stated.

What 32B actually does differently is place more absolute probability mass on the
yes/no tokens — which is what the corrupted score was measuring.

### RL-QA — direction unknown until the Original arm is re-measured

Both OOD evaluations took the Original arm from the corrupted `v_ref`, so the
Original admission AUC, its budget-2 selected sets, and every delta computed
against it need re-derivation:

- 8B reported Original admission AUC 0.3415; corrected baseline is 0.4020.
- 32B reported Original admission AUC 0.6571; corrected baseline is 0.3366. The
  RL-QA arm itself was measured through the unaffected policy path, so the 32B
  admission delta could move from +0.045 to roughly +0.37 — the opposite reading
  from the one reported.

**No corrected RL-QA delta is quoted here.** The selected sets change with the
score, so the QA arm cannot be corrected arithmetically; both scales need one
re-evaluation of the Original arm.

## What was fixed

- `verbal_salience` and `verbal_salience_raw` now compute
  `sigmoid(logsumexp(logits[yes]) − logsumexp(logits[no]))`: the exact ratio, in
  log space, with no guard epsilon.
- Measurement metadata moves from `workspace_measurement_metadata.v2` to `v3` and
  records the score definition. The M0 gate **refuses v2 artifacts** with the
  reason, so a v2 measurement can never gate a v3 campaign.
- Three regression tests: the epsilon regime returns the ratio and not the mass;
  the probe agrees with the RL admission policy to 1e-6 on identical logits; v2
  artifacts are refused. Suite: 216 passing under both model configurations.

## What must happen before any further campaign

1. Re-measure M0 under v3 for every scale used in a claim.
2. Re-run the 32B metacognitive campaign, which the corrected gate authorises.
3. Re-evaluate the RL-QA Original arm at 8B and 32B, then re-derive both gates.
4. Withdraw or amend the scale-transition claim in PR #2.
5. Amend PR #1 with the corrected 8B numbers; its GREEN verdict stands.

Phase 1 of the boundary campaign (Qwen3-14B M0) is **not started**: gating a new
scale on the corrupted measurement would compound the error.

## Provenance

Raw per-candidate records, including `p_yes`, `p_no` and the yes/no mass for
every measurement above:

```
data/results/a100_next_boundary_campaign/shared/v_correction/
data/results/a100_next_boundary_campaign/shared/reporter_interface_audit/
```
