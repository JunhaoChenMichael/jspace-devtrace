# Corrections

This index records claims in this repository that have been corrected or
withdrawn, and why. It exists because the affected reports were merged into
`main` before the defect was found, so a reader can otherwise encounter a
retracted number with nothing marking it as retracted.

Original reports are preserved. Each carries a banner pointing here; none has
been silently rewritten.

## 2026-08-30 — verbal-salience guard epsilon

### The defect

`src/experiments/measure.py` computed the verbal report `V` as

```python
py / (py + pn + 1e-9)
```

where `py` and `pn` are summed full-vocabulary softmax probabilities of the yes
and no token variants, and the `1e-9` was a division guard.

On this probe the model places almost all of its probability elsewhere. Measured
on Qwen3-32B, a representative candidate has `P(yes) = 5.86e-13` and
`P(no) = 5.18e-14` — three to four orders of magnitude below the guard. The
denominator reduced to the epsilon and the function returned approximately
`py * 1e9`: **a monotone function of the absolute yes-probability, not the
yes-versus-no ratio it documented.** Across every scale measured, 100% of
candidates sat in that regime.

Two independent confirmations. On identical logits the old form returns 0.0006
where the ratio is 0.9189, which is what the RL admission policy returns through
its own log-space path. And ranking by the reported `V` matches ranking by
absolute `P(yes)` to four decimals at every scale.

### The fix

```python
def _yes_vs_no(logits, yes_ids, no_ids):
    yes = torch.logsumexp(logits[yes_ids], dim=0)
    no = torch.logsumexp(logits[no_ids], dim=0)
    return float(torch.sigmoid(yes - no))
```

A repository-wide audit found the same defect in `locomo_gate`,
`longmemeval_gate`, `vprobe_robust` and `measure_vlm`, and the same defect class
in `vrating_baseline` where the digit mass sat under its own guard. All are
corrected through the shared helper.

Measurement metadata moved from `workspace_measurement_metadata.v2` to `v3` and
records the score definition. **The M0 gate refuses v2 artifacts by name**, so a
v2 measurement can never gate a v3 campaign and the two cannot be mixed in one
table. Seven regression tests lock the behaviour; the suite is 219 passing.

### What changed

| Claim | Status | Corrected value |
|---|---|---|
| 8B Binary Metacognitive Alignment, three seeds GREEN | **CORRECTED** | mean `delta V` +0.273 → **+0.213**; all three still pass, seed 1 still Strong |
| 8B RL-QA, three seeds pass | **CORRECTED** | QA +8.82/+7.35/+10.29 → **+7.35/+5.88/+8.82 pp**; all still clear +5 pp |
| 32B metacognitive `SCALE_BOUNDARY` | **WITHDRAWN** | gap +0.0348 → **+0.3553**; decision is `MISALIGNMENT_REGIME`, and the re-run returns **AMBER** |
| 32B RL-QA `FAIL` | **RECLASSIFIED** | admission +0.045 (CI spans 0) → **+0.365** [+0.306, +0.442]; QA +0.00 → **+4.41 pp**; now `ADMISSION_POSITIVE_QA_UNRESOLVED` |
| 14B→32B verbal jump, sharp scale transition, chat-pathway localisation, sparse-model interpretation | **WITHDRAWN** | the step is **−0.0115 with a CI spanning zero**; `V` has no scale trend |
| `W_rr` workspace readout | **UNAFFECTED** | reciprocal rank, no epsilon |
| Teacher labels, trained adapters, RL policy scores, full-context QA no-harm | **UNAFFECTED** | none consumed the defective path |

### A new result, not yet reviewed

The corrected sweep shows the workspace–report gap **widening** with scale
(+0.152 per decade, R² 0.386): `W_rr` improves from 0.561 to 0.692 across
1.7B→32B while `V` stays flat and below chance. That is the opposite of the
withdrawn claim. It has not been reviewed and should not be cited yet.

### Where the corrected results live

```
data/results/a100_next_boundary_campaign/
├── MEASUREMENT_BUG_CORRECTION_FINAL_SUMMARY.md
├── shared/
│   ├── V_EPSILON_DEFECT_IMPACT_REPORT.md
│   ├── CORRECTED_SCALE_SWEEP_REPORT.md
│   ├── reporter_interface_audit/     per-candidate probe-vs-policy comparison
│   └── v_correction/                 per-candidate p_yes, p_no and yes/no mass
├── qwen3_8b_metacog_v3/reports/CORRECTED_8B_METACOG_THREE_SEED_REPORT.md
├── qwen3_32b_metacog_v3/seed0/       corrected M0 gate, training, one-shot OOD
└── qwen3_rlqa_v3/                    corrected Original arm at both scales
```

### Open decisions

1. Whether to accept the 32B metacognitive **AMBER**. The protocol forbids
   tuning after AMBER, and none was done.
2. Whether to adopt the 32B RL-QA reclassification from `FAIL` to
   `ADMISSION_POSITIVE_QA_UNRESOLVED`.
3. Whether the corrected "gap widens with scale" finding is ready to be claimed.
