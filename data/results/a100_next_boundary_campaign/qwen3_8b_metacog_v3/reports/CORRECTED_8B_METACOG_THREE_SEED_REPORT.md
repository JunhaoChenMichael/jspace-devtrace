# Corrected Qwen3-8B Binary Metacognitive Alignment: three seeds

Classification: **CORE_SURVIVES**. Measured under the corrected verbal
score (schema v3) using the locked adapters; no retraining was performed.

## Decoupled (primary)

| Seed | V before | V after | ΔV as reported | **ΔV corrected** | ΔV ≥ 0.15 | V_after > 0.50 |
|---|---:|---:|---:|---:|:--:|:--:|
| 0 | 0.4020 | 0.5662 | +0.2247 | **+0.1642** | PASS | PASS |
| 1 | 0.4020 | 0.6889 | +0.3473 | **+0.2868** | PASS | PASS |
| 2 | 0.4020 | 0.5893 | +0.2477 | **+0.1873** | PASS | PASS |

Mean ΔV **+0.21277**, sample SD 0.06518. All seeds pass: **True**.

## Compositional (diagnostic)

| Seed | V before | V after | ΔV as reported | **ΔV corrected** | ΔV ≥ 0.15 | V_after > 0.50 |
|---|---:|---:|---:|---:|:--:|:--:|
| 0 | 0.4119 | 0.4265 | +0.1003 | **+0.0145** | FAIL | FAIL |
| 1 | 0.4119 | 0.5992 | +0.2730 | **+0.1873** | PASS | PASS |
| 2 | 0.4119 | 0.4683 | +0.1421 | **+0.0564** | FAIL | FAIL |

Mean ΔV **+0.08606**, sample SD 0.09013. All seeds pass: **False**.

## Probability-mass decomposition

| Condition | mass below epsilon, before | after | mass-alone AUC before | after |
|---|---:|---:|---:|---:|
| decoupled | 100% | 0% | 0.3755 | 0.4620 |
| compositional | 100% | 0% | 0.3842 | 0.4144 |

Before training every candidate sits below the guard epsilon, so the old score
ranked absolute yes-probability. After training none do, which is why the old and
corrected scores agree to four decimals for every trained seed: only the shared
baseline was corrupted, and correcting it shifts all three seeds by the same amount.

Ranking by yes/no mass alone reaches only ~0.38 before and ~0.44 after, well below
the corrected V of 0.57-0.69, so the corrected effect is carried by the yes-versus-no
preference and not by the mass channel.

## What this changes

The qualitative conclusion is unchanged: all three seeds pass the predeclared gate
and seed 1 remains Strong GREEN. The effect size is smaller, and every table quoting
the old mean of +0.273 must be updated to **+0.213**.
