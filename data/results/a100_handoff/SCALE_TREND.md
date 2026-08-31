> **WITHDRAWN — the transition reported here does not exist.**
> Every `V` in this document was computed with the defective probe. Corrected,
> the 14B->32B step is **-0.0115 with a confidence interval spanning zero**, and
> `V` has no scale trend at all (slope -0.026 per decade, R^2 0.034). The
> chat-pathway localisation and the sparse-model interpretation built on these
> values are withdrawn with it.
> The corrected sweep shows the workspace-report gap **widening** with scale
> (+0.152 per decade), the opposite of the claim made here.
> Corrected report: `data/results/a100_next_boundary_campaign/shared/CORRECTED_SCALE_SWEEP_REPORT.md`.
> Root cause: [`CORRECTIONS.md`](../../../CORRECTIONS.md).

# Metacognitive reporting gap across Qwen3 scale

Measurement only, no training. `V` is the verbal report AUC, `W_rr` the workspace readout AUC, and `gap = W_rr - V` is what Binary Metacognitive Alignment repairs. `V_raw` is the template-free readout.

## explicit

| Model | params | V | V_raw | W_rr | gap (W-V) |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.7B | 0.6317 | 0.5713 | 0.4818 | -0.1500 |
| Qwen3-4B | 4.0B | 0.4771 | 0.6325 | 0.5090 | +0.0318 |
| Qwen3-8B | 8.2B | 0.5010 | 0.6463 | 0.5090 | +0.0081 |
| Qwen3-14B | 14.8B | 0.4341 | 0.6829 | 0.5083 | +0.0742 |
| Qwen3-32B | 32.8B | 0.5327 | 0.7061 | 0.5084 | -0.0243 |

Surface-presence baseline: literal-mention AUC 0.4841 (load-bearing literally present 95%, negatives 99%).

Adjacent-scale paired deltas (shared episode draws, 4,000):

| Step | delta V | 95% CI | delta V_raw | 95% CI |
|---|---:|---|---:|---|
| 1.7B -> 4B | -0.1546 * | [-0.2234, -0.0869] | +0.0612 | [-0.0001, +0.1245] |
| 4B -> 8B | +0.0239 | [-0.0429, +0.0890] | +0.0138 | [-0.0279, +0.0552] |
| 8B -> 14B | -0.0669 * | [-0.1184, -0.0158] | +0.0366 | [-0.0219, +0.0938] |
| 14B -> 32B | +0.0986 * | [+0.0393, +0.1589] | +0.0233 | [-0.0312, +0.0789] |

`*` marks an interval that excludes zero.

- V vs log10(params): slope -0.0791 per decade, R² 0.277 over 5 sizes
- gap vs log10(params): slope +0.0966 per decade, R² 0.316 over 5 sizes

## evoked

| Model | params | V | V_raw | W_rr | gap (W-V) |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.7B | 0.4676 | 0.3915 | 0.5409 | +0.0733 |
| Qwen3-4B | 4.0B | 0.3608 | 0.5328 | 0.5744 | +0.2136 |
| Qwen3-8B | 8.2B | 0.3802 | 0.5723 | 0.6204 | +0.2402 |
| Qwen3-14B | 14.8B | 0.4334 | 0.5403 | 0.6390 | +0.2056 |
| Qwen3-32B | 32.8B | 0.6293 | 0.5867 | 0.6139 | -0.0154 |

Surface-presence baseline: literal-mention AUC 0.0072 (load-bearing literally present 0%, negatives 99%).

Adjacent-scale paired deltas (shared episode draws, 4,000):

| Step | delta V | 95% CI | delta V_raw | 95% CI |
|---|---:|---|---:|---|
| 1.7B -> 4B | -0.1068 * | [-0.1899, -0.0280] | +0.1413 * | [+0.0611, +0.2223] |
| 4B -> 8B | +0.0194 | [-0.0529, +0.0864] | +0.0395 | [-0.0139, +0.0967] |
| 8B -> 14B | +0.0532 * | [+0.0008, +0.1100] | -0.0320 | [-0.1013, +0.0348] |
| 14B -> 32B | +0.1959 * | [+0.1242, +0.2735] | +0.0464 | [-0.0168, +0.1109] |

`*` marks an interval that excludes zero.

- V vs log10(params): slope +0.1224 per decade, R² 0.322 over 5 sizes
- gap vs log10(params): slope -0.0550 per decade, R² 0.061 over 5 sizes

## decoupled

| Model | params | V | V_raw | W_rr | gap (W-V) |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.7B | 0.3273 | 0.2834 | 0.5607 | +0.2334 |
| Qwen3-4B | 4.0B | 0.4122 | 0.4405 | 0.5291 | +0.1169 |
| Qwen3-8B | 8.2B | 0.3415 | 0.5002 | 0.6545 | +0.3130 |
| Qwen3-14B | 14.8B | 0.3946 | 0.4474 | 0.6677 | +0.2731 |
| Qwen3-32B | 32.8B | 0.6571 | 0.5261 | 0.6919 | +0.0348 |

Surface-presence baseline: literal-mention AUC 0.0187 (load-bearing literally present 0%, negatives 96%).

Adjacent-scale paired deltas (shared episode draws, 4,000):

| Step | delta V | 95% CI | delta V_raw | 95% CI |
|---|---:|---|---:|---|
| 1.7B -> 4B | +0.0849 * | [+0.0057, +0.1571] | +0.1571 * | [+0.0819, +0.2324] |
| 4B -> 8B | -0.0707 * | [-0.1376, -0.0065] | +0.0597 | [-0.0045, +0.1280] |
| 8B -> 14B | +0.0530 | [-0.0126, +0.1218] | -0.0528 | [-0.1324, +0.0205] |
| 14B -> 32B | +0.2625 * | [+0.1696, +0.3634] | +0.0787 * | [+0.0266, +0.1308] |

`*` marks an interval that excludes zero.

- V vs log10(params): slope +0.2066 per decade, R² 0.585 over 5 sizes
- gap vs log10(params): slope -0.0801 per decade, R² 0.118 over 5 sizes

## compositional

| Model | params | V | V_raw | W_rr | gap (W-V) |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.7B | 0.2806 | 0.2411 | 0.4162 | +0.1355 |
| Qwen3-4B | 4.0B | 0.2897 | 0.3132 | 0.5050 | +0.2153 |
| Qwen3-8B | 8.2B | 0.3262 | 0.3943 | 0.5466 | +0.2204 |
| Qwen3-14B | 14.8B | 0.3403 | 0.2831 | 0.5086 | +0.1683 |
| Qwen3-32B | 32.8B | 0.6827 | 0.3145 | 0.5532 | -0.1295 |

Surface-presence baseline: literal-mention AUC 0.0120 (load-bearing literally present 0%, negatives 98%).

Adjacent-scale paired deltas (shared episode draws, 4,000):

| Step | delta V | 95% CI | delta V_raw | 95% CI |
|---|---:|---|---:|---|
| 1.7B -> 4B | +0.0090 | [-0.0937, +0.1046] | +0.0721 | [-0.0149, +0.1604] |
| 4B -> 8B | +0.0365 | [-0.0358, +0.1112] | +0.0811 * | [+0.0225, +0.1399] |
| 8B -> 14B | +0.0141 | [-0.0506, +0.0856] | -0.1112 * | [-0.1850, -0.0361] |
| 14B -> 32B | +0.3425 * | [+0.2719, +0.4237] | +0.0314 | [-0.0425, +0.1029] |

`*` marks an interval that excludes zero.

- V vs log10(params): slope +0.2735 per decade, R² 0.642 over 5 sizes
- gap vs log10(params): slope -0.1813 per decade, R² 0.383 over 5 sizes

## Diagnostic: sparse (MoE) model, reported separately

Qwen3-30B-A3B activates 3B of 30B parameters per token. It is NOT a substitute for the dense 32B scale point -- both campaign plans forbid that substitution -- and it is excluded from every fit above. It is here to separate total parameters from active compute.

| Model | Condition | V | V_raw | W_rr | gap (W-V) |
|---|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | explicit | 0.4684 | 0.6442 | 0.5326 | +0.0642 |
| Qwen3-30B-A3B | evoked | 0.4008 | 0.4762 | 0.6296 | +0.2287 |
| Qwen3-30B-A3B | decoupled | 0.2755 | 0.4133 | 0.6570 | +0.3815 |
| Qwen3-30B-A3B | compositional | 0.2086 | 0.2643 | 0.5174 | +0.3088 |

On Decoupled the sparse model reports worse than every dense model measured, including 1.7B, while its workspace readout matches the 8B dense model. Total parameter count does not carry the transition; active compute tracks it. The sparse model also differs in post-training, so this isolates the comparison rather than settling it -- but the within-model dissociation (workspace fine, report poor) does not depend on cross-model comparability.

## Caveats

- Five sizes from one model family; not compute-matched, so this is a scale trend rather than a scaling law, and it is not extrapolated.
- Sizes may differ in training data mix and post-training recipe; 'scale' here is a proxy for everything that changes between releases.
- Measurement only: no training, no adapters, no OOD-driven choices.
