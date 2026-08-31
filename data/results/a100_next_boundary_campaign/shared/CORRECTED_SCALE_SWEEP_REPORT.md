# Metacognitive reporting gap across Qwen3 scale

Measurement only, no training. `V` is the verbal report AUC, `W_rr` the workspace readout AUC, and `gap = W_rr - V` is what Binary Metacognitive Alignment repairs. `V_raw` is the template-free readout.

## explicit

| Model | params | V | V_raw | W_rr | gap (W-V) |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.7B | 0.5951 | 0.5713 | 0.4818 | -0.1133 |
| Qwen3-4B | 4.0B | 0.4812 | 0.6325 | 0.5090 | +0.0277 |
| Qwen3-8B | 8.2B | 0.5358 | 0.6463 | 0.5090 | -0.0268 |
| Qwen3-14B | 14.8B | 0.4454 | 0.6829 | 0.5083 | +0.0629 |
| Qwen3-32B | 32.8B | 0.5362 | 0.7061 | 0.5084 | -0.0278 |

Surface-presence baseline: literal-mention AUC 0.4841 (load-bearing literally present 95%, negatives 99%).

Adjacent-scale paired deltas (shared episode draws, 4,000):

| Step | delta V | 95% CI | delta V_raw | 95% CI |
|---|---:|---|---:|---|
| 1.7B -> 4B | -0.1138 * | [-0.1789, -0.0496] | +0.0612 | [-0.0001, +0.1245] |
| 4B -> 8B | +0.0546 | [-0.0031, +0.1116] | +0.0137 | [-0.0279, +0.0552] |
| 8B -> 14B | -0.0904 * | [-0.1588, -0.0237] | +0.0366 | [-0.0219, +0.0939] |
| 14B -> 32B | +0.0908 * | [+0.0299, +0.1553] | +0.0233 | [-0.0312, +0.0789] |

`*` marks an interval that excludes zero.

- V vs log10(params): slope -0.0494 per decade, R² 0.181 over 5 sizes
- gap vs log10(params): slope +0.0670 per decade, R² 0.246 over 5 sizes

## evoked

| Model | params | V | V_raw | W_rr | gap (W-V) |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.7B | 0.4391 | 0.3914 | 0.5409 | +0.1018 |
| Qwen3-4B | 4.0B | 0.4966 | 0.5328 | 0.5744 | +0.0778 |
| Qwen3-8B | 8.2B | 0.4521 | 0.5723 | 0.6204 | +0.1682 |
| Qwen3-14B | 14.8B | 0.3993 | 0.5404 | 0.6390 | +0.2397 |
| Qwen3-32B | 32.8B | 0.4258 | 0.5867 | 0.6139 | +0.1880 |

Surface-presence baseline: literal-mention AUC 0.0072 (load-bearing literally present 0%, negatives 99%).

Adjacent-scale paired deltas (shared episode draws, 4,000):

| Step | delta V | 95% CI | delta V_raw | 95% CI |
|---|---:|---|---:|---|
| 1.7B -> 4B | +0.0575 | [-0.0135, +0.1288] | +0.1413 * | [+0.0611, +0.2223] |
| 4B -> 8B | -0.0445 | [-0.1010, +0.0106] | +0.0395 | [-0.0139, +0.0968] |
| 8B -> 14B | -0.0529 | [-0.1159, +0.0086] | -0.0319 | [-0.1015, +0.0350] |
| 14B -> 32B | +0.0266 | [-0.0272, +0.0807] | +0.0464 | [-0.0168, +0.1108] |

`*` marks an interval that excludes zero.

- V vs log10(params): slope -0.0362 per decade, R² 0.248 over 5 sizes
- gap vs log10(params): slope +0.1036 per decade, R² 0.610 over 5 sizes

## decoupled

| Model | params | V | V_raw | W_rr | gap (W-V) |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.7B | 0.3174 | 0.2833 | 0.5607 | +0.2433 |
| Qwen3-4B | 4.0B | 0.4862 | 0.4405 | 0.5291 | +0.0429 |
| Qwen3-8B | 8.2B | 0.4020 | 0.5002 | 0.6545 | +0.2525 |
| Qwen3-14B | 14.8B | 0.3480 | 0.4474 | 0.6677 | +0.3196 |
| Qwen3-32B | 32.8B | 0.3366 | 0.5261 | 0.6919 | +0.3553 |

Surface-presence baseline: literal-mention AUC 0.0187 (load-bearing literally present 0%, negatives 96%).

Adjacent-scale paired deltas (shared episode draws, 4,000):

| Step | delta V | 95% CI | delta V_raw | 95% CI |
|---|---:|---|---:|---|
| 1.7B -> 4B | +0.1688 * | [+0.0906, +0.2482] | +0.1572 * | [+0.0819, +0.2324] |
| 4B -> 8B | -0.0842 * | [-0.1360, -0.0358] | +0.0597 | [-0.0046, +0.1279] |
| 8B -> 14B | -0.0540 | [-0.1199, +0.0134] | -0.0527 | [-0.1324, +0.0206] |
| 14B -> 32B | -0.0115 | [-0.0766, +0.0468] | +0.0787 * | [+0.0266, +0.1308] |

`*` marks an interval that excludes zero.

- V vs log10(params): slope -0.0255 per decade, R² 0.034 over 5 sizes
- gap vs log10(params): slope +0.1519 per decade, R² 0.386 over 5 sizes

## compositional

| Model | params | V | V_raw | W_rr | gap (W-V) |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.7B | 0.2811 | 0.2411 | 0.4162 | +0.1350 |
| Qwen3-4B | 4.0B | 0.4272 | 0.3132 | 0.5050 | +0.0778 |
| Qwen3-8B | 8.2B | 0.4119 | 0.3943 | 0.5466 | +0.1346 |
| Qwen3-14B | 14.8B | 0.3216 | 0.2831 | 0.5086 | +0.1870 |
| Qwen3-32B | 32.8B | 0.3315 | 0.3145 | 0.5532 | +0.2217 |

Surface-presence baseline: literal-mention AUC 0.0120 (load-bearing literally present 0%, negatives 98%).

Adjacent-scale paired deltas (shared episode draws, 4,000):

| Step | delta V | 95% CI | delta V_raw | 95% CI |
|---|---:|---|---:|---|
| 1.7B -> 4B | +0.1461 * | [+0.0767, +0.2137] | +0.0721 | [-0.0149, +0.1604] |
| 4B -> 8B | -0.0153 | [-0.0778, +0.0445] | +0.0811 * | [+0.0225, +0.1399] |
| 8B -> 14B | -0.0904 * | [-0.1717, -0.0082] | -0.1112 * | [-0.1850, -0.0361] |
| 14B -> 32B | +0.0099 | [-0.0633, +0.0801] | +0.0314 | [-0.0425, +0.1029] |

`*` marks an interval that excludes zero.

- V vs log10(params): slope +0.0051 per decade, R² 0.002 over 5 sizes
- gap vs log10(params): slope +0.0871 per decade, R² 0.609 over 5 sizes

## Diagnostic: sparse (MoE) model, reported separately

Qwen3-30B-A3B activates 3B of 30B parameters per token. It is NOT a substitute for the dense 32B scale point -- both campaign plans forbid that substitution -- and it is excluded from every fit above. It is here to separate total parameters from active compute.

| Model | Condition | V | V_raw | W_rr | gap (W-V) |
|---|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | explicit | 0.4725 | 0.6442 | 0.5326 | +0.0601 |
| Qwen3-30B-A3B | evoked | 0.3864 | 0.4762 | 0.6296 | +0.2431 |
| Qwen3-30B-A3B | decoupled | 0.2855 | 0.4133 | 0.6570 | +0.3715 |
| Qwen3-30B-A3B | compositional | 0.2015 | 0.2642 | 0.5174 | +0.3159 |

On Decoupled the sparse model reports worse than every dense model measured, including 1.7B, while its workspace readout matches the 8B dense model. Total parameter count does not carry the transition; active compute tracks it. The sparse model also differs in post-training, so this isolates the comparison rather than settling it -- but the within-model dissociation (workspace fine, report poor) does not depend on cross-model comparability.

## Caveats

- Five sizes from one model family; not compute-matched, so this is a scale trend rather than a scaling law, and it is not extrapolated.
- Sizes may differ in training data mix and post-training recipe; 'scale' here is a proxy for everything that changes between releases.
- Measurement only: no training, no adapters, no OOD-driven choices.
