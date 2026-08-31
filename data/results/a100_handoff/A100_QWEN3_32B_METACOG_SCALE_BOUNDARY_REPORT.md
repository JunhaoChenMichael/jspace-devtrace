> **WITHDRAWN — this verdict was produced by a defective measurement.**
> The `SCALE_BOUNDARY` decision rested on a Decoupled `V` of 0.6571. Corrected,
> `V` is **0.3366** and the gap is **+0.3553**, far above the 0.10 threshold, so
> the decision should have been `MISALIGNMENT_REGIME` and the campaign should
> have trained. Re-run under the corrected score it did train, and returned
> **AMBER** (`delta V` +0.0954, CI [+0.024, +0.176], below the +0.15 gate with
> `V_after` 0.432).
> Corrected campaign: `data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0/`.
> Root cause: [`CORRECTIONS.md`](../../../CORRECTIONS.md).

# Qwen3-32B Binary Metacognitive Alignment: seed-0 scaling gate

Report schema: `metacog-32b-scale-boundary/v1`. M0 decision: **SCALE_BOUNDARY**.

The plan gates a new scale point on a repairable reporting gap, not on reproducing the 8B numbers. Qwen3-32B does not have one, so training was forbidden and no OOD battery was opened. `automatic_seed_expansion_authorized = false`.

## 1. The gate

| Quantity | Value |
|---|---:|
| Decoupled V (before) | 0.65708 |
| Decoupled W_rr (before) | 0.69186 |
| **Reporting gap (W - V)** | **+0.03478** |
| Required gap | 0.10 |
| Decision | SCALE_BOUNDARY |

The historical 8B/paper values (V 0.337, W_rr 0.654) appear in the gate record as context only; a different model has no obligation to match them.

## 2. M0 measurement, all four conditions

| Condition | V | W_rr | gap |
|---|---:|---:|---:|
| explicit | 0.5327 | 0.5084 | -0.0243 |
| evoked | 0.6293 | 0.6139 | -0.0154 |
| decoupled | 0.6571 | 0.6919 | +0.0348 |
| compositional | 0.6827 | 0.5532 | -0.1295 |

Against the completed 8B campaign, on the same 335 Decoupled candidates:

| | Qwen3-8B | Qwen3-32B |
|---|---:|---:|
| Decoupled V | 0.3415 | 0.6571 |
| Decoupled W_rr | 0.6545 | 0.6919 |
| gap | +0.3130 | +0.0348 |

At 8B the verbal report is *below chance*: it is anti-correlated with utility, not merely uninformative. At 32B it is well above chance while the workspace readout barely moves.

## 3. Why: the reporter jumps between 14B and 32B

A five-size measurement sweep (1.7B / 4B / 8B / 14B / 32B, no training) locates the transition. Adjacent-scale deltas use shared episode draws, so the difference is paired.

| Condition | 14B->32B delta V (chat) | 95% CI | 14B->32B delta V_raw | 95% CI |
|---|---:|---|---:|---|
| evoked | +0.1959 | [+0.1242, +0.2735] | +0.0464 | [-0.0168, +0.1109] |
| decoupled | +0.2625 | [+0.1696, +0.3634] | +0.0787 | [+0.0266, +0.1308] |
| compositional | +0.3425 | [+0.2719, +0.4237] | +0.0314 | [-0.0425, +0.1029] |

Every other adjacent step is small and mostly not distinguishable from zero, so this is a transition between 14B and 32B rather than a smooth trend. The jump is several times larger in the chat-template channel than in the template-free `V_raw` channel, which places it in the instruct pathway rather than in the underlying next-token computation.

## 4. Why the small models score below chance

| Condition | load-bearing literally in context | negatives | literal-mention AUC |
|---|---:|---:|---:|
| explicit | 95% | 99% | 0.4841 |
| evoked | 0% | 99% | 0.0072 |
| decoupled | 0% | 96% | 0.0187 |
| compositional | 0% | 98% | 0.0120 |

The benchmark makes the load-bearing concept an unstated bridge, so 'does this word appear in the passage' is an almost perfect ANTI-predictor of utility on the non-Explicit conditions. A model answering from surface prominence therefore scores below chance, which is what the small models do. On Explicit, where the feature is uninformative, V sits near 0.5 at every size - the control that rules out a pipeline artefact.

This is an association, not a demonstrated mechanism: it shows where the small models' scores land relative to a surface baseline, not that they compute that feature.

## 5. What this means for the claim

The 8B dissociation survives: the workspace readout tracks utility while the verbal report does not. What the sweep adds is a boundary. The deficit that Binary Metacognitive Alignment repairs is not a fixed property of the architecture; it disappears between 14B and 32B without any intervention. Trained 8B reporters reach V about 0.57-0.69 on Decoupled; untrained 32B is 0.657 - the intervention buys roughly what this scale step gives for free.

That favours reading the effect as a capability limit in the verbal channel rather than a persistent access barrier, though the present evidence does not settle it: the sizes differ in post-training as well as in parameters.

## 6. Artifacts

- Run directory: `data/results/metacog32_a100_seed0/seed0`
- M0 gate: `data/results/metacog32_a100_seed0/seed0/m0/gate.json`, `data/results/metacog32_a100_seed0/seed0/m0/gate.md`
- Decision ledger: `data/results/metacog32_a100_seed0/seed0/decision_ledger.jsonl`
- Scale sweep: `data/results/scale_sweep/`, analysis in `SCALE_TREND.md`

## 7. Stop

No training, no adapter, no OOD attempt was consumed. Reopening this scale point requires a pre-registration amendment, not a rerun: the plan forbids manufacturing a gap by changing prompts, thinking mode, token scoring or the workspace readout.

## 8. Environment

- Repository commit: `b85e5b59612c3a835f6d3fefc8923dcc386781f0`
- Packages: accelerate 1.14.0, datasets 5.0.1, numpy 2.5.2, peft 0.13.2, scikit-learn 1.9.0, scipy 1.18.1, torch 2.7.1+cu126, transformers 4.57.6
