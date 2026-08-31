# Stage B1 report: RL-QA, seed 0, budget 2

## Decision

**Gate B1: GREEN.** The ID-selected RL-QA checkpoint improves Decoupled QA over
Original by **+8.82 percentage points** under episode-isolated greedy decoding
(`42.65%` versus `33.82%`). This exceeds the predeclared `+5pp` threshold and is
not worse than SFT-W (`41.18%`; RL-QA minus SFT-W `+1.47pp`). The latter
difference is not statistically resolved, so this run establishes viability
against Original, not an RL advantage over SFT-W.

The standard batch-8 throughput evaluation also passes Gate B1 (`+5.88pp`).
Thus the Gate direction does not depend on which of the two recorded decoding
batch sizes is used. Hybrid has **not** been launched; execution stops at the
required manual review point.

## Locked protocol

- Model: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`
- Data: fixed split seed 0, 175 train / 45 ID validation episodes; manifest
  SHA-256 `1988c8af1fc39884a7bd711f335f91db6c890612e6e0a45917488ba4b44abf0f`
- Training: seed 0, `k=2`, `G=8`, temperature 5.0, 300 steps, rank-32 LoRA,
  bf16, LR `1e-6`, KL beta `0.03`, two GRPO epochs
- Reward: `lambda_QA=1`, `lambda_W=0`; workspace reward was logged but never
  optimized
- Policy input: context and candidate concept only; the probe was hidden
- Recall: the same adapter-disabled frozen base model for every policy
- Validation: all 45 ID episodes at steps 0, 100, 200, and 300; OOD was not
  inspected until the checkpoint was locked

The formal artifact validator passed with no errors or warnings. The run wrote
304 metric records, 2,400 rollout records, and all required manifests/configs.

## Training diagnostics

| Diagnostic | Result |
|---|---:|
| Mixed-QA groups | 163/300 = **54.33%** |
| Nonzero reward-variance groups | 163/300 = **54.33%** |
| Unique exact sets per group | median **3**, mean **3.38** |
| One-set groups | 12/300 = **4.00%** |
| Mean normalized exact-set entropy | **0.6160** |
| Mean containment | **62.17%** |
| Mean KL | **0.03052** |
| Beta times mean KL | **0.000916** |
| Groups with nonzero finite gradient | **100%** |
| Unique training episodes sampled | 134/175 |

For the first 50 steps, mixed reward was 50%, median unique sets was 4, and
mean containment was 58%. For the last 50 steps, these were 56%, 3, and 65%.
The slight diversity reduction did not become collapse: the last two 25-step
windows each had only one one-set group, and mixed reward remained available.
There was therefore no sparse-reward, identical-set, exploding-KL, or
validation-decline trigger, and training was not extended.

Across all 2,400 sampled decisions, 908 (37.83%) failed to retain a
load-bearing item, 267 (11.13%) retained one but failed recall/composition, and
1,225 (51.04%) both retained one and answered correctly. Conditional QA was
82.10% with load-bearing retention versus 31.39% without it, a **+50.72pp**
gap. This independently confirms that the downstream reward remained strongly
coupled to admission decisions during B1.

## ID checkpoint selection

| Step | QA accuracy | Containment | Workspace set reward |
|---:|---:|---:|---:|
| 0 | 0.68889 | 0.71111 | 0.56481 |
| 100 | 0.68889 | 0.68889 | 0.55093 |
| 200 | **0.71111** | **0.77778** | 0.58148 |
| 300 | **0.71111** | **0.77778** | 0.58426 |

Steps 200 and 300 tied on the primary ID QA metric. The predeclared
first-maximum rule selected `best-step-200` before any OOD evaluation.

## Locked OOD evaluation

The table uses the corrected primary evaluation with one selected-set prompt
per greedy-generation batch. This prevents unrelated prompts in a batch from
changing a bf16 greedy answer through padding/batch-shape finite-precision
effects.

| Battery | Method | Pooled AUC | Top-2 containment | QA accuracy |
|---|---|---:|---:|---:|
| Decoupled | Original | 0.48590 | 0.45588 | 0.33824 |
| Decoupled | SFT-W | 0.52715 | **0.54412** | 0.41176 |
| Decoupled | RL-W | 0.49331 | 0.45588 | 0.36765 |
| Decoupled | **RL-QA** | **0.57306** | 0.52941 | **0.42647** |
| Decoupled | Workspace | 0.64119 | 0.52941 | 0.35294 |
| Decoupled | Oracle | 1.00000 | 1.00000 | 0.36765 |
| Compositional | Original | 0.34431 | 0.25000 | 0.38462 |
| Compositional | SFT-W | 0.39556 | 0.36538 | **0.44231** |
| Compositional | RL-W | 0.35508 | 0.25000 | 0.38462 |
| Compositional | **RL-QA** | **0.45482** | **0.44231** | **0.44231** |
| Compositional | Workspace | 0.51394 | 0.44231 | 0.50000 |
| Compositional | Oracle | 1.00000 | 1.00000 | 0.59615 |

Key paired results, using 4,000 episode-cluster bootstrap draws and exact
two-sided McNemar tests:

| Comparison | QA delta [95% CI] | McNemar discordance; p | Pooled-AUC delta [95% CI] |
|---|---:|---:|---:|
| Decoupled RL-QA − Original | **+0.08824** `[+0.02941,+0.16176]` | 0 vs 6; **0.03125** | **+0.08716** `[+0.06461,+0.11218]` |
| Decoupled RL-QA − SFT-W | +0.01471 `[-0.02941,+0.05919]` | 1 vs 2; 1.0 | **+0.04591** `[+0.02668,+0.06708]` |
| Compositional RL-QA − Original | +0.05769 `[-0.01923,+0.15385]` | 1 vs 4; 0.375 | **+0.11051** `[+0.08182,+0.14162]` |
| Compositional RL-QA − SFT-W | 0.00000 `[-0.05769,+0.05769]` | 1 vs 1; 1.0 | **+0.05926** `[+0.03487,+0.08501]` |

The Decoupled containment change was `+7.35pp` versus Original and `-1.47pp`
versus SFT-W. On Compositional it was `+19.23pp` versus Original and `+7.69pp`
versus SFT-W. This is not a selection-only outcome because the predeclared
primary Decoupled QA effect also clears the Gate.

## Exploitable-subset diagnosis

An episode is exploitable when oracle-selected memory produces a correct
answer. Decoupled had 25/68 exploitable episodes: RL-QA QA was 64% versus 52%
for Original and 64% for SFT-W; containment was 68%, 52%, and 68%, respectively.
Compositional had 31/52: RL-QA QA was 70.97% versus 61.29% for Original and
70.97% for SFT-W; containment was 48.39%, 32.26%, and 41.94%.

The oracle ceilings (36.76% on all Decoupled episodes and 59.62% on all
Compositional episodes under isolated decoding) show substantial
recall/composition noise, especially on Decoupled. Even so, RL-QA improves the
admission ranking and the primary downstream metric; admission remains a
learnable bottleneck rather than the only bottleneck.

## No-harm checks

- Full-context QA: RL-QA minus Original was exactly 0 on Decoupled
  (`0.67647` each), Compositional (`0.69231` each), and their 120-episode
  aggregate (`0.68333` each).
- Fresh Decoupled W_rr AUC: base `0.64086`, RL-QA `0.64174`; delta
  **+0.00088**, paired episode-cluster 95% CI
  `[-0.00169,+0.00377]` with 4,000 draws.

Neither the `-0.03` workspace threshold nor the `-2pp` full-context threshold
fired.

## Greedy-decoding batch sensitivity

The original unified evaluation used QA batch size 8. Auditing identical
selected sets revealed that some greedy answers changed when unrelated prompts
shared a bf16 generation batch. The original result is preserved as a
throughput/batched-decoding sensitivity result, not overwritten.

On Decoupled, batch 8 gave Original/SFT-W/RL-QA QA of 22/25/26 correct, so
RL-QA minus Original was `+5.88pp` (exact McNemar `p=0.125`). Isolated decoding
gave 23/28/29, so the effect was `+8.82pp` (`p=0.03125`). Compositional point QA
values were unchanged. Both Decoupled evaluations clear `+5pp`; only the
significance claim is batch-sensitive. The report therefore does not select a
favorable batch result: isolated decoding is designated primary for
reproducibility, and both results are disclosed.

The evaluator now records admission, QA, and no-harm batch sizes in its config,
with a regression test covering that provenance.

## Interpretation and next manual decision

B1 answers the clean question positively: downstream consequences alone can
train memory admission better than Original on the locked seed-0 test. It also
produces substantially better OOD admission AUC than SFT-W, but its downstream
QA is only nominally `+1.47pp` above SFT-W and does not establish superiority.
That is exactly the setting in which the Hybrid hypothesis remains useful:
workspace shaping may improve credit assignment while QA remains the ultimate
utility target.

If manually approved, the recommended next run is Stage C at seed 0 and `k=2`,
starting with `lambda_QA=1, lambda_W=0.5` and otherwise matching B1. The
predeclared ID-only coefficient order is `0.5`, `0.25`, `1.0`; no OOD result
should be viewed until one coefficient is locked. No extra RL-QA seeds, Hybrid
runs, RL-W tuning, Qwen3, or OLMo jobs have been launched.

## Artifacts and verification

- Formal run: `train/formal_rl-qa_Qwen2.5-7B-Instruct_rank_continuous_split0_s0_beta0p03_k2_lq1_lw0/`
- Strict run validation: `reports/stage_b1_rl_qa_validation.json`
- Corrected Decoupled evaluation: `eval/stage_b1_seed0_locked_retry1/decoupled_qa_batch1_sensitivity.json`
- Corrected Compositional evaluation: `eval/stage_b1_seed0_locked_retry1/compositional_qa_batch1_sensitivity.json`
- Preserved batch-8/no-harm evaluation: `eval/stage_b1_seed0_locked/ood_decoupled_compositional_qa_noharm.json`
- Gate summary: `reports/stage_b1_gate_summary_retry1.json`
- Workspace no-harm: `reports/stage_b1_workspace_noharm.json`
- Machine-readable final summary: `reports/stage_b1_final_summary.json`
- Full test suite: `python -m pytest -q` -> **80 passed**

The batch-1 Decoupled and Compositional artifacts have SHA-256 values
`26e671b22e8c790e8bf66cc99bbb92314155101d0cce88c1afc09f98c436f7aa`
and `e5bcb62a5fae681c1277737db4ccf498075e1c076843b0498b92c6c52dd70bda`,
respectively.
