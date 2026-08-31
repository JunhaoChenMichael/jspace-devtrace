# Stage B1 report: RL-QA three-seed expansion, budget 2

## Decision

**The seed expansion passes for the preregistered claim against Original.**
RL-QA improves both downstream QA and admission AUC over Original in every
training seed on both locked OOD batteries. The Decoupled three-seed mean is
`41.18% +/- 1.47pp` QA and `0.55321 +/- 0.02919` AUC, versus Original at
`33.82%` and `0.48590`. The mean paired effects are `+7.35pp` QA and
`+0.06731` AUC. Compositional QA is `44.23%` in all three seeds versus
Original at `38.46%`; all three admission-AUC effects are also positive.

The stronger claim against supervised workspace distillation is **not**
supported downstream. Against the fixed SFT-W seed-0 baseline, Decoupled QA
deltas are `+1.47pp`, `-1.47pp`, and `0`, while Compositional QA is exactly tied
in all three seeds. Admission AUC is higher on average than SFT-W, but seed 1
reverses direction on both batteries. The supported paper claim is therefore:
direct future-utility RL reproducibly improves admission and QA over Original;
it does not reproducibly improve downstream QA over SFT-W.

Campaign state remains:

`RL-W failed -> RL-QA replicated against Original -> Hybrid C3 failed`

No Hybrid coefficient, extra budget, alternate model, or new hyperparameter
was run during this expansion.

## Locked training protocol and audit

All runs use Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`, split seed 0, budget 2, group
size 8, temperature 5.0, 300 steps, rank-32 LoRA, bf16, LR `1e-6`, KL beta
`0.03`, two GRPO epochs, `lambda_QA=1`, and `lambda_W=0`. The probe is hidden
from admission and recall always uses the adapter-disabled base checkpoint.
All three runs share the byte-identical 175/45 split manifest. Checkpoints were
selected only by strict first maximum of ID QA before OOD evaluation.

| Seed | ID-selected step | ID QA | Mixed-reward groups | Median unique sets | Identical-set groups | Nonzero-gradient groups |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 32/45 = 0.71111 | 54.33% | 3 | 4.00% | 100.00% |
| 1 | 100 | 31/45 = 0.68889 | 49.33% | 3 | 4.00% | 100.00% |
| 2 | 300 | 32/45 = 0.71111 | 48.67% | 3 | 3.67% | 99.67% |

Seeds 1 and 2 receive the predeclared AMBER diagnostic because reward std was
below 0.1 in slightly more than half of groups. This did not become a software
or diversity failure: both completed 300 steps, median set diversity remained
3, identical-set groups stayed below 4%, gradients were almost always nonzero,
and the strict audit passed. Per protocol, neither run was extended or tuned.

## Raw OOD results

Each percentage is an episode-level accuracy. Mean and standard deviation use
the three raw seed values and sample standard deviation (`ddof=1`). Original
and SFT-W are fixed shared baselines, not three independently trained seeds.

### Decoupled: 68 episodes / 335 candidates

| Method | Admission AUC | QA | Top-2 containment | Full-context QA |
|---|---:|---:|---:|---:|
| Original | 0.48590 | 33.82% | 45.59% | 66.18% |
| SFT-W seed 0 | 0.52715 | 41.18% | 54.41% | 64.71% |
| RL-QA seed 0 | 0.57306 | 42.65% | 52.94% | 67.65% |
| RL-QA seed 1 | 0.51969 | 39.71% | 50.00% | 66.18% |
| RL-QA seed 2 | 0.56686 | 41.18% | 52.94% | 66.18% |
| **RL-QA mean +/- sample std** | **0.55321 +/- 0.02919** | **41.18 +/- 1.47pp** | **51.96 +/- 1.70pp** | **66.67 +/- 0.85pp** |

### Compositional: 52 episodes / 261 candidates

| Method | Admission AUC | QA | Top-2 containment | Full-context QA |
|---|---:|---:|---:|---:|
| Original | 0.34431 | 38.46% | 25.00% | 69.23% |
| SFT-W seed 0 | 0.39556 | 44.23% | 36.54% | 65.38% |
| RL-QA seed 0 | 0.45482 | 44.23% | 44.23% | 67.31% |
| RL-QA seed 1 | 0.38278 | 44.23% | 32.69% | 69.23% |
| RL-QA seed 2 | 0.44456 | 44.23% | 44.23% | 71.15% |
| **RL-QA mean +/- sample std** | **0.42739 +/- 0.03897** | **44.23 +/- 0.00pp** | **40.38 +/- 6.66pp** | **69.23 +/- 1.92pp** |

## Per-seed paired effects

The intervals below use 4,000 whole-episode paired bootstrap draws. McNemar is
exact, two-sided, and computed separately for every seed; no seed-by-episode
pseudoreplication or pooled McNemar test is used. `RL-only/base-only` gives the
discordant counts for QA.

### Versus Original

| Battery | Seed | QA delta [95% CI] | RL-only/base-only; exact p | AUC delta [95% CI] |
|---|---:|---:|---:|---:|
| Decoupled | 0 | +0.08824 `[+0.02941,+0.16176]` | 6/0; **0.03125** | +0.08716 `[+0.06461,+0.11218]` |
| Decoupled | 1 | +0.05882 `[+0.01471,+0.11765]` | 4/0; 0.125 | +0.03379 `[+0.02429,+0.04445]` |
| Decoupled | 2 | +0.07353 `[+0.01471,+0.14706]` | 5/0; 0.0625 | +0.08096 `[+0.06089,+0.10398]` |
| Compositional | 0 | +0.05769 `[-0.01923,+0.15385]` | 4/1; 0.375 | +0.11051 `[+0.08182,+0.14162]` |
| Compositional | 1 | +0.05769 `[0,+0.13462]` | 3/0; 0.25 | +0.03846 `[+0.02590,+0.05166]` |
| Compositional | 2 | +0.05769 `[-0.01923,+0.15385]` | 4/1; 0.375 | +0.10025 `[+0.07395,+0.12904]` |

The shared-draw mean paired effects are:

| Battery | Mean QA delta +/- seed std [episode-bootstrap 95% CI] | Mean AUC delta +/- seed std [episode-bootstrap 95% CI] |
|---|---:|---:|
| Decoupled | **+0.07353 +/- 0.01471** `[+0.02451,+0.13725]` | **+0.06731 +/- 0.02919** `[+0.05039,+0.08630]` |
| Compositional | **+0.05769 +/- 0.00000** `[-0.00641,+0.13462]` | **+0.08307 +/- 0.03897** `[+0.06116,+0.10648]` |

### Versus fixed SFT-W seed 0

| Battery | Seed | QA delta [95% CI] | RL-only/SFT-only; exact p | AUC delta [95% CI] |
|---|---:|---:|---:|---:|
| Decoupled | 0 | +0.01471 `[-0.02941,+0.05919]` | 2/1; 1.0 | +0.04591 `[+0.02668,+0.06708]` |
| Decoupled | 1 | -0.01471 `[-0.07353,+0.04412]` | 2/3; 1.0 | -0.00746 `[-0.01918,+0.00366]` |
| Decoupled | 2 | 0.00000 `[-0.05882,+0.05882]` | 2/2; 1.0 | +0.03971 `[+0.02382,+0.05735]` |
| Compositional | 0 | 0.00000 `[-0.05769,+0.05769]` | 1/1; 1.0 | +0.05926 `[+0.03487,+0.08501]` |
| Compositional | 1 | 0.00000 `[0,0]` | 0/0; 1.0 | -0.01279 `[-0.02981,+0.00302]` |
| Compositional | 2 | 0.00000 `[-0.05769,+0.05769]` | 1/1; 1.0 | +0.04900 `[+0.02642,+0.07304]` |

Mean Decoupled QA difference from SFT-W is exactly `0.00000 +/- 0.01471`
with shared-draw episode CI `[-0.05392,+0.05882]`. Mean Compositional QA
difference is exactly zero with zero seed dispersion. Mean AUC differences are
`+0.02605 +/- 0.02919` on Decoupled and `+0.03182 +/- 0.03897` on
Compositional, but the seed-1 reversal prevents an all-seed superiority claim.

## Containment and exploitable subset

Containment improves over Original in every seed. Decoupled seed deltas are
`+7.35pp`, `+4.41pp`, and `+7.35pp` (exact McNemar p values `0.0625`, `0.25`,
and `0.0625`). Compositional deltas are `+19.23pp`, `+7.69pp`, and `+19.23pp`
(p values `0.001953`, `0.125`, and `0.001953`). Against SFT-W, containment is
not stable: the mean differences are `-2.45pp +/- 1.70pp` on Decoupled and
`+3.85pp +/- 6.66pp` on Compositional.

The locked Decoupled exploitable subset contains 25/68 episodes. RL-QA QA is
64% in every seed, versus 52% for Original and 64% for SFT-W. Its admission AUC
is `0.64896`, `0.59229`, and `0.64792` (mean `0.62972 +/- 0.03242`), and every
seed remains above Original. This localizes the replicated QA gain to episodes
on which the frozen recall/composition system can exploit selected memory.

## No-harm

No predeclared no-harm threshold fires.

- Decoupled full-context deltas versus Original are `+1.47pp`, `0`, and `0`.
- Compositional full-context deltas versus Original are `-1.92pp`, `0`, and
  `+1.92pp`; the largest decline remains inside the strict `-2pp` threshold.
- Fresh Decoupled end-only W_rr AUC is `0.64174`, `0.64150`, and `0.64144`
  versus frozen base `0.64086`. Paired seed-minus-base deltas are `+0.00088`
  `[-0.00169,+0.00377]`, `+0.00063` `[-0.00171,+0.00287]`, and `+0.00058`
  `[-0.00166,+0.00302]`. Their mean is `+0.00070 +/- 0.00016`, far above the
  `-0.03` alert boundary.

## Statistical scope

The shared-draw mean CIs cluster over the fixed OOD episodes. They do not
pretend that three seeds provide a well-estimated random-effects population
interval. Cross-seed reproducibility is assessed separately through all-seed
direction and sample standard deviation. This distinction is why the report
accepts the Original comparison while qualifying the SFT-W comparison despite
positive mean AUC effects.

## Sealed-batch reproducibility audit

The five shared conditions (`Original`, fixed `SFT-W`, fixed seed-0 `RL-QA`,
`Workspace`, and `Oracle`) were compared against the previously sealed
source-specific batch-1 artifacts. Decoupled matched across 1,675 admission
scores and 340 selection/QA records; Compositional matched across 1,305 scores
and 260 selection/QA records. Compositional also matched all 156 comparable
adapter-enabled full-context records. The older Decoupled artifact had
`skip_no_harm=true`, so its condition-specific full-context comparison is
explicitly `not-applicable`; the common frozen-base full-context references
still match. The corrected audit passes with zero issues.

The first audit artifact is preserved as a failed software-audit attempt. It
incorrectly assumed that QA prompt concepts retain score-rank order, while the
frozen recall path canonically sorts the chosen set by candidate index. The
audit rule was minimally corrected, regression-tested, and rerun as `_retry1`;
no model output or scientific configuration changed.

## Artifacts

- Training audit: `reports/stage_b1_rl_qa_seed_expansion_training_audit.json`
- Pre-OOD lock: `reports/stage_b1_rl_qa_seed_expansion_pre_ood_lock.json`
- Decoupled raw: `eval/stage_b1_seed_expansion_k2/ood_decoupled_batch1.json`
- Compositional raw: `eval/stage_b1_seed_expansion_k2/ood_compositional_batch1.json`
- Three-seed OOD analysis: `reports/stage_b1_rl_qa_seed_expansion_ood_analysis.json`
- Independent OOD validation: `reports/stage_b1_rl_qa_seed_expansion_ood_analysis_validation.json`
- Workspace no-harm analysis: `reports/stage_b1_rl_qa_seed_expansion_workspace_noharm.json`
- Independent workspace validation: `reports/stage_b1_rl_qa_seed_expansion_workspace_noharm_validation.json`
- Shared-condition reproducibility audit: `reports/stage_b1_rl_qa_seed_expansion_shared_condition_audit_retry1.json`
- Preserved failed audit attempt: `reports/stage_b1_rl_qa_seed_expansion_shared_condition_audit.json`

All required seed/metric cells are present; none is `insufficient-data`.
