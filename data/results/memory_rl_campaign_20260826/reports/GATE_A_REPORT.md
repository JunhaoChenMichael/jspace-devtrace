# Gate A report — seed 0, budget 2

## Decision

**Gate A status: `worse`.** On the locked Decoupled evaluation,
`RL-W - SFT-W = -0.03385`, which is below the predeclared `-0.03` boundary.
The paired 4,000-draw episode-cluster bootstrap interval for `RL-W - SFT-W`
is `[-0.04871, -0.02048]`.

Per protocol, RL-W is not expanded to seeds 1 and 2, and OOD results are not
used to retune beta. Stage B and Stage C remain `insufficient-data` and were not
started.

## Locked configurations

- Model and teacher: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Shared split: seed 0, 175 train / 45 ID validation episodes, manifest SHA-256
  `1988c8af1fc39884a7bd711f335f91db6c890612e6e0a45917488ba4b44abf0f`.
- Both runs: LoRA rank 32, LR `1e-6`, bf16, 300 steps, rank-continuous teacher,
  budget 2. RL-W used beta `0.03`, group size 8, centered advantages, separate
  KL, and disabled dropout.
- Temperature was locked to 5.0 before OOD after a fixed-ID calibration showed
  the requested 0.7 setting produced insufficient mixed groups. It is recorded
  for SFT-W as a matched operational setting but is unused by SFT training.
- SFT-W checkpoint: ID-best step 300, AUC `0.65746` (baseline `0.62658`).
- RL-W checkpoint: ID-best step 200, AUC `0.63561` (baseline `0.62658`).

The RL-W ID diagnostic was AMBER but did not trigger a controlled beta/LR
branch: 58.0% of groups had mixed actions, 52.3% had nonzero reward variance,
42.0% were all-Yes/all-No, gradients were finite and nonzero, and mean KL was
only `0.000586`. Validation rose rather than fell.

## OOD admission results

All confidence intervals below use 4,000 episode-cluster bootstrap draws.

| Battery | Original AUC | SFT-W AUC | RL-W AUC | Workspace AUC | RL-W − SFT-W (95% CI) |
|---|---:|---:|---:|---:|---:|
| Decoupled, 68 episodes | 0.48590 | 0.52715 | 0.49331 | 0.64119 | -0.03385 [-0.04871, -0.02048] |
| Compositional, 52 episodes | 0.34431 | 0.39492 | 0.35554 | 0.51394 | -0.03938 [-0.06162, -0.02021] |

Budget-2 containment was `0.45588 / 0.54412 / 0.45588 / 0.52941` on
Decoupled and `0.25000 / 0.36538 / 0.26923 / 0.44231` on Compositional for
Original / SFT-W / RL-W / Workspace respectively.

## Decoupled downstream and no-harm checks

With one adapter-disabled frozen recall model and a 64-token answer budget,
budget-2 selection QA was:

| Condition | Accuracy |
|---|---:|
| Original | 0.35294 |
| SFT-W | 0.41176 |
| RL-W | 0.38235 |
| Workspace | 0.36765 |
| Oracle | 0.36765 |

The exact SFT-W versus RL-W McNemar comparison had 5 SFT-only-correct and 3
RL-only-correct episodes (`p=0.72656`, `n=68`).

Adapter-enabled full-context accuracy was `0.67647` for Original, `0.67647` for
SFT-W (delta `0.00000`), and `0.66176` for RL-W (delta `-0.01471`). Neither
adapter crossed the predeclared drop-beyond-2pp warning.

Fresh, matched `W_rr` measurements gave:

| Condition | W_rr AUC | Adapter − base (paired 95% CI) |
|---|---:|---:|
| Base | 0.64086 | — |
| SFT-W | 0.64191 | +0.00105 [-0.00140, +0.00364] |
| RL-W | 0.64095 | +0.00008 [-0.00240, +0.00267] |

These use 4,000 episode-cluster draws. Measurement used `--end-only`, which
preserves `W_rr` and `W_end`; `W_max` is intentionally not interpreted. Neither
adapter crossed the workspace-AUC drop-beyond-0.03 warning.

## Acceptance and provenance

- Full unit suite after the minimal software fixes: `62 passed`.
- Formal SFT-W and RL-W artifact validators: `pass`, with identical split
  manifests and `teacher_mismatch_override=false`.
- OOD outputs passed strict JSON, episode/count, budget, condition, bootstrap,
  QA, McNemar, and no-harm completeness assertions.
- Single-seed sample standard deviations are `insufficient-data` by design;
  seed expansion is prohibited by the Gate A result rather than silently
  omitted.
- The deferred `datasets` dependency and MMLU/ARC/GSM8K evaluations were not
  installed or run because the campaign stopped at Gate A.

## Required next review

The protocol permits downstream utility RL only after manual review of this
`worse` result. No RL-QA, Hybrid, extra seed, Stage D, scale sweep, online
workspace reward, or direct memory generation run has been launched.
