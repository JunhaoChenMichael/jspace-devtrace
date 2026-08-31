# Stage C OOD report: Hybrid-0.25, seed 0, budget 2

## 1. Gate C decision

**C3: NO ADDED VALUE.** On the primary Decoupled battery, Hybrid-0.25 and
RL-QA have identical downstream QA (`29/68`) and top-2 containment (`36/68`),
with zero paired discordance for both endpoints. Hybrid has a small, resolved
pooled-AUC shift of `+0.00606` with 95% CI `[+0.00011, +0.01353]`, but this
ranking-only change does not alter either primary endpoint. On the fixed
25-episode exploitable subset, Hybrid and RL-QA are also exactly identical in
QA (`16/25`) and containment (`17/25`).

This is not C1 because Decoupled QA does not strictly improve. It is not C2
because the preregistered exploitable-subset check provides no admission or QA
confirmation. It is not C4 because no meaningful regression is observed.
Compositional adds one correct Hybrid answer, but containment is unchanged and
AUC is unresolved; as cross-regime support, it cannot override the primary
Decoupled outcome. RL-QA remains the leading RL method pending manual review.

## 2. Locked configuration and hashes

- Method: `rl-hybrid`, seed 0, split seed 0, `k=2`, `lambda_QA=1`,
  `lambda_W=0.25`, `best-step-200`.
- Model: `Qwen/Qwen2.5-7B-Instruct`, immutable revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Adapter weights SHA-256:
  `b7ba79b685728c2249bf4d73e732f355bf5a458b4a80f4ee3d6b36d2d7341343`.
- Comparator adapter SHA-256 values: SFT-W
  `d74e71ffce9c5a585d1a3f2ab17832230bb8f15b463aad6148aa39c0f8855af9`,
  RL-W `e7a53252f3a71a44b3b1c1be464640c8768440926c10be1ff52172e24aeeda18`,
  and RL-QA
  `55a4e583c241c9c6aca394c0a58146dfa91de4f0297275a8b39170b5c2cfd25d`.
- Run-config SHA-256:
  `22eb6e198325b4f63ce64f6ed6a5b404e2b0d738010ede74c54815b10403a412`.
- Split manifest: fixed 175/45 split; content SHA-256
  `1988c8af1fc39884a7bd711f335f91db6c890612e6e0a45917488ba4b44abf0f`.
- Teacher matched the policy reference and
  `teacher_mismatch_override=false`.
- The approved smallest-lambda amendment was recorded before a completed OOD
  artifact existed. Reporter correlations were not used for selection.

## 3. Decoupled full comparison

| Method | Pooled admission AUC | Top-2 containment | QA accuracy |
|---|---:|---:|---:|
| Original | 0.48590 | 0.45588 | 0.33824 |
| SFT-W | 0.52715 | **0.54412** | 0.41176 |
| RL-W | 0.49331 | 0.45588 | 0.36765 |
| RL-QA | 0.57306 | 0.52941 | **0.42647** |
| **Hybrid-0.25** | **0.57912** | 0.52941 | **0.42647** |
| Workspace | 0.64119 | 0.52941 | 0.35294 |
| Oracle | 1.00000 | 1.00000 | 0.36765 |

## 4. Compositional full comparison

| Method | Pooled admission AUC | Top-2 containment | QA accuracy |
|---|---:|---:|---:|
| Original | 0.34431 | 0.25000 | 0.38462 |
| SFT-W | 0.39556 | 0.36538 | 0.44231 |
| RL-W | 0.35508 | 0.25000 | 0.38462 |
| RL-QA | **0.45482** | **0.44231** | 0.44231 |
| **Hybrid-0.25** | 0.45404 | **0.44231** | **0.46154** |
| Workspace | 0.51394 | **0.44231** | 0.50000 |
| Oracle | 1.00000 | 1.00000 | 0.59615 |

## 5. Hybrid versus RL-QA paired statistics

All deltas are Hybrid minus comparator. QA and containment intervals use the
prelocked 4,000 shared episode-cluster draws. The QA test is exact two-sided
paired McNemar; discordance is shown as `Hybrid-only / comparator-only`.

| Battery | QA delta [95% CI] | QA discordance; p | Containment delta [95% CI] | Pooled-AUC delta [95% CI] |
|---|---:|---:|---:|---:|
| Decoupled | 0.00000 `[0,0]` | 0 / 0; 1.0 | 0.00000 `[0,0]` | +0.00606 `[+0.00011,+0.01353]` |
| Compositional | +0.01923 `[0,+0.05769]` | 1 / 0; 1.0 | 0.00000 `[0,0]` | -0.00078 `[-0.00812,+0.00534]` |

## 6. Hybrid versus SFT-W paired statistics

| Battery | QA delta [95% CI] | QA discordance; p | Containment delta [95% CI] | Pooled-AUC delta [95% CI] |
|---|---:|---:|---:|---:|
| Decoupled | +0.01471 `[-0.02941,+0.05919]` | 2 / 1; 1.0 | -0.01471 `[-0.04412,0]` | +0.05197 `[+0.03433,+0.07187]` |
| Compositional | +0.01923 `[0,+0.05769]` | 1 / 0; 1.0 | +0.07692 `[+0.01923,+0.15385]` | +0.05847 `[+0.03493,+0.08433]` |

## 7. Hybrid versus Original paired statistics

| Battery | QA delta [95% CI] | QA discordance; p | Containment delta [95% CI] | Pooled-AUC delta [95% CI] |
|---|---:|---:|---:|---:|
| Decoupled | +0.08824 `[+0.02941,+0.16176]` | 6 / 0; **0.03125** | +0.07353 `[+0.01471,+0.13235]` | +0.09322 `[+0.07132,+0.11826]` |
| Compositional | +0.07692 `[+0.01923,+0.15385]` | 4 / 0; 0.125 | +0.19231 `[+0.09615,+0.30769]` | +0.10973 `[+0.08130,+0.14012]` |

These Original comparisons reproduce the downstream benefit already learned
by RL-QA; they do not show an incremental Hybrid benefit over RL-QA.

## 8. Decoupled exploitable-subset analysis

The subset was fixed before result inspection as episodes where
`refs['oracle@2'].correct == true`; the validator reproduced the expected
`25/68` episodes.

Here Oracle is the label-selected memory reference used to define the fixed
exploitable proxy, not a strict QA ceiling: on all Decoupled episodes its QA is
`25/68`, below Hybrid's `29/68`, because selected distractors and recall
composition can affect generation.

| Method | QA accuracy | Top-2 containment |
|---|---:|---:|
| Original | 0.52 | 0.52 |
| SFT-W | 0.64 | 0.68 |
| RL-W | 0.52 | 0.52 |
| RL-QA | **0.64** | **0.68** |
| **Hybrid-0.25** | **0.64** | **0.68** |
| Workspace | 0.52 | 0.56 |
| Oracle | 1.00 | 1.00 |

Hybrid minus RL-QA is exactly zero for QA and containment: both paired CIs are
`[0,0]`, both have zero discordant episodes, and both McNemar p-values are
`1.0`. Hybrid also ties SFT-W exactly on these two subset metrics. Relative to
Original, Hybrid is `+0.12` in QA (CI `[0,+0.24]`, 3/0 discordance,
`p=0.25`) and `+0.16` in containment (CI `[+0.04,+0.32]`).

## 9. No-harm checks

Fresh adapter-enabled full-context QA used isolated batch size 1:

| Battery | Original | Hybrid-0.25 | Hybrid minus Original | McNemar p |
|---|---:|---:|---:|---:|
| Decoupled | 0.66176 | 0.66176 | 0.00000 | 1.0 |
| Compositional | 0.69231 | 0.69231 | 0.00000 | 1.0 |

The `-2pp` full-context warning does not fire. A fresh post-adaptation latent
workspace W_rr measurement was not part of the authorized one-shot artifact,
so that separate no-harm delta is `insufficient-data`; it is not inferred from
another adapter or silently omitted.

This is an accuracy no-harm result, not a claim of text-level invariance. The
Hybrid and Original correctness vectors are identical, while their raw answer
strings differ on 9/68 Decoupled and 14/52 Compositional episodes.

## 10. Batch-size and decoding provenance

- Decoupled and Compositional were run as separate source-isolated evaluator
  calls so each preserved Stage-B1 admission batch boundaries and bootstrap
  seed 0.
- Admission scoring: bf16, batch 16, max length 2048.
- Selection QA: episode-isolated greedy decoding, batch 1, 64-token answer
  budget.
- Full-context no-harm QA: episode-isolated greedy decoding, batch 1.
- Selection recall used the same adapter-disabled frozen base model for every
  condition. Admission received only context and candidate concept; the future
  probe and answer were hidden.
- Original verbal scores came from the locked precomputed `v_ref`; all four
  adapters were freshly scored in the same source-local run.
- For the six shared conditions, candidate scores, episode order, selected-set
  QA, references, and by-spec/aggregate metrics exactly match the corrected
  Stage-B1 batch-1 artifacts (`0` mismatches on both batteries).

## 11. Artifacts and validator status

- Pre-OOD lock: `reports/stage_c_pre_ood_tiebreak_lock.json`, SHA-256
  `8950a15f49b5cea29ad6bb85fc99542be9695ba01960a03e067e80111db4b317`.
- Preflight/statistical lock: `reports/stage_c_ood_preflight.json`, SHA-256
  `d49b1d06dd0650228de70fdddef8aa41ddfce5a7aefac712e96789c7a326d7b7`.
- Decoupled raw artifact: `eval/stage_c_seed0_locked/ood_decoupled_batch1.json`,
  SHA-256 `0c90a9a37eebd760fd500cc1c724cbcc7279b7602f1a0b47102a41047d00ddb0`.
- Compositional raw artifact:
  `eval/stage_c_seed0_locked/ood_compositional_batch1.json`, SHA-256
  `5e67c6a601889b34caf93909c662bea6aa4642a0b8185e985bc842cad1ecc2d1`.
- Structured analysis: `reports/stage_c_ood_analysis.json`, SHA-256
  `2e061d1e6783874239137e1a929d59c16314fdc840e08a3a89b46f1ac71f6b4b`.
- Independent raw recomputation validator:
  `reports/stage_c_ood_analysis_validation.json`, status **PASS**, zero errors;
  SHA-256 `0274ee6602d9c7653e105d6b732094ba6478e0192ab487f01926a32c54af9bde`.
- Full test suite: `108 passed`; focused Stage-C tests: `16 passed`; Ruff:
  all checks passed.
- Analysis implementation SHA-256:
  `f3e7338366e9a0b3b023473f1ffb62f0153adce0bc8a0fdc15f63c47ea92e470`.

This is a single-seed Gate decision. It does not estimate cross-seed mean or
sample standard deviation and cannot support a multi-seed material-regression
claim. Runtime model/input/hash provenance is supplied by the companion lock
and preflight because the evaluator JSON schema does not embed every hash.

## 12. Stop condition

No Hybrid seed 1/2, RL-QA seed 1/2, Hybrid-0.5/1.0 OOD, coefficient change,
Qwen3, OLMo, direct-memory generation, online-workspace reward, or other
unauthorized follow-up was launched. Execution stops here for manual review.

**C3: NO ADDED VALUE**
