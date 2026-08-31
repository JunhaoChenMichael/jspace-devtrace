# Paper results master README

This is the canonical results index for writing the paper. It consolidates the
Alignment and Memory-RL campaigns run on NVIDIA RTX A5000 and
NVIDIA A100-SXM4-80GB, and records which results remain valid after the verbal
score correction. Last reconciled: 2026-08-31 at repository commit
`a27c260548f18bff75a529c93e68c8e9ee35610a`.

## Citation rule and measurement correction

Before schema v3, the verbal report was computed as
`P(yes)/(P(yes)+P(no)+1e-9)`. Yes/No mass was often near `1e-13`, so the guard
dominated and ranked absolute Yes probability rather than Yes-versus-No
preference. Schema v3 uses the exact log-space ratio:

```text
V = sigmoid(logsumexp(logits[yes]) - logsumexp(logits[no]))
```

`W_rr`, workspace teacher labels, adapters, RL policy scores, ID checkpoint
choices, and full-context no-harm checks are unaffected. Old `V`, `V_raw`,
Original verbal rankings, and budget-k sets selected by those rankings require
v3 re-evaluation. Only reports marked **CURRENT** below are paper-citable.
**HISTORICAL**, **PENDING V3**, and **SUPERSEDED** reports are provenance only.

Authoritative correction records:

- [`CORRECTIONS.md`](../CORRECTIONS.md)
- [`MEASUREMENT_BUG_CORRECTION_FINAL_SUMMARY.md`](../data/results/a100_next_boundary_campaign/MEASUREMENT_BUG_CORRECTION_FINAL_SUMMARY.md)
- [`V_EPSILON_DEFECT_IMPACT_REPORT.md`](../data/results/a100_next_boundary_campaign/shared/V_EPSILON_DEFECT_IMPACT_REPORT.md)

## Paper-level conclusions

| Claim | Best evidence | Paper status |
|---|---|---|
| Capable models contain load-bearing information in the workspace that their verbal report under-expresses. | Corrected Decoupled grid: Qwen2.5-7B-I gap `+0.1526`, Qwen3-8B `+0.2525`, OLMo-2-7B-I `+0.3135`, Mistral-7B-I `+0.2014`; every paired CI excludes zero. | **SUPPORTED** across four families. |
| Binary Metacognitive Alignment repairs Qwen3-8B reporting without moving the workspace. | Three fresh A100 seeds: corrected mean Decoupled `ΔV=+0.2128`; all pass; `|ΔW|<0.001`; no QA drop above 2 pp. | **SUPPORTED; primary Alignment result.** |
| The 8B repair replicates on Compositional OOD. | Corrected mean `ΔV=+0.0861`; only seed 1 passes `+0.15`. | **NOT REPLICATED; limitation.** |
| Alignment improves monotonically with scale. | 8B `+0.2128` (3 seeds), 14B `+0.3826` (1), 32B `+0.0954` (1). | **NOT SUPPORTED.** The observed 14B peak needs replication. |
| Probe-blind RL-QA improves Qwen3-8B admission and budget-2 recall over Original. | Corrected A100 gains: QA `+7.35/+5.88/+8.82 pp`; admission AUC `+0.4339/+0.4216/+0.4599`. | **SUPPORTED; primary RL result.** |
| Qwen3-32B RL-QA is a clean pass. | Seed 0 QA `+4.41 pp` against `+5 pp`; admission `+0.3653`, CI `[+0.3056,+0.4423]`. | **UNRESOLVED:** admission-positive, QA below gate. |
| RL-QA beats matched Workspace-SFT downstream. | A5000 Qwen2.5-7B did not reproduce a QA advantage; Qwen3-8B has no matched SFT-W run. | **NOT SUPPORTED.** |
| Adding workspace reward to RL-QA helps. | A5000 Hybrid-0.25 tied RL-QA on primary Decoupled QA and containment. | **NO ADDED VALUE** in the historical campaign. |

## Hardware, tasks, and models

### NVIDIA RTX A5000

| Task | Scientific model | Seeds | Campaign and interpretation |
|---|---|---:|---|
| Binary Metacognitive Alignment | `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218` | 0 | M0, frozen `W_ref`, rank-16 LoRA, ID lock, one-shot OOD. Training provenance is valid; reported verbal effects are schema v2 and **PENDING V3**. |
| Memory-RL | `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28` | RL-W/SFT-W 0; RL-QA 0/1/2; Hybrid 0 | Rank-32 LoRA, budget 2. Policy training and method decisions are informative; old Original-arm comparisons are **PENDING V3**. |
| Engineering smoke | `Qwen/Qwen2.5-3B-Instruct` | 0 | Wiring only, not scientific evidence. |

The A5000 Memory-RL preflight recorded 24,564 MiB total and 23,851 MiB free.
The Qwen3-8B Alignment preflight recorded 24,098 MiB free; canary peak
allocation was about 17.44 GB. Both used bf16.

Historical A5000 measurement work also covered GPT-2; Qwen2.5 0.5B/3B/7B base
and instruct; OLMo-2 1B/7B stages; Mistral-7B; Qwen2-VL-2B; and LLaVA-1.5-7B.
Most predates schema v3; use the fresh A100 grid for current V/W claims.

### NVIDIA A100-SXM4-80GB

| Task | Model | Seeds | Final status |
|---|---|---:|---|
| Binary Metacognitive Alignment | Qwen3-8B | 0, 1, 2 | All corrected-v3 GREEN; seed 1 Strong GREEN. |
| Binary Metacognitive Alignment | Qwen3-14B | 0 | Strong GREEN; single seed. |
| Binary Metacognitive Alignment | Qwen3-32B | 0 | AMBER; directional repair below gate. |
| RL-QA | Qwen3-8B | 0, 1, 2 | All corrected comparisons pass vs Original. |
| RL-QA | Qwen3-32B | 0 | `ADMISSION_POSITIVE_QA_UNRESOLVED`; expansion stopped. |
| Corrected predictive grid | 24 checkpoints across Qwen2.5, Qwen3, OLMo-2, Mistral, GPT-2 | measurement | Fresh v3 results on Explicit, Evoked, Decoupled, Compositional. |

These are A100 results, not H100 results.

## Alignment results

### Qwen3-8B: three A100 seeds, corrected v3

All three ID checkpoints were independently locked before any OOD access.

| Seed | V before | V after | Corrected `ΔV` | `ΔW` | Full-context QA drop | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.4020 | 0.5662 | **+0.1642** | +0.00017 | 0.00 pp | GREEN |
| 1 | 0.4020 | 0.6889 | **+0.2868** | +0.00091 | 0.00 pp | Strong GREEN |
| 2 | 0.4020 | 0.5893 | **+0.1873** | -0.00028 | 1.47 pp | GREEN |
| Mean | 0.4020 | 0.6148 | **+0.2128** | +0.00027 | 0.49 pp | Three-seed GREEN |

`ΔV` sample SD is `0.0652`. The original `+0.2732` mean is superseded.
Corrected Compositional `ΔV` is `+0.0145/+0.1873/+0.0564`, mean `+0.0861`.
Canonical source:
[`CORRECTED_8B_METACOG_THREE_SEED_REPORT.md`](../data/results/a100_next_boundary_campaign/qwen3_8b_metacog_v3/reports/CORRECTED_8B_METACOG_THREE_SEED_REPORT.md).

### Alignment across Qwen3 scale

| Model | Hardware | Seeds | Decoupled `ΔV` | Paired 95% CI | V after | `ΔW` | Comp. `ΔV` | Verdict |
|---|---|---:|---:|---|---:|---:|---:|---|
| Qwen3-8B | A100 | 3 | **+0.2128 mean** | See per-seed report | 0.566–0.689 | `<0.001` abs. | +0.0861 mean | GREEN |
| Qwen3-14B | A100 | 1 | **+0.3826** | `[+0.3029,+0.4701]` | 0.7306 | -0.00017 | **+0.3261** | Strong GREEN |
| Qwen3-32B | A100 | 1 | **+0.0954** | `[+0.0241,+0.1764]` | 0.4320 | +0.00116 | -0.0176 | AMBER |

The 14B peak and 32B decline are exploratory because both have one seed. The
32B AMBER result authorizes no post-hoc tuning.

Sources:

- [`Qwen3-14B M1 report`](../data/results/a100_next_boundary_campaign/qwen3_14b_metacog_v3/seed0/report/M1_GATE_REPORT.md)
- [`Qwen3-32B M1 report`](../data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0/report/M1_GATE_REPORT.md)
- [`paper reconciliation report`](../data/results/a100_next_boundary_campaign/CORRECTED_PAPER_RECONCILIATION_FINAL_REPORT.md)

### Qwen3-8B A5000 seed-0 pilot

The A5000 run selected step 250 on ID validation and demonstrated 24 GB
feasibility. Its historical report gives Decoupled `0.344→0.655`, `ΔV=+0.311`,
`ΔW=+0.001`, with zero Decoupled QA drop. Those V values are schema v2 and
**not paper-citable** until the exact A5000 adapter is re-evaluated under v3.
Checkpoint selection remains valid because it used the unaffected action
scorer. Historical source:
[`A5000 M1 report`](../data/results/metacog_alignment_campaign_seed0_retry1/report/M1_GATE_REPORT.md).

## Memory-RL results

### Qwen3-8B RL-QA on A100, corrected v3

Admission is probe-blind, selects exactly two items, and recall uses the frozen
adapter-disabled Qwen3-8B. Only the Original arm changed after correction.

| Seed | Original QA@2 | RL-QA QA@2 | `ΔQA` | Admission-AUC `Δ` | Paired AUC 95% CI | Full-context drop | Verdict |
|---:|---:|---:|---:|---:|---|---:|---|
| 0 | 1.47% | 8.82% | **+7.35 pp** | **+0.4339** | `[+0.3743,+0.5022]` | 1.47 pp | PASS |
| 1 | 1.47% | 7.35% | **+5.88 pp** | **+0.4216** | `[+0.3645,+0.4855]` | 0.00 pp | PASS |
| 2 | 1.47% | 10.29% | **+8.82 pp** | **+0.4599** | `[+0.4086,+0.5203]` | 1.47 pp | PASS |

QA is near the floor and gains rest on few discordant episodes; report exact
McNemar results and do not imply that absolute recall is solved. Canonical
source: correction summary, section 8.

### Qwen3-32B RL-QA on A100

| Seed | Original QA@2 | `ΔQA` | Admission-AUC `Δ` | Paired AUC 95% CI | Verdict |
|---:|---:|---:|---:|---|---|
| 0 | 7.35% | **+4.41 pp** | **+0.3653** | `[+0.3056,+0.4423]` | `ADMISSION_POSITIVE_QA_UNRESOLVED` |

The old FAIL verdict is withdrawn. Admission is resolved, but QA misses the
preregistered `+5 pp` gate. Cancelled seeds 1/2 are infrastructure stops, not
negative scientific runs.

### Qwen2.5-7B Memory-RL on A5000

The campaign's decision sequence was:

1. RL-W failed Gate A against SFT-W: Decoupled admission difference `-0.03385`,
   paired CI `[-0.04871,-0.02048]`; RL-W was not expanded.
2. RL-QA ran three seeds. It improved the historical Original arm, but did not
   reproducibly beat SFT-W on downstream QA.
3. Hybrid-0.25 tied RL-QA on primary Decoupled QA and containment; Gate C was
   `NO ADDED VALUE`.

Training and trained-policy comparisons remain useful, but the Original verbal
arm was schema v2. Original-relative admission and QA deltas are **PENDING V3**
and must not share a final table with corrected Qwen3 results.

Complete A5000 report sequence:

- [`Gate A`](../data/results/memory_rl_campaign_20260826/reports/GATE_A_REPORT.md)
- [`Stage B0`](../data/results/memory_rl_campaign_20260826/reports/STAGE_B0_REPORT.md)
- [`Stage B1 seed 0`](../data/results/memory_rl_campaign_20260826/reports/STAGE_B1_RL_QA_REPORT.md)
- [`Stage B1 three seeds`](../data/results/memory_rl_campaign_20260826/reports/STAGE_B1_RL_QA_THREE_SEED_REPORT.md)
- [`Stage C ID-only`](../data/results/memory_rl_campaign_20260826/reports/STAGE_C_ID_ONLY_REPORT.md)
- [`Stage C OOD`](../data/results/memory_rl_campaign_20260826/reports/STAGE_C_OOD_REPORT.md)
- [`Decision ledger`](../data/results/memory_rl_campaign_20260826/reports/decision_ledger.md)

## Corrected cross-family measurement

The fresh A100 v3 grid is the current source for measurement claims.

| Model | V | W_rr | Gap `W_rr−V` | Paired 95% CI | `Ms` |
|---|---:|---:|---:|---|---:|
| Qwen2.5-7B-Instruct | 0.4897 | 0.6423 | **+0.1526** | `[+0.0500,+0.2582]` | 0.762 |
| Qwen3-8B | 0.4020 | 0.6545 | **+0.2525** | `[+0.1713,+0.3396]` | 0.614 |
| OLMo-2-1124-7B-Instruct | 0.3529 | 0.6664 | **+0.3135** | `[+0.2297,+0.3992]` | 0.530 |
| Mistral-7B-Instruct-v0.3 | 0.4700 | 0.6713 | **+0.2014** | `[+0.1000,+0.3015]` | 0.700 |

Small-scale universality is false: Qwen2.5-0.5B, Qwen3-0.6B and Qwen3-4B
Decoupled intervals span zero. The full grid is
[`CORRECTED_PRIMARY_CONTRASTS_AND_SCALE_REPORT.md`](../data/results/a100_next_boundary_campaign/reports/CORRECTED_PRIMARY_CONTRASTS_AND_SCALE_REPORT.md).

Scale/interface results:

- Qwen3 Decoupled gap widens 8B→32B by `+0.1027`, CI
  `[+0.0093,+0.1944]`.
- Adjacent steps inside that range are unresolved and Evoked is non-monotonic;
  this is not a universal scaling law.
- `V_chat−V_raw` is `-0.0981/-0.0994/-0.1895` at 8B/14B/32B, each CI below
  zero. The capable-scale chat interface reads worse than the raw probe.

The chat penalty and 14B Alignment peak require author review before becoming
headline claims.

## Report registry and precedence

### CURRENT: use for paper numbers

| Report | Role |
|---|---|
| [`CORRECTED_PAPER_RECONCILIATION_FINAL_REPORT.md`](../data/results/a100_next_boundary_campaign/CORRECTED_PAPER_RECONCILIATION_FINAL_REPORT.md) | Best narrative reconciliation across families, scale, Alignment and missing inputs. |
| [`MEASUREMENT_BUG_CORRECTION_FINAL_SUMMARY.md`](../data/results/a100_next_boundary_campaign/MEASUREMENT_BUG_CORRECTION_FINAL_SUMMARY.md) | Exact correction impact on 8B/32B Alignment and RL-QA. |
| [`CORRECTED_PRIMARY_CONTRASTS_AND_SCALE_REPORT.md`](../data/results/a100_next_boundary_campaign/reports/CORRECTED_PRIMARY_CONTRASTS_AND_SCALE_REPORT.md) | Full v3 grid and paired contrasts. |
| [`CORRECTED_8B_METACOG_THREE_SEED_REPORT.md`](../data/results/a100_next_boundary_campaign/qwen3_8b_metacog_v3/reports/CORRECTED_8B_METACOG_THREE_SEED_REPORT.md) | Canonical corrected 8B Alignment table. |
| [`14B M1 report`](../data/results/a100_next_boundary_campaign/qwen3_14b_metacog_v3/seed0/report/M1_GATE_REPORT.md) | Single-seed Strong-GREEN Alignment. |
| [`32B M1 report`](../data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0/report/M1_GATE_REPORT.md) | Single-seed AMBER Alignment. |
| [`CORRECTED_SCALE_SWEEP_REPORT.md`](../data/results/a100_next_boundary_campaign/shared/CORRECTED_SCALE_SWEEP_REPORT.md) | Dense Qwen3 scale plus sparse diagnostic. |
| [`V3_IMPACT_INVENTORY.md`](../data/results/a100_next_boundary_campaign/reports/V3_IMPACT_INVENTORY.md) | Correction coverage and missing inputs. |

### HISTORICAL / PENDING V3

- A5000 Qwen3-8B M1: valid training provenance, v2 verbal effect.
- A5000 Qwen2.5-7B Memory-RL reports: valid training provenance; Original arm
  needs v3 re-evaluation.
- [`docs/RESULTS.md`](RESULTS.md) and
  [`data/results/master_table.md`](../data/results/master_table.md): historical
  pre-correction ledger; rebuild verbal tables before use.

### SUPERSEDED / WITHDRAWN

`data/results/a100_handoff/` preserves original reports with correction
banners. Do not cite the old 8B Alignment mean `+0.273`, old 8B RL-QA gains,
32B `SCALE_BOUNDARY`, 32B RL-QA `FAIL`, or the claimed 14B→32B verbal jump.

### Local objective study: excluded

The ignored local report
`data/results/metacog_objective_study_seed0/reports/METACOG_OBJECTIVE_STUDY_SEED0.md`
compares Binary/Soft/Pairwise/Listwise on Qwen3-8B seed 0. It uses the old
baseline (`original V=0.3439`), and its implementation is uncommitted. The
tracked v3 inventory correctly records it as absent. Do not use its conclusion
until code review, commit, and v3 evaluation of all locked adapters.

## Recommended paper tables

1. Cross-family capable-scale v3 measurement with paired gap CIs.
2. Qwen3-8B Alignment per-seed corrected table; 14B/32B as exploratory rows.
3. Qwen3-8B RL-QA per-seed corrected QA/admission table; 32B as boundary row.
4. Qwen3 scale/interface paired contrasts with non-monotonic caveat.
5. Negative results: Compositional non-replication, RL-W worse than SFT-W,
   Hybrid no added value, 32B Alignment AMBER.

Never pool seed-by-episode observations. Report individual seeds, mean and
sample SD. Shared-draw bootstrap CIs describe paired effects over the fixed
episode set, not random-effects uncertainty over three seeds. Keep exact
McNemar tests beside low-count QA effects.

## Paper-safe wording

> Across four capable-scale model families, the latent workspace predicts
> load-bearing information better than the verbal report. Workspace-derived
> self-distillation repairs this reporter gap in Qwen3-8B across three seeds
> while leaving workspace and full-context behavior stable. Separately,
> probe-blind RL-QA improves Qwen3-8B admission and budgeted recall over the
> original policy across three seeds.

Required qualifications:

- Do not claim replicated Compositional transfer at 8B.
- Do not claim monotonic Alignment scaling or that RL-QA beats SFT-W.
- Do not call A100 results H100 results.
- Do not restore withdrawn transition/boundary/32B-FAIL language.
- Keep near-floor QA and small discordant counts visible.

## Remaining work before camera-ready

- Re-evaluate the exact A5000 Qwen3-8B adapter under v3 if shown in a table.
- Rebuild the A5000 Qwen2.5-7B RL Original arm under v3.
- Restore/re-evaluate verbal-gated downstream artifacts marked absent.
- Replicate 14B and 32B Alignment before asserting a stable 14B peak.
- Decide whether chat penalty and 8B→32B gap widening belong in the main paper.
- Regenerate the separate paper repository exclusively from CURRENT v3 data.
