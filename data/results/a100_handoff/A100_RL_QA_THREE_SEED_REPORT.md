> **CORRECTED — the numbers below are superseded.**
> The Original arm was scored with the defective verbal probe, which also chose
> its budget-2 selected sets, so both its admission AUC and its QA accuracy were
> wrong. Re-evaluated against the same locked adapters the conclusion survives at
> smaller effect sizes: QA deltas **+7.35 / +5.88 / +8.82 pp** (was +8.82 / +7.35
> / +10.29) and admission deltas **+0.434 / +0.422 / +0.460** (was +0.494 / +0.482
> / +0.520). All three seeds still clear +5 pp.
> Corrected data: `data/results/a100_next_boundary_campaign/qwen3_rlqa_v3/Qwen3-8B/`.
> Root cause: [`CORRECTIONS.md`](../../../CORRECTIONS.md).

# Qwen3-8B RL-QA: three-seed NVIDIA A100-SXM4-80GB replication

Report schema: `rlqa-three-seed-report/v1`. Decision: **PASS**.

Primary source `decoupled`, budget 2. Seed x episode observations are never pooled; the mean paired effect uses shared episode draws.

## 1. Success criteria

| Criterion | Result |
|---|---|
| Primary: every seed's Decoupled QA delta vs Original is positive | PASS |
| Primary: three-seed mean paired effect positive, 95% CI excludes zero | PASS |
| Admission: every seed's Decoupled admission-AUC delta positive | PASS |
| No harm: full-context QA drop <= 2 pp per seed | PASS |
| No harm: fresh W_rr drop <= 0.03 per seed | PASS |

## 2. Per-seed Decoupled results

| Seed | QA (original) | QA (RL-QA) | QA delta (pp) | admission AUC delta | CI 95% | exact McNemar p | full-context QA drop (pp) | W_rr drop |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 0.0882 | +8.82 | +0.49433 | [+0.4196, +0.5741] | 0.03125 | +1.47 | +0.00085 |
| 1 | 0.0000 | 0.0735 | +7.35 | +0.48210 | [+0.4088, +0.5628] | 0.0625 | -0.00 | +0.00006 |
| 2 | 0.0000 | 0.1029 | +10.29 | +0.52038 | [+0.4495, +0.5962] | 0.01562 | +1.47 | +0.00085 |

## 3. Three-seed aggregate

| Quantity | seed 0 | seed 1 | seed 2 | mean | sample sd |
|---|---|---|---|---|---|
| Decoupled QA delta (pp) | +8.82353 | +7.35294 | +10.29412 | +8.82353 | 1.47059 |
| admission AUC delta | +0.49433 | +0.48210 | +0.52038 | +0.49894 | 0.01955 |
| full-context QA drop (pp) | +1.47059 | -0.00000 | +1.47059 | +0.98039 | 0.84904 |
| W_rr drop | +0.00085 | +0.00006 | +0.00085 | +0.00059 | 0.00046 |

## 4. Shared-draw mean paired QA effect

Mean Decoupled QA gain across seeds: **+8.82 pp**, 95% CI [+2.94, +15.69] pp over 4000 shared episode-cluster draws on 68 episodes.

## 5. Compositional diagnostics

| Seed | QA delta (pp) | admission AUC delta |
|---|---|---|
| 0 | +7.69 | +0.53561 |
| 1 | +7.69 | +0.54297 |
| 2 | +13.46 | +0.54555 |

## 6. Frozen recipe

```json
{
  "answer_tokens": 64,
  "beta": 0.03,
  "budget": 2,
  "checkpoint_selection": "strict first maximum of ID QA on the fixed ID validation split",
  "dtype": "bfloat16",
  "eval_every": 100,
  "group_size": 8,
  "grpo_epochs": 2,
  "lambda_qa": 1.0,
  "lambda_w": 0.0,
  "learning_rate": 1e-06,
  "lora_rank": 32,
  "max_length": 2048,
  "max_steps": 300,
  "optimisation_seeds": [
    0,
    1,
    2
  ],
  "save_every": 300,
  "split_seed": 0,
  "temperature": 5.0
}
```

Temperature decision:

```json
{
  "decision": "temperature = 5.0",
  "evidence": {
    "T=0.7": {
      "artifact": "data/results/rlqa_a100/b0_s0_k2_g16/summary.json",
      "gate_b0": "GREEN",
      "median_unique_selected_sets": 7.0,
      "mixed_QA_reward_groups_fraction": 0.8742857142857143
    },
    "T=5.0": {
      "artifact": "data/results/rlqa_a100/b0_s0_k2_g16_t5/summary.json",
      "gate_b0": "GREEN",
      "median_unique_selected_sets": 7.0,
      "mixed_QA_reward_groups_fraction": 0.88
    }
  },
  "note": "The preflight script's own selection rule (lowest candidate with median unique sets >= 4) returns 0.7 for Qwen3-8B. The controlling H100 plan overrides that rule for this campaign: 5.0 is retained because it passes both B0 gates. Neither temperature was chosen using Decoupled or Compositional data.",
  "rule_applied": "docs/H100_NEXT_CAMPAIGNS.md: keep 5.0 if it satisfies the existing mixed reward/diversity gates; otherwise stop and write a pre-OOD amendment"
}
```

## 7. Pre-OOD checkpoint lock

| Seed | selected step | checkpoint tree SHA-256 |
|---|---|---|
| 0 | 300 | `1808d8df596a02f367dc6d18f9f22e1e0d1cd928bc55d0b71b63ee9a3254a417` |
| 1 | 300 | `9fbe892bd5fe338a1d0bc258079ea80d63401025817bb2d0b6d9790afc074e7c` |
| 2 | 200 | `3447b330539cd23402bb62bac71d216bf11c1a75061a5c644fd37cd6cf0e8d64` |

Lock manifest self-hash: `6b2e29d26e7965dd70acd2aa9a03fb66400f1f84511c67b8e5d9918f11d16cb5`. Shared split manifest: `09e64597d095e6f0122b9e6ec291fbf44cdf2dda4559540fc3e087b420e3025b`.

## 8. Scope and stop

This allocation trained RL-QA only. RL-W, Hybrid, Soft Binary, Pairwise, Listwise, larger models and combined RL+metacognitive objectives were not run. No SFT-W baseline exists at this scale, so the SFT-W comparison named in the controlling plan is reported as unavailable rather than silently promoted or replaced. This report is evidence for manual review and authorises nothing further.

## 9. Operator command plan and run directories

### Seed 0 — `data/results/rlqa_a100/runs/formal_rl-qa_Qwen3-8B_rank_continuous_split0_s0_beta0p03_k2_lq1_lw0`


### Seed 1 — `data/results/rlqa_a100/runs/formal_rl-qa_Qwen3-8B_rank_continuous_split0_s1_beta0p03_k2_lq1_lw0`


### Seed 2 — `data/results/rlqa_a100/runs/formal_rl-qa_Qwen3-8B_rank_continuous_split0_s2_beta0p03_k2_lq1_lw0`


## 10. Environment, hardware and hashes

- Repository commit: `b85e5b59612c3a835f6d3fefc8923dcc386781f0`
- Python 3.12.14, conda prefix `/rodata/azradonc_dev/m253405/myconda/envs/jspace`
- Packages: accelerate 1.14.0, datasets 5.0.1, numpy 2.5.2, peft 0.13.2, scikit-learn 1.9.0, scipy 1.18.1, torch 2.7.1+cu126, transformers 4.57.6

GPU inventory reported by the collecting host:

```
0, GPU-1f184c0a-df97-4048-f4b5-e0a3923e1dbe, NVIDIA A100-SXM4-80GB, 580.159.04, 81920 MiB
1, GPU-da080f5a-c107-1f33-2e8a-0dc554a830fc, NVIDIA A100-SXM4-80GB, 580.159.04, 81920 MiB
2, GPU-67f94b0f-850b-648d-f760-5cc2601b21a2, NVIDIA A100-SXM4-80GB, 580.159.04, 81920 MiB
3, GPU-dd7a83fd-cf45-e17f-6515-c81f3bf08606, NVIDIA A100-SXM4-80GB, 580.159.04, 81920 MiB
```

Benchmark hashes:

| File | SHA-256 |
|---|---|
| `data/benchmarks/battery_v1_final.json` | `35d973619080a7606b0a3046ffcc7169fd919fac950ebb0d38b67ac13b017525` |
| `data/benchmarks/battery_v2_final.json` | `e8ce4db85d3a0c09879575850e36b91821af8be155e86495cd059ae35485e7d6` |
| `data/benchmarks/battery_v2_g2.json` | `32b96a17bc35c895ee1fd6f6b49b6dc7c7a888b1a83e247925b1d9c8c8660cf3` |
| `data/benchmarks/battery_v3d.json` | `483d3fb6a72970c99c1cba2ed118dd6289825b7f73332af07300205ceb361432` |
| `data/benchmarks/battery_v4_final.json` | `4550cf11c9b7d837b0b1d921c9151d997cb1e421c800fc21da2ebafe4203bf25` |

Executed source hashes:

| File | SHA-256 |
|---|---|
| `scripts/lock_rlqa_checkpoints.py` | `abf5ac95cf6fe6d7d0b545f33469ce664969fb8b2908c9b1227e4fbea399bbd1` |
| `scripts/report_metacog_three_seed.py` | `1f7eeb48c8419dd92c614aa74cbc937ae6f7abdb4179deb6c6775ccc56fb8c1f` |
| `scripts/run_metacog_alignment_campaign.py` | `17f96a12b96cf06ad3d032b0ca95696a8f6691b9a0a46cfe2e00c266d48dc00f` |
| `src/analysis/evaluate_metacog_m1_ood.py` | `89bab654707aa2a7b0f0ba38d5bca3e76b5a8f0f07a3d51db975cbc76ec69b41` |
| `src/analysis/gate_metacog_m0.py` | `df589f5b26d90abdbdcd413bac9029e5b0dde8c5b4ff3d577c000d3aa26ff70e` |
| `src/analysis/memory_rl_gates.py` | `f5f9e0e9b4eb92ba163f2bdb7e6b5b8ce8ee357df8f3fa2c0eb68783d79703d2` |
| `src/analysis/report_metacog_m1.py` | `f231b6f95e0de6f5ede61b4dff2a62b5b7c6b4fb61a88eb243067619e4ecb24a` |
| `src/experiments/evaluate_memory_rl.py` | `ee1b054422860bc62950a448ef3cb15bf31dbd2234e29bc2bd82fccb27ea7874` |
| `src/experiments/measure.py` | `b886cd27352a796feea3d4801aca651dd68d255920d83e19271eb7da20137cf9` |
| `src/experiments/preflight_qa_reward.py` | `ba19e91ef77fed94f99d8650913893ecb9f073d9d16b0dd6209d95ef8eeb81b0` |
| `src/experiments/train_memory_rl.py` | `ff3ee5f3c37437e60eda9c29af4aedc5c9f942ae26d922f4530ba4e77aef9b71` |
| `src/experiments/train_metacog_m1.py` | `e4ee5efdde3ee5a868cc4d2d7e0af12b2b717805aaa1c478a97a0e0a50ef12de` |
| `src/jlens.py` | `5251f792f44c8e2846564036ec2a945290ec74fbb86e67aef2cb74690c5582c4` |
| `src/memory_rl/recall.py` | `b402272dc457b7b7a87a6cb40fd01051af2f817e32b9ba0bd21d6eed9170f238` |

## 11. Exceptions and deviations from the controlling plan

1. Hardware: the controlling plan names H100; this allocation ran on NVIDIA A100-SXM4-80GB, one GPU per Slurm job.
2. Scope: RL-QA only, as the controlling plan directs. No SFT-W run exists at Qwen3-8B scale, so the SFT-W comparison is reported as unavailable rather than substituted.
3. src/analysis/stage_b_rlqa_multiseed_ood.py was not used: its pre-OOD lock contract hard-codes the completed Qwen2.5-7B campaign's SFT adapter path and requires an SFT-W adapter this RL-QA-only allocation does not have. The predeclared statistics it would compute (4,000-draw episode-cluster bootstrap, exact per-seed McNemar, per-seed values with mean and sample sd, no seed x episode pooling) are produced here instead.
4. Temperature 5.0 was retained under the controlling plan's rule after a dedicated Stage-B0 run at T=5 passed both gates; the preflight script's own lowest-passing-candidate rule would have selected 0.7. Neither temperature was chosen using Decoupled or Compositional data.
5. Absolute budget-2 recall QA on Decoupled is near the floor for every condition (Original 0.000, oracle 0.059), so the positive per-seed deltas rest on few discordant episodes. Exact McNemar counts are reported per seed so the reader can see exactly how many episodes each gain rests on.
