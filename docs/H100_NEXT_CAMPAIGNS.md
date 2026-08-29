# Next H100 campaigns

For a map of all repository documentation and the recommended operator reading
order, first read [`H100_DOCUMENT_GUIDE.md`](H100_DOCUMENT_GUIDE.md).

This is the single handoff entry point after the completed A5000 campaigns.
Nothing in this document authorizes an unattended launch: verify the H100 host,
implement and test the two launchers, print both plans, then obtain a final
operator review before allocating GPUs.

The next allocation contains two separate confirmatory tracks:

1. **H100 Memory-RL capable-scale replication** — locked RL-QA only.
2. **H100 Qwen3-8B Binary Metacognitive Alignment — seeds 0/1/2.**

Do not merge the objectives, initialize one track from the other track's
adapter, or use results from one track to tune the other. A combined/factorial
study would be a later, separately approved campaign.

## Evidence entering H100

### Memory-RL

The completed Qwen2.5-7B campaign establishes:

- RL-W failed Gate A and was not expanded.
- RL-QA replicated over Original across seeds 0/1/2. On Decoupled, its mean QA
  gain was +7.35 percentage points and its mean admission-AUC gain was +0.06731.
- RL-QA did not reproducibly improve downstream QA over SFT-W.
- Hybrid-0.25 had no added Decoupled QA or containment value over RL-QA.

Therefore the only RL method promoted to H100 is **RL-QA**. Do not spend the
confirmatory allocation on RL-W, Hybrid coefficient sweeps, or OOD tuning.
The source report is
`data/results/memory_rl_campaign_20260826/reports/STAGE_B1_RL_QA_THREE_SEED_REPORT.md`.

### Binary Metacognitive Alignment

The Qwen3-8B A5000 seed-0 pilot is Strong GREEN:

- Decoupled V AUC: 0.34391 -> 0.65516, delta +0.31125;
- Decoupled W_rr AUC delta: +0.00074;
- Decoupled full-context QA drop: 0 percentage points;
- ID-only selected checkpoint: step 250;
- 4,000-draw Decoupled V delta CI: [+0.238, +0.392].

The source report is
`data/results/metacog_alignment_campaign_seed0_retry1/report/M1_GATE_REPORT.md`.

## Track A — H100 Memory-RL capable-scale replication

### Question

Does direct future-QA reward reproducibly improve memory admission and
budgeted recall at Qwen3-8B scale, without damaging the latent workspace or
full-context capability?

### Locked starting specification

- Model: `Qwen/Qwen3-8B`.
- Model/tokenizer revision:
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Method: RL-QA only; `lambda_QA=1`, `lambda_W=0`.
- Optimization seeds: 0, 1, 2; split seed remains 0 for every run.
- Budget: 2; group size: 8; LoRA rank: 32; bf16.
- Optimizer steps: 300; learning rate: `1e-6`; KL beta: `0.03`;
  two GRPO epochs.
- Admission is probe-blind. Recall uses the adapter-disabled frozen model.
- Training/ID data remain Explicit, Evoked, and Evoked-G2 only.
- Decoupled and Compositional remain sealed until each seed has an ID-only
  checkpoint lock.

The sampling temperature `5.0` was calibrated for Qwen2.5-7B and is **not
silently portable**. Before formal H100 training, run a Qwen3-8B ID-train-only
exact-set diversity preflight. Keep 5.0 if it satisfies the existing mixed
reward/diversity gates; otherwise stop and write a separate pre-OOD amendment.
Do not choose temperature using Decoupled or Compositional.

### Execution order

1. Add Qwen3-8B support to the Memory-RL launcher without changing the existing
   Qwen2.5-7B reproduction path.
2. Run unit tests, a data-only dry run, the Stage-B0 reward/diversity preflight,
   and a 5–20 step H100 canary.
3. Freeze one Qwen3-8B RL-QA recipe before formal training.
4. Train seeds 0/1/2 independently; select each checkpoint by the existing
   strict first maximum of ID QA.
5. Lock all three checkpoints and hashes before any OOD access.
6. Evaluate each seed once on Decoupled and Compositional with batch-1 isolated
   recall, 4,000 episode-cluster bootstrap draws, and exact per-seed McNemar.
7. Aggregate individual values, mean, and sample standard deviation without
   pooling seed-by-episode observations.
8. Generate the H100 RL-QA report and stop for manual review.

### RL success criteria

- Primary: every seed's Decoupled QA delta versus Original is positive and the
  three-seed mean paired effect is positive with a 95% episode-bootstrap CI
  excluding zero.
- Admission support: every seed's Decoupled admission-AUC delta versus Original
  is positive.
- No harm: full-context QA drop is at most 2 percentage points and fresh W_rr
  drop is at most 0.03 per seed.
- Comparisons against SFT-W are reported but are not silently promoted to the
  primary claim if unresolved.

## Track B — H100 Qwen3-8B Binary Metacognitive Alignment x 3 seeds

### Question

Does the A5000 Strong-GREEN reporter repair replicate across optimization seeds
on H100 while keeping workspace and full-context behavior stable?

### Locked recipe

- Model/tokenizer revision:
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Seeds: 0, 1, 2. Seed 0 is rerun on H100; do not mix its A5000 checkpoint into
  the H100 three-seed aggregate.
- Frozen original `M_ref`; within each episode, top-2 frozen `W_ref` candidates
  are Yes and all others are No.
- Train only Explicit, Evoked, and Evoked-G2 with the same split construction.
- Binary target, LoRA rank 16, alpha 32, all q/k/v/o and gate/up/down targets.
- bf16, batch 1, accumulation 4, AdamW, LR `1e-5`, two epochs, 500-step cap,
  maximum length 1024 with the existing deterministic length ladder.
- Evaluate step 0, 100, 250, and actual terminal; select highest ID verbal AUC,
  with earliest-step tie-break.
- Keep Decoupled and Compositional sealed until each seed is independently
  locked. Each seed receives exactly one combined OOD attempt.

The H100 launcher must preserve the corrected in-place PEFT checkpoint
round-trip verification. It must accept only an H100 device and must not reuse
the A5000 campaign directory.

### Execution order

1. Build a fresh H100-only launcher and output schema; do not weaken the A5000
   runner's device checks.
2. Verify the immutable model/cache, all five benchmark hashes, software
   versions, bf16, and sufficient H100 memory.
3. Run the full test suite and one 5–20 step H100 canary.
4. Run formal Binary M1 for seeds 0/1/2 in separate fresh directories.
5. Create all three ID locks before opening OOD for any seed.
6. Run one-shot Decoupled + Compositional OOD per seed, using 4,000 paired
   episode-cluster bootstrap draws and the same full-context QA audit.
7. Report every seed, mean, sample standard deviation, and a shared-draw mean
   paired effect; never replace individual-seed gates with only a pooled result.
8. Generate the 3-seed replication report and stop.

### Metacognitive success criteria

Apply the existing gate independently to every seed:

- Decoupled delta V >= +0.15 and V_after > 0.50;
- absolute Decoupled delta W < 0.03;
- Decoupled full-context QA drop <= 2 percentage points.

The three-seed replication is GREEN only if all three seeds are GREEN. Report
Strong GREEN per seed when delta V >= +0.25. Compositional metrics, including
the A5000 seed-0 QA drop, remain mandatory diagnostics but do not alter the
predeclared Decoupled primary gate.

## Scheduling, isolation, and required deliverables

- Prefer one campaign per physical H100 at a time. If tracks run concurrently,
  use disjoint GPU UUIDs, locks, caches, and run directories.
- Never resume into or overwrite a prior run directory.
- Preserve raw logs, configs, split/teacher manifests, checkpoint hashes,
  decision ledgers, OOD attempt records, validators, and Markdown reports.
- Do not launch additional seeds, RL-W, Hybrid, Soft Binary, Pairwise, Listwise,
  32B/70B, or a combined RL+metacognitive objective from these campaigns.
- Required final reports:
  `H100_RL_QA_THREE_SEED_REPORT.md` and
  `H100_QWEN3_BINARY_METACOG_THREE_SEED_REPORT.md`.

## Pre-launch checklist

- [ ] H100 hardware and scheduling interface documented.
- [ ] Both H100 launchers implemented and covered by tests.
- [ ] Exact command plans reviewed before output directories are created.
- [ ] Model, tokenizer, battery, and source-code hashes recorded.
- [ ] RL-QA Qwen3 reward/diversity preflight passes without OOD access.
- [ ] Both engineering canaries pass checkpoint reload and no-harm checks.
- [ ] Seeds, split seed, gates, and one-shot OOD rules are frozen.
- [ ] Operator explicitly authorizes the H100 allocation.
