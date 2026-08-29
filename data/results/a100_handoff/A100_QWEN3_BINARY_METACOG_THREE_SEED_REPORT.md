# Qwen3-8B Binary Metacognitive Alignment: three-seed NVIDIA A100-SXM4-80GB replication

Report schema: `metacog-alignment-three-seed-report/v1`. Replication decision: **GREEN**.

The predeclared gate is applied to each seed independently and the replication is GREEN only if all three seeds are GREEN. Pooled values describe the effect; they never replace a per-seed gate.

## 1. Model and hardware

- Model: `Qwen/Qwen3-8B`
- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Tokenizer revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Accelerator: NVIDIA A100-SXM4-80GB

## 2. Per-seed Decoupled gate

### Decoupled (primary, predeclared)

| Seed | V before | V after | delta V | delta W | full-context QA before | after | drop (pp) | gate |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.34154 | 0.56620 | +0.22466 | +0.00017 | 0.7059 | 0.7059 | +0.00 | GREEN |
| 1 | 0.34154 | 0.68886 | +0.34732 | +0.00091 | 0.7059 | 0.7059 | +0.00 | GREEN (strong) |
| 2 | 0.34154 | 0.58923 | +0.24769 | -0.00028 | 0.7059 | 0.6912 | +1.47 | GREEN |

## 3. Compositional diagnostics

Mandatory diagnostics; they do not alter the Decoupled primary gate.

### Compositional (diagnostic)

| Seed | V before | V after | delta V | delta W | full-context QA before | after | drop (pp) | gate |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.32619 | 0.42648 | +0.10029 | +0.00308 | 0.7115 | 0.7115 | +0.00 | NOT_GREEN |
| 1 | 0.32619 | 0.59928 | +0.27310 | +0.00087 | 0.7115 | 0.7308 | -1.92 | GREEN (strong) |
| 2 | 0.32619 | 0.46835 | +0.14216 | +0.00023 | 0.7115 | 0.7308 | -1.92 | NOT_GREEN |

## 4. Three-seed aggregate (Decoupled)

| Quantity | seed 0 | seed 1 | seed 2 | mean | sample sd |
|---|---|---|---|---|---|
| delta V | +0.22466 | +0.34732 | +0.24769 | +0.27322 | 0.06520 |
| V after | +0.56620 | +0.68886 | +0.58923 | +0.61476 | 0.06520 |
| delta W | +0.00017 | +0.00091 | -0.00028 | +0.00027 | 0.00060 |
| full-context QA drop (pp) | +0.00000 | +0.00000 | +1.47059 | +0.49020 | 0.84904 |

## 5. Shared-draw mean paired effect

Mean Decoupled delta V across seeds: **+0.27322**, 95% CI [+0.21048, +0.34195] over 4000 shared episode-cluster draws.

## 6. Gate thresholds

```json
{
  "max_abs_delta_w": 0.03,
  "max_qa_drop_pp": 2.0,
  "min_delta_v": 0.15,
  "min_v_after": 0.5,
  "strong_delta_v": 0.25
}
```

## 7. Source artifacts

| Seed | one-shot OOD result |
|---|---|
| 0 | `data/results/metacog_a100_3seed/seed0/ood/result.json` |
| 1 | `data/results/metacog_a100_3seed/seed1/ood/result.json` |
| 2 | `data/results/metacog_a100_3seed/seed2/ood/result.json` |

## 8. Stop

This report is evidence for manual review. It authorises no extension, no additional seed, no larger model, and no combined objective.

## 9. Operator command plan and run directories

### Seed 0 — `data/results/metacog_a100_3seed/seed0`

Executed command plan (ID phase):

```
[m0] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/measure.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --dtype bfloat16 --device cuda --end-only --no-verbal-raw --battery /rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json --out /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/explicit.json
[m0_gate] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/analysis/gate_metacog_m0.py --explicit /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/explicit.json --evoked /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/evoked.json --decoupled /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/decoupled.json --compositional /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/compositional.json --paper-v 0.337 --paper-w-rr 0.654 --tolerance 0.05 --out-json /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/gate.json --out-md /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/gate.md
[teacher_prep] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/measure.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --dtype bfloat16 --device cuda --end-only --no-verbal-raw --battery /rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json --out /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/evoked_g2.json
[canary] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/train_metacog_m1.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --seed 0 --train-spec explicit=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/explicit.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json --train-spec evoked=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/evoked.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_final.json --train-spec evoked_g2=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/evoked_g2.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json --out-dir /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/canary --canary-steps 10
[m1] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/train_metacog_m1.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --seed 0 --train-spec explicit=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/explicit.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json --train-spec evoked=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/evoked.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_final.json --train-spec evoked_g2=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m0/evoked_g2.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json --out-dir /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed0/m1
```

- ID-only locked step: **414** (selection scope `id_validation`, metric `verbal_auc`, tie-break `earliest_step`)
- Locked checkpoint: `m1/checkpoints/step-000414`
- Checkpoint tree SHA-256: `08ae9c7a2d36dfa960b64369970624969ef7e408ae3f0ffd36ef1a6bce3894b4`
- ID validation AUC at lock: 0.61324
- OOD attempt id: `bb245a9fe6ce4926b4c8d10bb543d474` (limit 1, conditions ['decoupled', 'compositional'])
- OOD attempt started: 2026-08-29T05:21:33+00:00
- decision ledger: `data/results/metacog_a100_3seed/seed0/decision_ledger.jsonl`
- raw command logs: `data/results/metacog_a100_3seed/seed0/command_logs`
- per-seed M1 gate report: `data/results/metacog_a100_3seed/seed0/report/M1_GATE_REPORT.md`
- ID-phase stop marker: `data/results/metacog_a100_3seed/seed0/ID_LOCK_STOP.json`
- final stop marker: `data/results/metacog_a100_3seed/seed0/STOP.json`

### Seed 1 — `data/results/metacog_a100_3seed/seed1`

Executed command plan (ID phase):

```
[m0] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/measure.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --dtype bfloat16 --device cuda --end-only --no-verbal-raw --battery /rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json --out /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/explicit.json
[m0_gate] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/analysis/gate_metacog_m0.py --explicit /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/explicit.json --evoked /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/evoked.json --decoupled /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/decoupled.json --compositional /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/compositional.json --paper-v 0.337 --paper-w-rr 0.654 --tolerance 0.05 --out-json /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/gate.json --out-md /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/gate.md
[teacher_prep] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/measure.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --dtype bfloat16 --device cuda --end-only --no-verbal-raw --battery /rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json --out /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/evoked_g2.json
[canary] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/train_metacog_m1.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --seed 1 --train-spec explicit=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/explicit.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json --train-spec evoked=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/evoked.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_final.json --train-spec evoked_g2=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/evoked_g2.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json --out-dir /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/canary --canary-steps 10
[m1] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/train_metacog_m1.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --seed 1 --train-spec explicit=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/explicit.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json --train-spec evoked=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/evoked.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_final.json --train-spec evoked_g2=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m0/evoked_g2.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json --out-dir /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed1/m1
```

- ID-only locked step: **414** (selection scope `id_validation`, metric `verbal_auc`, tie-break `earliest_step`)
- Locked checkpoint: `m1/checkpoints/step-000414`
- Checkpoint tree SHA-256: `6a669d69891d74fcc9fbe53959cdfb822d547bce12e6882133f3ecfd5c345d91`
- ID validation AUC at lock: 0.57522
- OOD attempt id: `6d0c7d29e4df4563935a6ce79b788fbf` (limit 1, conditions ['decoupled', 'compositional'])
- OOD attempt started: 2026-08-29T05:21:33+00:00
- decision ledger: `data/results/metacog_a100_3seed/seed1/decision_ledger.jsonl`
- raw command logs: `data/results/metacog_a100_3seed/seed1/command_logs`
- per-seed M1 gate report: `data/results/metacog_a100_3seed/seed1/report/M1_GATE_REPORT.md`
- ID-phase stop marker: `data/results/metacog_a100_3seed/seed1/ID_LOCK_STOP.json`
- final stop marker: `data/results/metacog_a100_3seed/seed1/STOP.json`

### Seed 2 — `data/results/metacog_a100_3seed/seed2`

Executed command plan (ID phase):

```
[m0] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/measure.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --dtype bfloat16 --device cuda --end-only --no-verbal-raw --battery /rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json --out /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/explicit.json
[m0_gate] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/analysis/gate_metacog_m0.py --explicit /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/explicit.json --evoked /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked.json --decoupled /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/decoupled.json --compositional /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/compositional.json --paper-v 0.337 --paper-w-rr 0.654 --tolerance 0.05 --out-json /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/gate.json --out-md /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/gate.md
[teacher_prep] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/measure.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --dtype bfloat16 --device cuda --end-only --no-verbal-raw --battery /rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json --out /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked_g2.json
[canary] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/train_metacog_m1.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --seed 2 --train-spec explicit=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/explicit.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json --train-spec evoked=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_final.json --train-spec evoked_g2=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked_g2.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json --out-dir /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/canary --canary-steps 10
[m1] /rodata/azradonc_dev/m253405/myconda/envs/jspace/bin/python /rodata/azradonc_dev/m253405/jspace-devtrace/src/experiments/train_metacog_m1.py --model Qwen/Qwen3-8B --model-revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --seed 2 --train-spec explicit=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/explicit.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v1_final.json --train-spec evoked=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_final.json --train-spec evoked_g2=/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked_g2.json::/rodata/azradonc_dev/m253405/jspace-devtrace/data/benchmarks/battery_v2_g2.json --out-dir /rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m1
```

- ID-only locked step: **250** (selection scope `id_validation`, metric `verbal_auc`, tie-break `earliest_step`)
- Locked checkpoint: `m1/checkpoints/step-000250`
- Checkpoint tree SHA-256: `d17669b04e453f5633bf5da51667bc8c5b164e34e2794539d1c6f9636ff414da`
- ID validation AUC at lock: 0.58500
- OOD attempt id: `911e40f4e819484ca749ef3353ad62ab` (limit 1, conditions ['decoupled', 'compositional'])
- OOD attempt started: 2026-08-29T05:21:33+00:00
- decision ledger: `data/results/metacog_a100_3seed/seed2/decision_ledger.jsonl`
- raw command logs: `data/results/metacog_a100_3seed/seed2/command_logs`
- per-seed M1 gate report: `data/results/metacog_a100_3seed/seed2/report/M1_GATE_REPORT.md`
- ID-phase stop marker: `data/results/metacog_a100_3seed/seed2/ID_LOCK_STOP.json`
- final stop marker: `data/results/metacog_a100_3seed/seed2/STOP.json`

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

1. Hardware: the controlling plan names H100; this allocation ran on NVIDIA A100-SXM4-80GB (gen-a100.p, one GPU per Slurm job). The launcher's device check was not weakened: EXPECTED_GPU became a closed allowlist selected by METACOG_EXPECTED_GPU, still matching one exact device name, and the A5000 default is unchanged.
2. Launcher: docs/H100_NEXT_CAMPAIGNS.md asks for a fresh non-A5000 launcher. Rather than fork 2k lines, the existing orchestrator was parameterised for device and for the preregistered seeds {0,1,2}; the A5000 seed-0 contract still reproduces byte-for-byte with the environment variable unset (191 pre-existing tests pass under both configurations).
3. Two-phase execution: new --stop-after id_lock and --resume-ood flags let all three seeds reach an ID-only lock before any seed opened Decoupled/Compositional. Each seed still received exactly one OOD attempt, recorded in ood/attempt_started.json, and a resumed phase refuses to run if the ledger already records an attempt.
4. Report naming: the plan's required filename is H100_QWEN3_BINARY_METACOG_THREE_SEED_REPORT.md. This file is named A100_... so that A100 evidence is never mistaken for H100 evidence.
5. Infrastructure incidents, both preserved and not overwritten: seed0_failed_gpu_lock_collision_20260828 (index-keyed /tmp GPU lock collided when two array tasks shared a node; the lock is now keyed on the physical GPU UUID) and seed{0,1,2}_failed_device_gate_20260829 (train_metacog_m1.py held a second hard-coded A5000 check). Neither consumed an OOD attempt; both failed before any sealed data was touched.
6. HF cache: the pinned snapshot was fetched by commit, which leaves no refs/main entry, so offline loads that resolve the default revision failed. refs/main was written with the same commit b968826d9c46dd6066d109eabc6255188de91218, which README.md records as the repository's current main; every run still pins the revision explicitly.
