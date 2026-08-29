# A100 runbook for the two confirmatory tracks

This is the operational record of running `docs/H100_NEXT_CAMPAIGNS.md` on an
A100 cluster instead of an H100 host. It does not change any scientific
contract in that document; it records how the same contract was executed, what
had to become configurable, and where the evidence lives.

## Hardware actually used

| Item | Value |
|---|---|
| Partition | `gen-a100.p` (19 nodes, `gpu:a100:4`, 96 CPUs, ~1 TB RAM) |
| Device | `NVIDIA A100-SXM4-80GB` (driver 580.159.04) |
| GPUs per job | **1** — every stage is a separate single-GPU Slurm job |
| Environment | conda env `jspace`: torch 2.7.1+cu126, transformers 4.57.6, peft 0.13.2 |
| Model cache | `HF_HOME=/rodata/azradonc_dev/m253405/cache`, `HF_HUB_OFFLINE=1` |

Compute nodes have no outbound internet, so `Qwen/Qwen3-8B` at revision
`b968826d9c46dd6066d109eabc6255188de91218` is pre-staged from the login node.
Fetching by commit leaves no `refs/main` entry, which breaks offline loads that
resolve the default revision; `refs/main` was written with that same commit, and
every command still pins the revision explicitly.

## What became configurable, and what did not

The A5000 launcher was parameterised rather than forked. Nothing was loosened:

- `METACOG_EXPECTED_GPU` selects **one exact device name** from a closed
  allowlist (`SUPPORTED_GPUS`). Unset, it is `NVIDIA RTX A5000` and the
  completed A5000 seed-0 campaign reproduces unchanged.
- `train_metacog_m1.py` held a second hard-coded A5000 check; it now compares
  against the same constant.
- Seeds are the preregistered `{0, 1, 2}`; anything else is still refused.
- The GPU exclusion lock is keyed on the **physical GPU UUID**, not the visible
  index. Under a cgroup-isolated scheduler every job sees its GPU as index 0, so
  the old index-keyed lock made two jobs on one node collide.
- New `--stop-after id_lock` / `--resume-ood` split the campaign in two phases
  so all three seeds hold an ID-only lock before any seed opens OOD. A resumed
  phase refuses to run if the ledger already records an OOD attempt.

## Job graph

```
Track B (metacognitive alignment)          Track A (RL-QA)
-------------------------------            ---------------------------------
metacog_id_phase.sbatch  (array 0-2)       rlqa_teacher_measure.sbatch
  M0 x4 -> gate -> teacher -> canary         W_ref rows: v1f / v2f / v2g2
  -> M1 -> ID lock, OOD sealed             (login) train_memory_rl --dry-run
        |                                    -> split manifest hash
        v  all three locks exist            rlqa_b0_preflight.sbatch  (+ _t5)
metacog_ood_phase.sbatch (array 0-2)         Gate B0, temperature decision
  one-shot Decoupled + Compositional              |
  -> per-seed M1_GATE_REPORT.md             RECIPE_FREEZE.json
        |                                        |
        v                                   rlqa_canary.sbatch (frozen recipe)
report_metacog_three_seed.py                     | afterok
  -> A100_QWEN3_BINARY_METACOG_...md        rlqa_formal.sbatch (array 0-2)
                                                 |
                                            lock_rlqa_checkpoints.py
                                                 |  all three locks exist
                                            rlqa_ood_eval.sbatch
                                            rlqa_workspace_noharm.sbatch
```

Every `slurm/*.sbatch` sources `slurm/a100_env.sh`, which sets the environment,
the expected device, and the pinned revision in one place.

## Order of operations that must not be reordered

1. All three ID locks exist **before** any seed's OOD job is submitted. Both
   OOD launchers re-check every seed's lock file and refuse otherwise.
2. Each seed gets exactly one OOD attempt, recorded in
   `ood/attempt_started.json` before the command runs. A failed attempt is not
   retried; the run directory is preserved and reported.
3. A failed gate or validator preserves its run directory. Two infrastructure
   failures are kept as `*_failed_gpu_lock_collision_*` and
   `*_failed_device_gate_*`; neither consumed an OOD attempt.

## Deliverables

- `data/results/a100_handoff/A100_QWEN3_BINARY_METACOG_THREE_SEED_REPORT.md`
- `data/results/a100_handoff/A100_RL_QA_THREE_SEED_REPORT.md`
- `data/results/a100_handoff/provenance.json` (env, GPU inventory, data and
  source hashes)

Reports carry an `A100_` prefix so A100 evidence is never mistaken for the
H100 evidence the controlling plan asks for.
