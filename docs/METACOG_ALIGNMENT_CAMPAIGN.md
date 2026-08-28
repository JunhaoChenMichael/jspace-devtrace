# Metacognitive Alignment scaling campaign

This runbook operationalizes only the NVIDIA RTX A5000 gate for capable-scale
Metacognitive Alignment. The launcher stops after the Qwen3-8B seed-0 M1 report.
It contains no H100 command or transition.

The executable entry point is:

```bash
.venv-metacog/bin/python scripts/run_metacog_alignment_campaign.py \
  --model-revision MODEL_COMMIT_SHA \
  --tokenizer-revision TOKENIZER_COMMIT_SHA \
  --gpu-index PHYSICAL_A5000_INDEX
```

Both revisions must be immutable 40-hex Hugging Face commit IDs. The primary
target is fixed to `Qwen/Qwen3-8B`; the Qwen2.5-7B engineering fallback is not a
permitted silent substitute in this campaign.

Qwen3 requires `transformers>=4.51.3`. Run the campaign from the dedicated
`.venv-metacog` environment; the host's older default environment cannot load
`Qwen3Config`. Because the launcher builds every child command from its own
`sys.executable`, starting it with `.venv-metacog/bin/python` keeps M0, M1, OOD,
and reporting in that same environment.

The reproducible dependency overlay is `requirements-metacog.txt`:

```bash
.venv-metacog/bin/pip install -r requirements-metacog.txt
```

## Scope and separation from the completed RL work

Metacognitive Alignment repairs the reporter:

```text
frozen Qwen3-8B workspace W_ref -> top-2 pseudo-labels -> SFT/LoRA -> verbal V
```

The completed RL campaign trains memory admission for future QA utility:

```text
selection -> future QA -> RL reward -> memory-control policy
```

These are separate interventions. The RL result is already recorded as RL-W
failed, RL-QA replicated over Original across three seeds, and Hybrid W+QA C3
with no added value. This launcher does not call
`scripts/run_memory_rl_mvp.sh`, `src/experiments/train_memory_rl.py`, reuse an
RL adapter, import an RL gate, or merge RL results into an M0/M1 decision. See
`docs/RL_EXPERIMENTS.md` for that completed campaign.

## Fixed execution graph

The only automatic graph is:

```text
M0 measurements
  -> M0 reproduction gate
  -> frozen-teacher Evoked-G2 preparation
  -> 10-step canary
  -> M1 seed-0 pilot
  -> ID-only checkpoint lock
  -> one-shot Decoupled + Compositional OOD
  -> M1 Markdown report
  -> STOP for manual review
```

There is no branch to M2, no H100 scheduler command, and no model larger than
Qwen3-8B. A GREEN M1 decision is evidence for a human to review, not permission
for this launcher to allocate H100 compute.

The two hard training barriers are:

- M0 must be exactly `GREEN`. `INVESTIGATE`, a malformed gate, or a gate command
  error stops before the canary.
- The canary must exit successfully and write a manifest with
  `canary_passed=true`, a passing `status`, and `eligible_for_ood=false`.
  Otherwise formal M1 is not launched.

## Output protection and GPU preflight

The default output is date-stamped in UTC:

```text
data/results/metacog_alignment/YYYY-MM-DD_qwen3-8b_a5000_seed0/
```

If that path already exists, including as a symlink, the launcher refuses to
start. It never resumes into or overwrites a prior campaign. Use an explicit,
fresh `--run-dir` for a separately preregistered run.

Before any scientific command, the launcher records and validates two
`nvidia-smi` calls. The host may contain multiple GPUs, but the campaign selects
and reserves exactly one physical card. Pass `--gpu-index` on a shared server;
if it is omitted, the launcher deterministically chooses the idle eligible
A5000 with the most free memory. It requires:

- the selected card's exact device name to be `NVIDIA RTX A5000`;
- at least 22,000 MiB total and free memory;
- no existing compute process on the selected GPU UUID; and
- a nonblocking campaign lock for that physical index.

Processes on other GPUs are recorded but do not block the selected card. Every
child command receives the verified physical index as `CUDA_VISIBLE_DEVICES`.
An H100 selection, inadequate free memory, an unparseable process query, an
occupied target A5000, or a held campaign lock all fail closed. The 22,000 MiB
floor cannot be lowered by CLI.

Some shared hosts run a small resident monitoring/display process on every GPU.
The default remains exclusive (`0` MiB allowed). After inspecting the PID and
confirming at least 22,000 MiB is still free, an operator may explicitly allow
at most 512 MiB total resident usage, for example
`--allow-existing-processes-under-mib 512`. The process list and exception are
written to the ledger. This flag never permits a large co-tenant job and never
lowers the free-memory gate.

To inspect the fully rendered command plan without touching the filesystem or
querying a GPU:

```bash
.venv-metacog/bin/python scripts/run_metacog_alignment_campaign.py \
  --model-revision MODEL_COMMIT_SHA \
  --tokenizer-revision TOKENIZER_COMMIT_SHA \
  --print-plan
```

## M0 baseline and reproduction gate

`src/experiments/measure.py` is called in bf16 on the frozen original model.
It skips the unused `V_raw` pass; M0 computes only the required V and W signals.
The default plan first writes immutable raw results and `.metadata` sidecars
for the four M0 conditions. Only after the M0 gate is GREEN, it measures
Evoked-G2 as an ID-only frozen-teacher input for M1:

| Name | Repository battery | Role |
|---|---|---|
| Explicit | `data/benchmarks/battery_v1_final.json` | M0 and M1 train pool |
| Evoked | `data/benchmarks/battery_v2_final.json` | M0 and M1 train pool |
| Evoked-G2 | `data/benchmarks/battery_v2_g2.json` | M1 train pool only |
| Decoupled | `data/benchmarks/battery_v4_final.json` | M0 gate and sealed OOD |
| Compositional | `data/benchmarks/battery_v3d.json` | M0 characterization and sealed OOD |

The gate command is:

```text
src/analysis/gate_metacog_m0.py
  --explicit PATH --evoked PATH --decoupled PATH --compositional PATH
  --paper-v 0.337 --paper-w-rr 0.654 --tolerance 0.05
  --out-json PATH --out-md PATH
```

It checks model/tokenizer revisions, chat-template provenance, dtype, Yes/No
token sets, runtime metadata, raw hashes, counts, pooled AUC, within-episode
AUC, and Yes rate. Its primary Decoupled gate is:

```text
abs(V_new - 0.337) <= 0.05
AND
abs(W_rr_new - 0.654) <= 0.05
```

The gate writes `decision=GREEN` or `decision=INVESTIGATE`. The latter may use
exit status 1 after writing its artifacts; the launcher records the result and
stops without training.

## Canary and M1 training contract

`src/experiments/train_metacog_m1.py` is the only trainer used. The launcher
passes exactly three canonical training sources: Explicit, Evoked, and
Evoked-G2. Decoupled, Decoupled-L, Compositional, confusable-absent, and
LoCoMo-wrapped Decoupled are never passed as training or ID-validation sources.

The trainer owns the scientific details and provenance checks:

- `M_ref` is the frozen original revision and provides `W_ref` labels;
- the trainable student is a LoRA copy, seed 0;
- within each episode, the top two candidates by frozen `W_ref` are Yes and all
  others are No;
- fixed ID train/validation splits are reused deterministically;
- the starting recipe is bf16, LoRA rank 16, batch 1, accumulation 4, learning
  rate `1e-5`, two epochs, maximum length 1024, and about 500 steps;
- formal checkpoint candidates are steps 0, 100, 250, and 500; and
- selection is highest ID verbal AUC, ties resolved by the earliest step.

The canary invocation adds `--canary-steps 10` (the accepted range is 5–20).
Its `canary_manifest.json` must demonstrate finite loss and gradients,
checkpoint save/load integrity, adapter enable/disable behavior, workspace
evaluation, throughput, and memory health. A canary checkpoint is explicitly
ineligible for OOD.

The formal invocation omits `--canary-steps`. It must emit at least
`run_config.json`, `provenance.json`, `split_manifest.json`,
`teacher_labels.jsonl`, `truncation_stats.json`, `training_metrics.jsonl`,
`validation_metrics.jsonl`, checkpoints, `summary.json`, and
`lock_manifest.json`.

## ID lock and one-shot OOD

The trainer lock must state:

```json
{
  "selection_scope": "id_validation",
  "selection_metric": "verbal_auc",
  "tie_break": "earliest_step",
  "ood_evaluated": false,
  "checkpoint_path": "checkpoints/step-000250",
  "step": 250,
  "validation_auc": 0.0,
  "checkpoint_tree_sha256": "...",
  "split_manifest_sha256": "...",
  "run_config_sha256": "...",
  "validation_metrics_sha256": "..."
}
```

The launcher rejects symlinks and independently recomputes the checkpoint tree
hash. The canonical tree digest hashes sorted records
`relative_path + NUL + file_sha256 + newline`. It then writes a second,
campaign-owned `id_lock/lock_manifest.json`, binding the source lock, selected
checkpoint, ID-selection rule, and the only authorized OOD conditions.

Immediately before OOD, both the trainer lock and checkpoint tree are rehashed.
The launcher then creates `ood/attempt_started.json` with exclusive-create
semantics. Creation consumes the sole OOD attempt before evaluation begins. A
crash or nonzero OOD exit does not permit a retry in that run.

The OOD interface implemented by `src/analysis/evaluate_metacog_m1_ood.py` is:

```text
src/analysis/evaluate_metacog_m1_ood.py
  --lock-manifest CAMPAIGN_LOCK
  --attempt-id PRECOMMITTED_ID
  --baseline-decoupled M0_JSON
  --baseline-compositional M0_JSON
  --decoupled-battery BATTERY
  --compositional-battery BATTERY
  --bootstrap-samples 4000 --bootstrap-seed 0
  --out-json OOD_RESULT
```

It evaluates Decoupled and Compositional together in one invocation, loads the
locked adapter once, runs adapter-disabled/enabled full-context QA with identical
prompts and item order, and writes:

- the exact precommitted `attempt_id` and campaign-lock SHA-256;
- `bootstrap={"samples":4000,"unit":"episode_cluster"}`;
- exactly `conditions.decoupled` and `conditions.compositional`; and
- a `GREEN`, `AMBER`, or `RED` M1 decision.

The scientific payload must contain V/W before, after, and delta; within-episode
metrics; Yes rates; full-context QA; and paired episode-cluster bootstrap CIs.
The M1 Decoupled gate remains:

```text
GREEN: Delta V >= +0.15, V_after > 0.50,
       abs(Delta W) < 0.03, full-context QA drop <= 2pp
AMBER: +0.05 < Delta V < +0.15 with both no-harm checks passing
RED:   Delta V <= +0.05, workspace drop > 0.03,
       or full-context QA drop > 2pp
```

No automated AMBER retry exists. Any controlled diagnostic branch requires a
new human-reviewed plan and may change exactly one preregistered variable.

## Report and decision ledger

The report interface implemented by `src/analysis/report_metacog_m1.py` is:

```text
src/analysis/report_metacog_m1.py
  --m0-gate PATH --canary-manifest PATH --m1-summary PATH
  --lock-manifest PATH --ood-result PATH --decision-ledger PATH
  --out report/M1_GATE_REPORT.md
```

The launcher validates that its Markdown contains these headings or phrases:

1. M0 baseline reproduction status
2. Model/tokenizer revisions
3. A5000 memory/throughput configuration
4. Teacher-label construction audit
5. Training configuration
6. Loss/gradient health
7. ID checkpoint-selection table
8. Locked checkpoint
9. Decoupled V before/after
10. Decoupled W before/after
11. Compositional V/W before/after
12. Within-episode metrics
13. Yes-rate before/after
14. Full-context QA before/after
15. Bootstrap CIs
16. GREEN / AMBER / RED decision
17. Artifact paths and hashes
18. No H100 job was launched

`decision_ledger.jsonl` is append-only. It records the fully tokenized and
shell-escaped form of every command, cwd, pinned GPU, start/finish timestamps,
exit status, log path/hash, declared artifact hashes, every gate decision, lock
hashes, OOD attempt consumption, the report hash, and the final STOP. After the
last event, `decision_ledger.jsonl.sha256` seals the ledger without introducing
a circular self-hash.

On success, `STOP.json` states `manual_review_required`,
`h100_launched=false`, and `next_stage_launched=false`. On a gate stop or error,
the ledger likewise records a fail-closed stop and the process returns nonzero.

## Custom command plan for interface development

`--plan PATH` may supply another JSON plan, solely for synthetic contract tests while the
OOD/report CLIs are developed. It must retain exactly `m0`, `m0_gate`, `canary`,
`m1`, `ood`, and `report`; OOD is exactly one command. Each command is an argv
array and is executed without a shell. Declared outputs and all gate/manifest
paths must remain inside the fresh run directory. Plans mentioning H100,
memory-RL, RL-QA, a non-Qwen3-8B `--model`, or a nonzero `--seed` are rejected.

Execution of a custom plan additionally requires the conspicuous
`--unsafe-test-plan` flag. Scientific runs must use the built-in plan. Use
`--print-plan` to audit rendered JSON without enabling custom execution.

The launcher currently runs the graph in one uninterrupted invocation and does
not support in-place resume. A gate stop or pre-OOD failure requires a new,
fresh run directory. Once `ood/attempt_started.json` exists, that run's OOD
attempt is permanently consumed; do not delete the marker or rerun OOD. If OOD
completed and only report generation needs inspection, run the CPU-only report
command manually against the sealed artifacts instead of rerunning the
campaign.
