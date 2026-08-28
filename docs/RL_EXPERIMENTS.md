# Workspace-Guided RL: MVP Experiment Guide

This extension trains a memory-admission policy without changing the paper's
workspace readout, PRG, causal intervention, or metacognitive-alignment claims.
The policy always scores candidates before seeing the future probe question.

The implementation entry points are:

- `src/experiments/train_memory_rl.py`: matched Workspace-SFT and Stages A/B/C.
- `src/experiments/preflight_qa_reward.py`: no-training Stage-B0 exact-budget
  reward-diversity audit on the sealed ID training split.
- `src/experiments/evaluate_memory_rl.py`: unified original/workspace/adapter
  evaluation on held-out batteries.
- `src/analysis/validate_qa_reward_preflight.py`: strict B0 artifact and
  recomputation validator.
- `scripts/run_memory_rl_mvp.sh`: one-stage-at-a-time serial runner.

The runner never promotes a run to the next stage. After each stage, stop,
inspect its metrics, make the gate decision, and explicitly start a new command.

## 1. Conditions and rewards

### Matched baseline: `sft-w`

The primary matched Workspace-SFT uses the same rank-continuous teacher as
RL-W: `r > 0.5` is the Yes target and each cross-entropy term is weighted by
`|2r-1|`. It must use the same training episodes, split manifest, LoRA capacity,
prompt, and evaluation harness as RL-W. The legacy top-2 baseline remains
available with `WORKSPACE_OBJECTIVE=top-k`; compare it only with an RL-W run
using that same setting. Existing 0.5B/1B metacognitive checkpoints are not a
matched Qwen2.5-7B baseline; train this condition for the 7B comparison.

### Stage A: `rl-w`

The constrained action is Yes/No. For candidate workspace percentile `r`, the
reward is `2r-1` for Yes and `1-2r` for No. The workspace values are loaded from
the original checkpoint's immutable result JSON, so the trainable adapter cannot
raise its reward by changing its own hidden states.

Use centered advantages for this binary reward. Per-group z-scoring removes the
`|2r-1|` margin and reduces the continuous reward to a threshold label; the
trainer therefore defaults to `center` in `rl-w` mode.

### Stage B: `rl-qa`

The policy scores every candidate from `(context, candidate)` only, samples an
exact-budget set with Gumbel top-k / Plackett-Luce log probabilities, and receives
binary QA reward from a frozen, adapter-disabled recall model. The probe and gold
answer are available only inside the recall evaluator.

### Stage C: `rl-hybrid`

The policy-gradient reward is normalized by the non-zero coefficient sum so
changing a ratio does not silently change the reward/KL scale:

```text
(lambda_qa * QA_reward + lambda_w * workspace_set_reward)
  / (lambda_qa + lambda_w)

loss = GRPO_loss + beta * KL
```

Start with `lambda_qa=1`, `lambda_w=0.5`, and mean percentile workspace reward.
Use the contrastive workspace set reward only as a declared ablation.

## 2. Data contract and leakage prevention

Training is restricted in code to these sources:

| Source | Battery | Frozen Qwen-7B teacher rows |
|---|---|---|
| Explicit | `data/benchmarks/battery_v1_final.json` | `data/results/results_v1f_7B-Instruct.json` |
| Evoked | `data/benchmarks/battery_v2_final.json` | `data/results/results_v2f_7B-Instruct.json` |
| Evoked-G2 | `data/benchmarks/battery_v2_g2.json` | `data/results/results_v2g2_7B-Instruct.json` |

The released training pool contains 220 episodes and 1,040 candidates. Each
source is split independently at the episode level. With `seed=0` and the
default 20% holdout, this yields 175 train and 45 validation episodes. Every run
writes `split_manifest.json`, including battery/result SHA-256 hashes and the
exact source-qualified episode IDs.

`SPLIT_SEED=0` is held fixed across every condition and optimization seed.
`SEEDS=0,1,2` changes initialization/sampling only; changing the held-out split
between seeds would confound method variance with different training examples.

Never use the following for training, early stopping, coefficient selection, or
choosing a checkpoint:

| Held-out role | Battery | Existing reference rows |
|---|---|---|
| Compositional | `battery_v3d.json` | `results_v3f_7B-Instruct.json` |
| Decoupled | `battery_v4_final.json` | `results_v4f_7B-Instruct.json` |
| Decoupled-L | `battery_v4_xl.json` | `results_v4xl_7B-Instruct.json` |
| Confusable absent | `battery_v4_relabs.json` | `results_v4ra_7B-Instruct.json` |
| Scaled confusable absent | `battery_v4xl_relabs.json` | `results_v4xlra_7B-Instruct.json` |
| LoCoMo-wrapped Decoupled | `battery_v4_locomo_wrapped.json` | `results_v4wrap_7B-Instruct.json` |

Decoupled-L contains 58 exact Decoupled episodes. The absent-foil and LoCoMo
sets are also derived from Decoupled episodes. Treat them as paired stress tests,
not independent samples that can be pooled to inflate `n`.

For either LoCoMo-wrapped battery set evaluator `--max-length 2048` (and reduce
`--batch-size` if needed); those contexts reach roughly 1,230 Qwen tokens, while
the quick evaluator default is 512.

Hyperparameters, early stopping, and checkpoint selection use only the 45
in-distribution validation episodes. Run OOD evaluation once after a stage's
configuration is locked. Looking at Decoupled after every sweep silently turns it
into a validation set.

## 3. Environment check

Activate the project environment, then verify the packages used by the trainer:

```bash
python -m pip install -r requirements.txt

python - <<'PY'
import torch, transformers, peft, numpy, sklearn
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
PY
```

The implementation is custom PyTorch GRPO-style optimization; TRL is not
required. Set a GPU explicitly when sharing a server:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Run names are deterministic, and both the runner and trainer refuse to overwrite
an existing run directory. Change `OUT_ROOT` for a fresh campaign.

## 4. Dry-run and 3B smoke test

Dry-run validates source restrictions, battery/result joins, fingerprints, and
the split without loading a model:

```bash
bash scripts/run_memory_rl_mvp.sh dry-run sft-w
bash scripts/run_memory_rl_mvp.sh dry-run rl-w
bash scripts/run_memory_rl_mvp.sh dry-run rl-qa
bash scripts/run_memory_rl_mvp.sh dry-run rl-hybrid
```

Start the GPU path with Qwen2.5-3B and a small episode/step limit. The repository
does not contain 3B workspace measurements, so this profile deliberately reuses
the frozen 7B rows and writes `teacher_mismatch_override=true`; it validates
software wiring only and is not a scientific result:

```bash
SEEDS=0 BETAS=0 BUDGETS=2 \
  bash scripts/run_memory_rl_mvp.sh smoke sft-w

SEEDS=0 BETAS=0.03 BUDGETS=2 \
  bash scripts/run_memory_rl_mvp.sh smoke rl-w
```

Inspect both `summary.json` and `metrics.jsonl`. Only after parsing, reward
variance, KL, validation AUC, and Yes-rate look sane should Stage B be launched:

```bash
SEEDS=0 BETAS=0.03 BUDGETS=2 \
  bash scripts/run_memory_rl_mvp.sh smoke rl-qa
```

Only after Stage B produces non-constant QA reward should Stage C be launched:

```bash
SEEDS=0 BETAS=0.03 BUDGETS=2 LAMBDA_QA=1 LAMBDA_W=0.5 \
  bash scripts/run_memory_rl_mvp.sh smoke rl-hybrid
```

The smoke profile defaults to 10 steps, 8 training episodes, 4 validation
episodes, group size 4, and Qwen2.5-3B-Instruct. Override these with
`MAX_STEPS`, `LIMIT_TRAIN_EPISODES`, `LIMIT_VALIDATION_EPISODES`, and
`GROUP_SIZE`.

## 5. Stage B0 QA reward preflight

Before training RL-QA, run the no-training preflight on all 175 sealed ID-train
episodes. It samples 16 exact-budget sets per episode from a fresh, zero-init
rank-32 LoRA policy, then evaluates the already-detached sets with one greedy,
adapter-disabled frozen recall model. Admission prompts contain only context and
candidate concept; the script fails closed if a probe appears in a rendered
policy prompt.

Temperature is selected without QA or OOD feedback: using common per-episode
random streams, choose the lowest of `0.7,1,2,3,5` whose median number of unique
sets in 16 draws is at least four. If none passes, use the highest candidate.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
python src/experiments/preflight_qa_reward.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --expected-model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --expected-split-sha256 1988c8af1fc39884a7bd711f335f91db6c890612e6e0a45917488ba4b44abf0f \
  --seed 0 --split-seed 0 --budget 2 --group-size 16 \
  --temperature-candidates 0.7,1,2,3,5 \
  --lora-rank 32 --max-length 2048 --answer-tokens 64 \
  --device cuda --dtype bfloat16 \
  --out-dir data/results/memory_rl_b0_s0_k2_g16

python src/analysis/validate_qa_reward_preflight.py \
  data/results/memory_rl_b0_s0_k2_g16 \
  --expected-model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --expected-manifest-sha256 1988c8af1fc39884a7bd711f335f91db6c890612e6e0a45917488ba4b44abf0f
```

The run writes `samples.jsonl` (2,800 logical draws), `groups.jsonl` (175
episode groups), `references.jsonl`, `temperature_calibration.json`, the sealed
split/config/dropout artifacts, and `summary.json`. Duplicate sampled sets share
one deterministic recall generation but remain separate logical draws.

Gate B0 is GREEN only when at least 40% of G=16 groups have mixed binary QA
reward and the median number of unique sets is at least four. Below 20% mixed is
RED; every remaining case is AMBER. Because Stage B1 trains with G=8, the report
also includes two fixed G=8 halves per episode as a sensitivity analysis; this
does not replace the preregistered G=16 gate.

## 6. Manual decision gates

### Gate A

Compare RL-W directly with matched SFT-W.

- If RL-W exceeds SFT-W by at least 0.03 Decoupled verbal AUC with no material
  no-harm regression, lock the recipe and run three seeds.
- If the difference is below 0.03, treat them as tied and proceed to QA reward;
  do not spend a large grid on pure teacher imitation.
- If RL-W is worse by more than 0.03, inspect reward variance, Yes/No collapse,
  KL, and parsing. Permit at most two controlled ID-validation sweeps.

### Gate B

- Proceed when RL-QA gains at least five QA percentage points over the original
  reporter and preferably over SFT-W.
- If containment rises but QA does not, inspect oracle-memory QA and report the
  exploitable subset instead of tuning admission against a recall ceiling.
- If group reward standard deviation is below 0.1 in most batches, stop sparse
  QA-only tuning and test the hybrid reward.

### Gate C

The strongest result is hybrid RL beating both RL-W and RL-QA. A second useful
result is hybrid RL matching SFT-W AUC while improving downstream QA. If RL-QA
beats hybrid, lower `lambda_w` to 0.05, 0.1, or 0.25 and interpret workspace as
a shaping prior rather than a target.

No gate is encoded as an automatic branch in the runner.

## 7. Formal Qwen2.5-7B runs

Use seed 0 for ID-validation tuning. Run one command, inspect it, then decide
whether to run the next value; do not submit a large Cartesian grid.

Matched SFT baseline:

```bash
SEEDS=0 BETAS=0 BUDGETS=2 \
  bash scripts/run_memory_rl_mvp.sh formal sft-w
```

Stage A beta checks, one at a time:

```bash
SEEDS=0 BETAS=0.01 BUDGETS=2 bash scripts/run_memory_rl_mvp.sh formal rl-w
SEEDS=0 BETAS=0.03 BUDGETS=2 bash scripts/run_memory_rl_mvp.sh formal rl-w
SEEDS=0 BETAS=0.1  BUDGETS=2 bash scripts/run_memory_rl_mvp.sh formal rl-w
```

After selecting beta on ID validation, run the locked Stage A condition across
three seeds. For example:

```bash
SEEDS=0,1,2 BETAS=0.03 BUDGETS=2 \
  bash scripts/run_memory_rl_mvp.sh formal rl-w
```

Stage B development and, if Gate B passes, final seeds/budgets:

```bash
SEEDS=0 BETAS=0.03 BUDGETS=2 TEMPERATURE=5 GROUP_SIZE=8 \
  LAMBDA_QA=1 LAMBDA_W=0 DIAGNOSTIC_EVERY=25 \
  bash scripts/run_memory_rl_mvp.sh formal rl-qa

SEEDS=0,1,2 BETAS=0.03 BUDGETS=2,3 \
  bash scripts/run_memory_rl_mvp.sh formal rl-qa
```

Stage C coefficient checks must also be run and inspected one at a time:

```bash
SEEDS=0 BETAS=0.03 BUDGETS=2 TEMPERATURE=5 GROUP_SIZE=8 LAMBDA_QA=1 LAMBDA_W=0.5 \
  bash scripts/run_memory_rl_mvp.sh formal rl-hybrid

SEEDS=0 BETAS=0.03 BUDGETS=2 TEMPERATURE=5 GROUP_SIZE=8 LAMBDA_QA=1 LAMBDA_W=0.25 \
  bash scripts/run_memory_rl_mvp.sh formal rl-hybrid

SEEDS=0 BETAS=0.03 BUDGETS=2 TEMPERATURE=5 GROUP_SIZE=8 LAMBDA_QA=1 LAMBDA_W=1 \
  bash scripts/run_memory_rl_mvp.sh formal rl-hybrid
```

Every selector validation also records candidate-level reporter diagnostics on
the fixed ID split. `V_RL` is the adapter-enabled constrained Yes probability
at temperature 1, `W_ref` is the immutable raw reference-model W_rr, and
`y_utility` is `1` exactly for a load-bearing candidate. Pearson and tie-aware
Spearman correlations are reported both pooled and as an equal-episode mean;
pooled intervals use 4,000 source-stratified episode-cluster bootstrap draws. Source-specific
point estimates are retained because Explicit, Evoked, and Evoked-G2 can have
different score scales. The candidate rows and summaries are written to
`reporter_correlations.jsonl`, while the summaries are duplicated in the
corresponding `metrics.jsonl` validation records.

These correlations are mechanistic diagnostics only. They must not enter
checkpoint or Hybrid-coefficient selection, which remains ID QA first and
workspace set reward only as the predeclared tie-breaker.

After the coefficient and each method's checkpoint are locked, use the
ID-only scorer to compare Original, SFT-W, RL-QA, and Hybrid under an identical
bf16 batch-size-1 forward protocol:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python src/analysis/evaluate_id_reporter_alignment.py \
  --run sft-w=PATH_TO_SFT_RUN \
  --run rl-qa=PATH_TO_RL_QA_RUN \
  --run hybrid-lw0.5=PATH_TO_HYBRID_0P5_RUN \
  --run hybrid-lw0.25=PATH_TO_HYBRID_0P25_RUN \
  --run hybrid-lw1.0=PATH_TO_HYBRID_1P0_RUN \
  --batch-size 1 --bootstrap-samples 4000 --bootstrap-seed 0 \
  --out PATH_TO_NEW_ID_REPORTER_JSON
```

The scorer accepts only the three predeclared training sources, rebuilds and
checks the sealed 45-episode/215-candidate split, verifies model revision,
adapter activation, candidate/prompt/token order, and refuses to overwrite its
output. It reports paired correlation differences from shared
source-stratified episode-cluster draws. The Original condition is freshly
forwarded through the base model rather than copied from historical `V` rows.

Then run only the locked hybrid configuration across the paper seeds and both
budgets:

```bash
SEEDS=0,1,2 BETAS=0.03 BUDGETS=2,3 LAMBDA_QA=1 LAMBDA_W=0.5 \
  bash scripts/run_memory_rl_mvp.sh formal rl-hybrid
```

Useful overrides are:

```text
MAX_STEPS=300|600|1200
LEARNING_RATE=5e-7|1e-6|2e-6
LORA_RANK=32|64
GROUP_SIZE=8
TEMPERATURE=0.7
ANSWER_TOKENS=64
WORKSPACE_SET_REWARD=mean|contrastive
WORKSPACE_OBJECTIVE=rank-continuous|top-k
```

The default rank-continuous setting and `center` advantage mode are deliberate
deviations from literal per-group z-score GRPO: with only two actions, z-scoring
cancels `|2r-1|`. `run_config.json` records both the requested and effective
normalization. KL is computed separately and cannot be cancelled by centering.

The trainer also binds `teacher-tag=7B-Instruct` to
`Qwen/Qwen2.5-7B-Instruct`; a primary run fails if the W teacher and policy/KL
reference differ. `--allow-teacher-mismatch` exists only for smoke tests and
declared cross-model ablations.

## 8. Locked OOD evaluation

The evaluator accepts repeatable battery specs and adapter names. Point each
adapter at the selected `best-step-*` directory recorded in its
`best_checkpoint.json`.

First measure AUC/containment without QA on Decoupled and Compositional:

```bash
python src/experiments/evaluate_memory_rl.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --spec decoupled=data/results/results_v4f_7B-Instruct.json::data/benchmarks/battery_v4_final.json \
  --spec compositional=data/results/results_v3f_7B-Instruct.json::data/benchmarks/battery_v3d.json \
  --adapter sft-w=/path/to/sft/best-step-N \
  --adapter rl-w=/path/to/rl-w/best-step-N \
  --adapter rl-qa=/path/to/rl-qa/best-step-N \
  --adapter rl-hybrid=/path/to/hybrid/best-step-N \
  --budgets 2,3 \
  --skip-qa \
  --skip-no-harm \
  --out data/results/memory_rl_ood_auc.json
```

Run downstream QA on Decoupled/Decoupled-L with a 64-token answer budget:

```bash
python src/experiments/evaluate_memory_rl.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --spec decoupled=data/results/results_v4f_7B-Instruct.json::data/benchmarks/battery_v4_final.json \
  --spec decoupled_l=data/results/results_v4xl_7B-Instruct.json::data/benchmarks/battery_v4_xl.json \
  --adapter sft-w=/path/to/sft/best-step-N \
  --adapter rl-qa=/path/to/rl-qa/best-step-N \
  --adapter rl-hybrid=/path/to/hybrid/best-step-N \
  --budgets 2,3 \
  --max-new-tokens 64 \
  --out data/results/memory_rl_ood_qa.json
```

The evaluator runs a 4,000-draw episode-cluster bootstrap by default, including
paired AUC-difference intervals. Its full QA command also reports a separate
adapter-enabled full-context check; selection QA always uses the same
adapter-disabled frozen base recall model.

Measure whether the adapter damaged the latent workspace itself, using the same
battery before and after adaptation:

```bash
python src/experiments/measure.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapter /path/to/locked/best-step-N \
  --battery data/benchmarks/battery_v4_final.json \
  --out data/results/noharm_v4_locked_adapter.json \
  --dtype bfloat16

python src/analysis/analyze.py \
  data/results/results_v4f_7B-Instruct.json \
  data/results/noharm_v4_locked_adapter.json
```

Run general capability checks with the same base and paired LoRA adapter:

```bash
python src/experiments/general_eval.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tag original --out data/results/noharm_general_original.json

python src/experiments/general_eval.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapter /path/to/locked/best-step-N \
  --tag locked-adapter --out data/results/noharm_general_adapter.json
```

Finally run the confusable-absent stress test. Main-paper comparisons require
episode-cluster AUC bootstrap (`B=4000`), exact McNemar for paired QA, and
mean/std plus individual values across three seeds.

When the unified evaluator contains seed-qualified adapter names, summarize the
predeclared gates without silently dropping a missing seed:

```bash
python src/analysis/memory_rl_gates.py \
  data/results/memory_rl_ood_qa.json \
  --sft 'sft-w-s*' \
  --rl-w 'rl-w-s*' \
  --rl-qa 'rl-qa-s*' \
  --hybrid 'rl-hybrid-s*' \
  --source decoupled --budget 2 \
  --out data/results/memory_rl_gate_summary.json
```

The script reports individual values, mean, sample standard deviation, Gate
A/B/C status, and adapter-minus-base full-context deltas. “No major capability
degradation” remains a manual scientific judgment because the README does not
predeclare a universal acceptable threshold across capability suites.

## 9. Run artifacts and diagnostics

Each training directory contains:

- `run_config.json`: complete CLI, source paths, package versions, resolved model
  commit, parameter counts, teacher identity, and effective normalization.
- `split_manifest.json`: immutable input hashes and split IDs.
- `dropout_audit.json` (RL runs): stochastic-dropout fields disabled for
  on-policy likelihood ratios.
- `metrics.jsonl`: baseline, training, and ID-validation metrics.
- Selector training rows additionally record mixed QA/containment flags, exact
  set diversity, exact set-policy entropy, containment, and the full candidate
  Yes-probability vector. Fixed 25-step windows are recomputed in `summary.json`
  and printed to the run log.
- `rollouts.jsonl`: sampled actions/sets, reward components, KL, containment,
  QA result, oracle/full-context references, and failure type.
- `best_checkpoint.json` and `best-step-*`: ID-selected adapter.
- `final_adapter`: last-step adapter, not automatically the selected model.
- `summary.json`: step count, best validation metric, elapsed time, diagnostics.

Before accepting a run, check reward trend, reward standard deviation, KL scale,
Yes-rate, containment, oracle recall ceiling, and whether failures are selection
or recall/composition failures. Never select the final checkpoint from OOD
metrics.
