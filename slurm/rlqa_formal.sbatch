#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -J rlqa_formal
#SBATCH -o logs/rlqa_formal_seed%a.out
#SBATCH -e logs/rlqa_formal_seed%a.err
#SBATCH --array=0-2
source slurm/a100_env.sh
SEED=${SLURM_ARRAY_TASK_ID}
TEMP=$(python -c "import json;print(json.load(open('data/results/rlqa_a100/RECIPE_FREEZE.json'))['recipe']['temperature'])")
echo "seed=$SEED temperature=$TEMP"
# Locked Track A recipe: RL-QA only, lambda_QA=1, lambda_W=0, budget 2,
# group 8, LoRA r32, bf16, 300 steps, lr 1e-6, KL beta 0.03, two GRPO epochs.
# Split seed stays 0 for every optimisation seed. ID data only.
SEEDS=$SEED BETAS=0.03 BUDGETS=2 TEMPERATURE="$TEMP" GROUP_SIZE=8 \
  LAMBDA_QA=1 LAMBDA_W=0 DIAGNOSTIC_EVERY=25 \
  MODEL=Qwen/Qwen3-8B TEACHER_TAG=qwen3-8B \
  OUT_ROOT=data/results/rlqa_a100/runs \
  bash scripts/run_memory_rl_mvp.sh formal rl-qa
echo "RLQA_FORMAL_SEED${SEED}_OK"
