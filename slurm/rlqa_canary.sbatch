#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -J rlqa_canary
#SBATCH -o logs/rlqa_canary.out
#SBATCH -e logs/rlqa_canary.err
source slurm/a100_env.sh
# Engineering canary for the frozen Qwen3-8B RL-QA recipe: same recipe as the
# formal seeds, 10 optimiser steps, ID data only.
TEMP=$(python -c "import json;print(json.load(open('data/results/rlqa_a100/RECIPE_FREEZE.json'))['recipe']['temperature'])")
echo "frozen sampling temperature from Stage B0 (ID-train only): $TEMP"
SEEDS=0 BETAS=0.03 BUDGETS=2 TEMPERATURE="$TEMP" GROUP_SIZE=8 \
  LAMBDA_QA=1 LAMBDA_W=0 MAX_STEPS=10 EVAL_EVERY=5 SAVE_EVERY=0 \
  MODEL=Qwen/Qwen3-8B TEACHER_TAG=qwen3-8B \
  OUT_ROOT=data/results/rlqa_a100/canary_t5 \
  bash scripts/run_memory_rl_mvp.sh formal rl-qa
echo "RLQA_CANARY_OK"
