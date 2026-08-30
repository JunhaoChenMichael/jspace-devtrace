#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=250G
#SBATCH -t 48:00:00
#SBATCH -J rlqa32_formal12
#SBATCH -o logs/rlqa32_formal_seed%a.out
#SBATCH -e logs/rlqa32_formal_seed%a.err
#SBATCH --array=1-2
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-32B"
export MODEL_REV=9216db5781bf21249d130ec9da846c4624c16137
source slurm/a100_env.sh
TEMP=$(python -c "import json;print(json.load(open('data/results/rlqa32_a100/RECIPE_FREEZE.json'))['recipe']['temperature'])")
# Seed 0 ONLY. The plan forbids automatic expansion to seeds 1/2.
# checkpoint/eval steps 0/100/200/300, selection = strict first maximum of ID QA.
SEEDS=$SLURM_ARRAY_TASK_ID BETAS=0.03 BUDGETS=2 TEMPERATURE="$TEMP" GROUP_SIZE=8 \
  LAMBDA_QA=1 LAMBDA_W=0 DIAGNOSTIC_EVERY=25 \
  EVAL_EVERY=100 SAVE_EVERY=100 \
  MODEL=Qwen/Qwen3-32B TEACHER_TAG=qwen3-32B \
  OUT_ROOT=data/results/rlqa32_a100/runs \
  bash scripts/run_memory_rl_mvp.sh formal rl-qa
echo "RLQA32_FORMAL_SEED${SLURM_ARRAY_TASK_ID}_OK"
