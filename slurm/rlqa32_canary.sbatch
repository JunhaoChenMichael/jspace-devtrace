#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=250G
#SBATCH -t 8:00:00
#SBATCH -J rlqa32_canary
#SBATCH -o logs/rlqa32_canary.out
#SBATCH -e logs/rlqa32_canary.err
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-32B"
export MODEL_REV=9216db5781bf21249d130ec9da846c4624c16137
source slurm/a100_env.sh
TEMP=$(python -c "import json;print(json.load(open('data/results/rlqa32_a100/RECIPE_FREEZE.json'))['recipe']['temperature'])")
echo "frozen temperature: $TEMP"
SEEDS=0 BETAS=0.03 BUDGETS=2 TEMPERATURE="$TEMP" GROUP_SIZE=8 \
  LAMBDA_QA=1 LAMBDA_W=0 MAX_STEPS=10 EVAL_EVERY=5 SAVE_EVERY=0 \
  MODEL=Qwen/Qwen3-32B TEACHER_TAG=qwen3-32B \
  OUT_ROOT=data/results/rlqa32_a100/canary \
  bash scripts/run_memory_rl_mvp.sh formal rl-qa
python - <<'PY'
import torch
print("PEAK_ALLOCATED_GIB", round(torch.cuda.max_memory_allocated()/2**30, 2) if torch.cuda.is_available() else "n/a")
PY
echo "RLQA32_CANARY_OK"
