#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=250G
#SBATCH -t 24:00:00
#SBATCH -J metacog32_id
#SBATCH -o logs/metacog32_id_seed0.out
#SBATCH -e logs/metacog32_id_seed0.err
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-32B"
export MODEL_REV=9216db5781bf21249d130ec9da846c4624c16137
source slurm/a100_env.sh
RUN_DIR=$REPO/data/results/metacog32_a100_seed0/seed0
echo "model=$METACOG_EXPECTED_MODEL rev=$MODEL_REV run_dir=$RUN_DIR"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader
# Seed 0 only. Phase 1 stops at the ID-only checkpoint lock with Decoupled and
# Compositional still sealed; the one-shot OOD is a separate, reviewed step.
# The M0 gate runs in scale_gap mode: a 32B model is not expected to reproduce
# the published 8B AUCs, only to show a repairable reporting gap.
python scripts/run_metacog_alignment_campaign.py \
  --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
  --seed 0 --run-dir "$RUN_DIR" --stop-after id_lock
