#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=250G
#SBATCH -t 24:00:00
#SBATCH -J metacog32_v3_ood
#SBATCH -o logs/metacog32_v3_ood.out
#SBATCH -e logs/metacog32_v3_ood.err
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-32B"
export MODEL_REV=9216db5781bf21249d130ec9da846c4624c16137
source slurm/a100_env.sh
RUN_DIR=$REPO/data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0
test -f "$RUN_DIR/id_lock/lock_manifest.json" || { echo "REFUSING: no ID lock" >&2; exit 2; }
python scripts/run_metacog_alignment_campaign.py \
  --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
  --seed 0 --run-dir "$RUN_DIR" --resume-ood
