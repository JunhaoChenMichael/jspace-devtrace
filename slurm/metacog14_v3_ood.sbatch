#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=250G
#SBATCH -t 24:00:00
#SBATCH -J metacog14_v3_ood
#SBATCH -o logs/metacog14_v3_ood.out
#SBATCH -e logs/metacog14_v3_ood.err
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-14B"
export MODEL_REV=40c069824f4251a91eefaf281ebe4c544efd3e18
source slurm/a100_env.sh
RUN_DIR=$REPO/data/results/a100_next_boundary_campaign/qwen3_14b_metacog_v3/seed0
test -f "$RUN_DIR/id_lock/lock_manifest.json" || { echo "REFUSING: no ID lock" >&2; exit 2; }
python scripts/run_metacog_alignment_campaign.py \
  --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
  --seed 0 --run-dir "$RUN_DIR" --resume-ood
