#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=250G
#SBATCH -t 48:00:00
#SBATCH -J metacog32_v3
#SBATCH -o logs/metacog32_v3_seed0.out
#SBATCH -e logs/metacog32_v3_seed0.err
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-32B"
export MODEL_REV=9216db5781bf21249d130ec9da846c4624c16137
source slurm/a100_env.sh
RUN_DIR=$REPO/data/results/a100_next_boundary_campaign/qwen3_32b_metacog_v3/seed0
echo "corrected (v3) 32B campaign -> $RUN_DIR"
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv,noheader
# M0 is re-measured under the corrected verbal score. The scale_gap mode gates
# on W - V >= 0.10; with the corrupted V this returned SCALE_BOUNDARY, and the
# corrected measurement is expected to pass. Stops at the ID lock either way.
python scripts/run_metacog_alignment_campaign.py \
  --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
  --seed 0 --run-dir "$RUN_DIR" --stop-after id_lock
