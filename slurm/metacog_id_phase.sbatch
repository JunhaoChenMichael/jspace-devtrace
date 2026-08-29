#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -J metacog_id
#SBATCH -o logs/metacog_id_seed%a.out
#SBATCH -e logs/metacog_id_seed%a.err
#SBATCH --array=0-2
source slurm/a100_env.sh
SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$REPO/data/results/metacog_a100_3seed/seed${SEED}
echo "seed=$SEED run_dir=$RUN_DIR CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
# Phase 1 only: M0 -> gate -> frozen teacher -> canary -> M1 -> ID checkpoint lock.
# Decoupled/Compositional stay sealed; the OOD phase is a separate submission
# that runs only after all three seeds are locked.
python scripts/run_metacog_alignment_campaign.py \
  --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
  --seed "$SEED" --run-dir "$RUN_DIR" --stop-after id_lock
