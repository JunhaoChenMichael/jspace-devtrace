#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -J metacog_id0
#SBATCH -o logs/metacog_id_seed0.out
#SBATCH -e logs/metacog_id_seed0.err
source slurm/a100_env.sh
SEED=0
RUN_DIR=$REPO/data/results/metacog_a100_3seed/seed${SEED}
echo "seed=$SEED run_dir=$RUN_DIR CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv,noheader
python scripts/run_metacog_alignment_campaign.py \
  --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
  --seed "$SEED" --run-dir "$RUN_DIR" --stop-after id_lock
