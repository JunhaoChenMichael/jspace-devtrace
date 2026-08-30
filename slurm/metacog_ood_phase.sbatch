#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -J metacog_ood
#SBATCH -o logs/metacog_ood_seed%a.out
#SBATCH -e logs/metacog_ood_seed%a.err
#SBATCH --array=0-2
source slurm/a100_env.sh
SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$REPO/data/results/metacog_a100_3seed/seed${SEED}
# Phase 2: the single, one-shot Decoupled + Compositional attempt for this seed.
# Submitted only after ALL THREE seeds hold an ID-only checkpoint lock.
for s in 0 1 2; do
  test -f "$REPO/data/results/metacog_a100_3seed/seed${s}/id_lock/lock_manifest.json" || {
    echo "REFUSING: seed ${s} has no ID lock; OOD stays sealed" >&2; exit 2; }
done
python scripts/run_metacog_alignment_campaign.py \
  --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
  --seed "$SEED" --run-dir "$RUN_DIR" --resume-ood
