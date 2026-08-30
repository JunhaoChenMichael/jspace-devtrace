#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -J rlqa_b0t5
#SBATCH -o logs/rlqa_b0_preflight_t5.out
#SBATCH -e logs/rlqa_b0_preflight_t5.err
source slurm/a100_env.sh
SPLIT_SHA=5fc4d99e708744ebf4a872c86287c337d25a32c8725c5c1656740cb3a00cfb2a
OUT=data/results/rlqa_a100/b0_s0_k2_g16_t5
# Stage B0: no-training exact-budget reward/diversity preflight on the 175
# sealed ID-train episodes.  Temperature is chosen here, from ID-train only --
# never from Decoupled or Compositional.
python src/experiments/preflight_qa_reward.py \
  --model Qwen/Qwen3-8B --teacher-tag qwen3-8B \
  --workspace-teacher-model Qwen/Qwen3-8B \
  --expected-model-revision "$MODEL_REV" \
  --expected-split-sha256 "$SPLIT_SHA" \
  --seed 0 --split-seed 0 --budget 2 --group-size 16 \
  --temperature-candidates 5 \
  --lora-rank 32 --max-length 2048 --answer-tokens 64 \
  --device cuda --dtype bfloat16 \
  --out-dir "$OUT"
python src/analysis/validate_qa_reward_preflight.py "$OUT" \
  --expected-model-revision "$MODEL_REV" \
  --expected-manifest-sha256 "$SPLIT_SHA"
echo "B0_OK"
