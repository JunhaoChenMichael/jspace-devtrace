#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=250G
#SBATCH -t 24:00:00
#SBATCH -J rlqa32_b0
#SBATCH -o logs/rlqa32_b0_preflight.out
#SBATCH -e logs/rlqa32_b0_preflight.err
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-32B"
export MODEL_REV=9216db5781bf21249d130ec9da846c4624c16137
source slurm/a100_env.sh
SPLIT_SHA=d09d6dbdd4fe5d159de9d120963b6229dcffe337de64364cc03192f4198c6035
OUT=data/results/rlqa32_a100/preflight/b0_s0_k2_g16_t5
# Stage B0 for the 32B scaling gate. Temperature 5.0 is the inherited 8B value
# and the ONLY candidate: the plan says keep 5.0 if it passes the gates, and
# forbids searching. Recall runs at batch size 1 as the plan requires.
python src/experiments/preflight_qa_reward.py \
  --model Qwen/Qwen3-32B --teacher-tag qwen3-32B \
  --workspace-teacher-model Qwen/Qwen3-32B \
  --expected-model-revision "$MODEL_REV" \
  --expected-split-sha256 "$SPLIT_SHA" \
  --seed 0 --split-seed 0 --budget 2 --group-size 16 \
  --temperature-candidates 5 \
  --lora-rank 32 --max-length 2048 --answer-tokens 64 \
  --recall-batch-size 1 \
  --device cuda --dtype bfloat16 \
  --out-dir "$OUT"
python src/analysis/validate_qa_reward_preflight.py "$OUT" \
  --expected-model-revision "$MODEL_REV" \
  --expected-manifest-sha256 "$SPLIT_SHA"
echo "B0_32B_OK"
