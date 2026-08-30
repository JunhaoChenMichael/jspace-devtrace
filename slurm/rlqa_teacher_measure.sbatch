#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 4:00:00
#SBATCH -J rlqa_teacher
#SBATCH -o logs/rlqa_teacher_measure.out
#SBATCH -e logs/rlqa_teacher_measure.err
source slurm/a100_env.sh
# Track A ID-only W_ref teacher rows for Qwen3-8B: Explicit, Evoked, Evoked-G2.
# Same frozen model, revision, dtype and readout flags as the metacognitive M0
# stage, but written to Track-A-owned filenames so the two tracks stay disjoint.
set -x
for spec in "battery_v1_final:results_v1f_qwen3-8B" \
            "battery_v2_final:results_v2f_qwen3-8B" \
            "battery_v2_g2:results_v2g2_qwen3-8B"; do
  BATTERY=${spec%%:*}; OUT=${spec##*:}
  if [[ -e data/results/${OUT}.json ]]; then echo "exists, skipping ${OUT}"; continue; fi
  python src/experiments/measure.py \
    --model Qwen/Qwen3-8B --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
    --dtype bfloat16 --device cuda --end-only --no-verbal-raw \
    --battery data/benchmarks/${BATTERY}.json \
    --out data/results/${OUT}.json
done
echo "TEACHER_MEASURE_OK"
