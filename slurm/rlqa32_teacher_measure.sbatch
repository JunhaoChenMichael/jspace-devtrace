#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=200G
#SBATCH -t 8:00:00
#SBATCH -J rlqa32_teacher
#SBATCH -o logs/rlqa32_teacher_measure.out
#SBATCH -e logs/rlqa32_teacher_measure.err
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-32B"
export MODEL_REV=9216db5781bf21249d130ec9da846c4624c16137
source slurm/a100_env.sh
# Track A (32B) ID-only W_ref teacher rows: Explicit, Evoked, Evoked-G2.
# Same readout flags as 8B, byte-for-byte, so the scale comparison is clean.
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
set -x
for spec in "battery_v1_final:results_v1f_qwen3-32B" \
            "battery_v2_final:results_v2f_qwen3-32B" \
            "battery_v2_g2:results_v2g2_qwen3-32B"; do
  BATTERY=${spec%%:*}; OUT=${spec##*:}
  if [[ -e data/results/${OUT}.json ]]; then echo "exists, skipping ${OUT}"; continue; fi
  python src/experiments/measure.py \
    --model Qwen/Qwen3-32B --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
    --dtype bfloat16 --device cuda --end-only --no-verbal-raw \
    --battery data/benchmarks/${BATTERY}.json \
    --out data/results/${OUT}.json
done
set +x
python -c "
import torch;print('peak allocated GiB:', round(torch.cuda.max_memory_allocated()/2**30,2) if torch.cuda.is_available() else 'n/a')"
echo "TEACHER32_OK"
