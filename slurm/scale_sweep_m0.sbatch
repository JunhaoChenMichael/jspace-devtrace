#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=250G
#SBATCH -t 12:00:00
#SBATCH -J scale_sweep
#SBATCH -o logs/scale_sweep_%a.out
#SBATCH -e logs/scale_sweep_%a.err
#SBATCH --array=0-4
source slurm/a100_env.sh
MODELS=(Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
REV=$(cat "$HF_HOME/hub/models--Qwen--$M/refs/main")
OUT=data/results/scale_sweep/$M
mkdir -p "$OUT"
echo "model=Qwen/$M revision=$REV"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Measurement only: no training, no adapters, no OOD-driven decisions.
# V_raw is left ENABLED here (unlike the campaigns) so the template-free
# readout separates instruct/chat-template behaviour from base capability.
for spec in "explicit:battery_v1_final" "evoked:battery_v2_final" \
            "decoupled:battery_v4_final" "compositional:battery_v3d"; do
  COND=${spec%%:*}; BAT=${spec##*:}
  [ -e "$OUT/$COND.json" ] && { echo "exists, skipping $COND"; continue; }
  python src/experiments/measure.py \
    --model "Qwen/$M" --model-revision "$REV" --tokenizer-revision "$REV" \
    --dtype bfloat16 --device cuda --end-only \
    --battery "data/benchmarks/$BAT.json" --out "$OUT/$COND.json"
done
echo "SWEEP_OK $M"
