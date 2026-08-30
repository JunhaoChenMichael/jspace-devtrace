#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=250G
#SBATCH -t 6:00:00
#SBATCH -J sweep_moe3
#SBATCH -o logs/scale_sweep_v3_moe.out
#SBATCH -e logs/scale_sweep_v3_moe.err
source slurm/a100_env.sh
M=Qwen3-30B-A3B
REV=$(cat "$HF_HOME/hub/models--Qwen--$M/refs/main")
OUT=data/results/scale_sweep_v3_diagnostic/$M
mkdir -p "$OUT"
# DIAGNOSTIC ONLY, and deliberately outside both campaigns: the plans forbid
# substituting an MoE for the dense 32B scale point, and this does not do that.
# 30B total / 3B active parameters separates "does the transition track total
# parameters or active compute?" -- a question the dense sweep cannot answer.
echo "model=Qwen/$M revision=$REV (MoE: 128 experts, 8 active)"
for spec in "explicit:battery_v1_final" "evoked:battery_v2_final" \
            "decoupled:battery_v4_final" "compositional:battery_v3d"; do
  COND=${spec%%:*}; BAT=${spec##*:}
  [ -e "$OUT/$COND.json" ] && continue
  python src/experiments/measure.py \
    --model "Qwen/$M" --model-revision "$REV" --tokenizer-revision "$REV" \
    --dtype bfloat16 --device cuda --end-only \
    --battery "data/benchmarks/$BAT.json" --out "$OUT/$COND.json"
done
echo "SWEEP_MOE_OK"
