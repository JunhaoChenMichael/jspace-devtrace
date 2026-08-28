#!/usr/bin/env bash
# run_scaled.sh — measure every checkpoint on the FINAL scaled batteries.
# Usage: bash run_scaled.sh [v1_battery] [v2_battery] [v3_battery]
# 9 checkpoints x 3 batteries = 27 jobs, queued over all GPUs (one job per GPU slot).
# Readout = case-variant W_rr (max over lowercase/capitalized first tokens);
# every model gets template-free V_raw; instruct models also get chat V.
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate jspace
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

B1=${1:-data/benchmarks/battery_v1_final.json}
B2=${2:-data/benchmarks/battery_v2_final.json}
B3=${3:-data/benchmarks/battery_v3d.json}
NGPU=$(nvidia-smi -L | wc -l)

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct|7B-Instruct"
  "Qwen/Qwen2.5-7B|7B-base"
  "Qwen/Qwen2.5-3B-Instruct|3B-Instruct"
  "Qwen/Qwen2.5-3B|3B-base"
  "Qwen/Qwen2.5-1.5B-Instruct|1.5B-Instruct"
  "Qwen/Qwen2.5-1.5B|1.5B-base"
  "Qwen/Qwen2.5-0.5B-Instruct|0.5B-Instruct"
  "Qwen/Qwen2.5-0.5B|0.5B-base"
  "openai-community/gpt2|gpt2-base"
)
# v2 first (headline claims), then v3 (decoupled), then v1 (regime contrast)
BATTERIES=("$B2|v2f" "$B3|v3f" "$B1|v1f")

pids=()
slot=0
for bspec in "${BATTERIES[@]}"; do
  IFS='|' read -r battery bver <<<"$bspec"
  for mspec in "${MODELS[@]}"; do
    IFS='|' read -r model mtag <<<"$mspec"
    gpu=$((slot % NGPU))
    if [ -n "${pids[$gpu]:-}" ]; then wait "${pids[$gpu]}"; fi
    tag="${bver}_${mtag}"
    echo "[launch] gpu=$gpu $tag"
    CUDA_VISIBLE_DEVICES=$gpu python src/experiments/measure.py --model "$model" \
        --dtype bfloat16 --battery "$battery" \
        --out "data/results/results_${tag}.json" >"logs/measure_${tag}.log" 2>&1 &
    pids[$gpu]=$!
    slot=$((slot + 1))
  done
done
wait
echo "ALL SCALED MEASURES DONE"
