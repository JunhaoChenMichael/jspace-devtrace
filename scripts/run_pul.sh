#!/usr/bin/env bash
# run_replay.sh — W_rep grid: {Qwen 0.5/1.5/3/7B-I, OLMo-2-7B-I} x {v3d, v4, v2f}
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate jspace
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
NGPU=$(nvidia-smi -L | wc -l)

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct|7B-Instruct"
  "allenai/OLMo-2-1124-7B-Instruct|olmo7b-Instruct"
  "Qwen/Qwen2.5-3B-Instruct|3B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct|1.5B-Instruct"
  "Qwen/Qwen2.5-0.5B-Instruct|0.5B-Instruct"
)
BATTERIES=(
  "data/benchmarks/battery_v3d.json|v3"
  "data/benchmarks/battery_v4_final.json|v4"
  "data/benchmarks/battery_v2_final.json|v2f"
)

pids=()
slot=0
for bspec in "${BATTERIES[@]}"; do
  IFS='|' read -r battery bver <<<"$bspec"
  for mspec in "${MODELS[@]}"; do
    IFS='|' read -r model mtag <<<"$mspec"
    out="data/results/results_pul_${bver}_${mtag}.json"
    [ -f "$out" ] && { echo "[skip] $out exists"; continue; }
    gpu=$((slot % NGPU))
    if [ -n "${pids[$gpu]:-}" ]; then wait "${pids[$gpu]}"; fi
    echo "[launch] gpu=$gpu pul_${bver}_${mtag}"
    CUDA_VISIBLE_DEVICES=$gpu python src/experiments/pul_readout.py --model "$model" \
        --battery "$battery" --out "$out" >"logs/pul_${bver}_${mtag}.log" 2>&1 &
    pids[$gpu]=$!
    slot=$((slot + 1))
  done
done
wait
echo "ALL PUL RUNS DONE"
