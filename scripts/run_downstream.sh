#!/usr/bin/env bash
# run_downstream.sh — memory-budgeted QA on the FINAL batteries.
# 4 instruct sizes x 3 batteries = 12 jobs, queued over all GPUs.
# Uses the fixed random baseline, per-episode logs, McNemar, embedding+recency.
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate jspace
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

NGPU=$(nvidia-smi -L | wc -l)

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct|7B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct|3B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct|1.5B-Instruct"
  "Qwen/Qwen2.5-0.5B-Instruct|0.5B-Instruct"
)
# battery|results-version-prefix
BATTERIES=(
  "data/benchmarks/battery_v2_final.json|v2f"
  "data/benchmarks/battery_v3d.json|v3f"
  "data/benchmarks/battery_v1_final.json|v1f"
)

pids=()
slot=0
for bspec in "${BATTERIES[@]}"; do
  IFS='|' read -r battery bver <<<"$bspec"
  for mspec in "${MODELS[@]}"; do
    IFS='|' read -r model mtag <<<"$mspec"
    gpu=$((slot % NGPU))
    if [ -n "${pids[$gpu]:-}" ]; then wait "${pids[$gpu]}"; fi
    res="data/results/results_${bver}_${mtag}.json"
    out="data/results/downstream_${bver}_${mtag}.json"
    echo "[launch] gpu=$gpu ${bver}_${mtag}"
    CUDA_VISIBLE_DEVICES=$gpu python src/experiments/downstream.py --model "$model" \
        --dtype bfloat16 --results "$res" --battery "$battery" --budgets 1,2,3 \
        --out "$out" >"logs/down_${bver}_${mtag}.log" 2>&1 &
    pids[$gpu]=$!
    slot=$((slot + 1))
  done
done
wait
echo "ALL FINAL DOWNSTREAM DONE"
