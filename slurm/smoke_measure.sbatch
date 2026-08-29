#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:40:00
#SBATCH -J jspace_smoke
#SBATCH -o logs/smoke_measure.out
#SBATCH -e logs/smoke_measure.err
source slurm/a100_env.sh
OUT=/tmp/claude-1536624804/-rodata-azradonc-dev-m253405/1f24e0a9-298b-4d17-a045-7cd7ef1d7d41/scratchpad/smoke_measure.json
python src/experiments/measure.py \
  --model Qwen/Qwen3-8B --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
  --dtype bfloat16 --device cuda --end-only --no-verbal-raw \
  --battery data/benchmarks/battery_v1_final.json --limit-episodes 2 --out "$OUT"
echo "SMOKE_OK"; python -c "import json;d=json.load(open('$OUT'));print(type(d), len(d) if hasattr(d,'__len__') else d)"
