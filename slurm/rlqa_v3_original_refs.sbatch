#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=250G
#SBATCH -t 8:00:00
#SBATCH -J rlqa_v3_refs
#SBATCH -o logs/rlqa_v3_refs_%a.out
#SBATCH -e logs/rlqa_v3_refs_%a.err
#SBATCH --array=0-1
source slurm/a100_env.sh
MODELS=(Qwen3-8B Qwen3-32B); TAGS=(qwen3-8B qwen3-32B)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; T=${TAGS[$SLURM_ARRAY_TASK_ID]}
REV=$(cat "$HF_HOME/hub/models--Qwen--$M/refs/main")
for spec in "battery_v4_final:results_v4f_${T}_v3" "battery_v3d:results_v3f_${T}_v3"; do
  B=${spec%%:*}; O=${spec##*:}
  [ -e "data/results/${O}.json" ] && continue
  python src/experiments/measure.py --model "Qwen/$M" --model-revision "$REV" \
    --tokenizer-revision "$REV" --dtype bfloat16 --device cuda --end-only --no-verbal-raw \
    --battery "data/benchmarks/$B.json" --out "data/results/${O}.json"
done
echo "RLQA_V3_REFS_OK $M"
