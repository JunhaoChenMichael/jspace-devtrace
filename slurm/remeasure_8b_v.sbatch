#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 6:00:00
#SBATCH -J remeasure_v
#SBATCH -o logs/remeasure_8b_v_%a.out
#SBATCH -e logs/remeasure_8b_v_%a.err
#SBATCH --array=0-3
source slurm/a100_env.sh
REV=$(cat "$HF_HOME/hub/models--Qwen--Qwen3-8B/refs/main")
OUT=data/results/a100_next_boundary_campaign/shared/v_correction
mkdir -p "$OUT"
S=$SLURM_ARRAY_TASK_ID
run() {  # run <label> <adapter-or-empty>
  local label=$1 adapter=$2
  for spec in "decoupled:battery_v4_final" "compositional:battery_v3d"; do
    C=${spec%%:*}; B=${spec##*:}
    [ -e "$OUT/8b_${label}_${C}.json" ] && continue
    if [ -z "$adapter" ]; then
      python scripts/measure_v_decomposition.py --model Qwen/Qwen3-8B --model-revision "$REV" \
        --battery "data/benchmarks/$B.json" --label "${label}_${C}" --out "$OUT/8b_${label}_${C}.json"
    else
      python scripts/measure_v_decomposition.py --model Qwen/Qwen3-8B --model-revision "$REV" \
        --adapter "$adapter" --battery "data/benchmarks/$B.json" \
        --label "${label}_${C}" --out "$OUT/8b_${label}_${C}.json"
    fi
  done
}
if [ "$S" -eq 3 ]; then
  run base ""
else
  RUN=data/results/metacog_a100_3seed/seed${S}
  CK=$(python -c "import json;print(json.load(open('$RUN/id_lock/lock_manifest.json'))['checkpoint_path'])")
  run "seed${S}" "$RUN/$CK"
fi
echo "REMEASURE_OK $S"
