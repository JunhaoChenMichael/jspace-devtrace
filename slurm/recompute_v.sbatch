#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=250G
#SBATCH -t 4:00:00
#SBATCH -J recompute_v
#SBATCH -o logs/recompute_v_%a.out
#SBATCH -e logs/recompute_v_%a.err
#SBATCH --array=0-2
source slurm/a100_env.sh
MODELS=(Qwen3-8B Qwen3-14B Qwen3-32B)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
REV=$(cat "$HF_HOME/hub/models--Qwen--$M/refs/main")
OUT=data/results/a100_next_boundary_campaign/shared/reporter_interface_audit
mkdir -p "$OUT"
[ -e "$OUT/v_epsilon_recompute_${M}.json" ] || \
python scripts/recompute_v_without_epsilon.py "Qwen/$M" "$REV" \
  data/benchmarks/battery_v4_final.json "$OUT/v_epsilon_recompute_${M}.json"
echo "RECOMPUTE_OK $M"
