#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=250G
#SBATCH -t 6:00:00
#SBATCH -J rep_audit
#SBATCH -o logs/reporter_interface_audit_%a.out
#SBATCH -e logs/reporter_interface_audit_%a.err
#SBATCH --array=0-1
source slurm/a100_env.sh
MODELS=(Qwen3-8B Qwen3-32B)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
REV=$(cat "$HF_HOME/hub/models--Qwen--$M/refs/main")
OUT=data/results/a100_next_boundary_campaign/shared/reporter_interface_audit
# ID-side (Evoked, a training source) and the already-released Decoupled M0
# battery. Measurement only; no new OOD attempt is created or consumed.
for spec in "evoked:battery_v2_final" "decoupled:battery_v4_final"; do
  L=${spec%%:*}; B=${spec##*:}
  [ -e "$OUT/${M}_${L}.json" ] && continue
  python scripts/audit_reporter_interface_equivalence.py \
    --model "Qwen/$M" --model-revision "$REV" \
    --battery "data/benchmarks/$B.json" --label "$L" \
    --out "$OUT/${M}_${L}.json"
done
# The population on which the RL track reported its step-0 Yes rate.
TAG=$([ "$M" = "Qwen3-32B" ] && echo qwen3-32B || echo qwen3-8B)
if [ ! -e "$OUT/${M}_rl_validation.json" ]; then
  python scripts/audit_reporter_interface_equivalence.py \
    --model "Qwen/$M" --model-revision "$REV" \
    --rl-validation-split "$TAG" --label rl_validation \
    --out "$OUT/${M}_rl_validation.json"
fi
echo "REPORTER_AUDIT_OK $M"
