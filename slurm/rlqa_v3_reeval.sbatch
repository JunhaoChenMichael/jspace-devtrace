#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=250G
#SBATCH -t 24:00:00
#SBATCH -J rlqa_v3_reeval
#SBATCH -o logs/rlqa_v3_reeval_%a.out
#SBATCH -e logs/rlqa_v3_reeval_%a.err
#SBATCH --array=0-1
MODELS=(Qwen3-8B Qwen3-32B); TAGS=(qwen3-8B qwen3-32B)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; T=${TAGS[$SLURM_ARRAY_TASK_ID]}
export METACOG_EXPECTED_MODEL="Qwen/$M"
source slurm/a100_env.sh
REV=$(cat "$HF_HOME/hub/models--Qwen--$M/refs/main")
OUT=data/results/a100_next_boundary_campaign/qwen3_rlqa_v3/$M
mkdir -p "$OUT"
# Only the Original arm changes: the RL-QA adapters are the ones already locked,
# and the policy scorer never used the defective path.
ADAPTERS=()
if [ "$M" = "Qwen3-8B" ]; then
  for s in 0 1 2; do
    RUN=$(ls -d data/results/rlqa_a100/runs/formal_rl-qa_*_s${s}_*/ | head -1)
    CK=$(python -c "import json;print(json.load(open('${RUN}best_checkpoint.json'))['path'])")
    ADAPTERS+=(--adapter "rl-qa-s${s}=$CK")
  done
else
  RUN=$(ls -d data/results/rlqa32_a100/runs/formal_rl-qa_*_s0_*/ | grep -v CANCELLED | head -1)
  CK=$(python -c "import json;print(json.load(open('${RUN}best_checkpoint.json'))['path'])")
  ADAPTERS+=(--adapter "rl-qa-s0=$CK")
fi
echo "adapters: ${ADAPTERS[@]}"
SPECS=(--spec "decoupled=data/results/results_v4f_${T}_v3.json::data/benchmarks/battery_v4_final.json"
       --spec "compositional=data/results/results_v3f_${T}_v3.json::data/benchmarks/battery_v3d.json")
[ -e "$OUT/ood_auc_v3.json" ] || python src/experiments/evaluate_memory_rl.py \
  --model "Qwen/$M" "${SPECS[@]}" "${ADAPTERS[@]}" \
  --budgets 2 --skip-qa --skip-no-harm --device cuda --dtype bfloat16 --bootstrap-seed 0 \
  --out "$OUT/ood_auc_v3.json"
[ -e "$OUT/ood_qa_v3.json" ] || python src/experiments/evaluate_memory_rl.py \
  --model "Qwen/$M" "${SPECS[@]}" "${ADAPTERS[@]}" \
  --budgets 2 --max-new-tokens 64 --qa-batch-size 1 \
  --device cuda --dtype bfloat16 --bootstrap-seed 0 \
  --out "$OUT/ood_qa_v3.json"
echo "RLQA_V3_REEVAL_OK $M"
