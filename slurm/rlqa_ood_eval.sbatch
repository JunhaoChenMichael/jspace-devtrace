#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J rlqa_ood
#SBATCH -o logs/rlqa_ood_eval.out
#SBATCH -e logs/rlqa_ood_eval.err
source slurm/a100_env.sh
LOCK=data/results/rlqa_a100/id_lock/lock_manifest.json
test -f "$LOCK" || { echo "REFUSING: no three-seed ID lock; OOD stays sealed" >&2; exit 2; }
OUT=data/results/rlqa_a100/ood
mkdir -p "$OUT"

# 1. Frozen-base reference measurements on the two sealed conditions.
for spec in "battery_v4_final:results_v4f_qwen3-8B" "battery_v3d:results_v3f_qwen3-8B"; do
  B=${spec%%:*}; O=${spec##*:}
  [ -e data/results/${O}.json ] || python src/experiments/measure.py \
    --model Qwen/Qwen3-8B --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
    --dtype bfloat16 --device cuda --end-only --no-verbal-raw \
    --battery data/benchmarks/${B}.json --out data/results/${O}.json
done

ADAPTERS=()
for s in 0 1 2; do
  RUN=$(ls -d data/results/rlqa_a100/runs/formal_rl-qa_*_s${s}_*/ 2>/dev/null | head -1)
  CKPT=$(python -c "import json;print(json.load(open('${RUN}best_checkpoint.json'))['path'])")
  ADAPTERS+=(--adapter "rl-qa-s${s}=${CKPT}")
  echo "seed $s adapter: $CKPT"
done

SPECS=(--spec "decoupled=data/results/results_v4f_qwen3-8B.json::data/benchmarks/battery_v4_final.json"
       --spec "compositional=data/results/results_v3f_qwen3-8B.json::data/benchmarks/battery_v3d.json")

# 2. Admission AUC / containment, no QA.
python src/experiments/evaluate_memory_rl.py \
  --model Qwen/Qwen3-8B "${SPECS[@]}" "${ADAPTERS[@]}" \
  --budgets 2 --skip-qa --skip-no-harm \
  --device cuda --dtype bfloat16 --bootstrap-seed 0 \
  --out "$OUT/ood_auc.json"

# 3. Budgeted recall QA + full-context no-harm, batch-1 isolated recall.
python src/experiments/evaluate_memory_rl.py \
  --model Qwen/Qwen3-8B "${SPECS[@]}" "${ADAPTERS[@]}" \
  --budgets 2 --max-new-tokens 64 --qa-batch-size 1 \
  --device cuda --dtype bfloat16 --bootstrap-seed 0 \
  --out "$OUT/ood_qa.json"

# 4. Predeclared gates over the three seeds (individual, mean, sample sd).
python src/analysis/memory_rl_gates.py "$OUT/ood_qa.json" \
  --rl-qa 'rl-qa-s*' --source decoupled --budget 2 \
  --out "$OUT/gate_summary.json"
echo "RLQA_OOD_OK"
