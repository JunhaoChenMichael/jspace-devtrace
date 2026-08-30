#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=250G
#SBATCH -t 24:00:00
#SBATCH -J rlqa32_ood12
#SBATCH -o logs/rlqa32_ood_seed%a.out
#SBATCH -e logs/rlqa32_ood_seed%a.err
#SBATCH --array=1-2
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-32B"
export MODEL_REV=9216db5781bf21249d130ec9da846c4624c16137
source slurm/a100_env.sh
S=${SLURM_ARRAY_TASK_ID}
LOCK=data/results/rlqa32_a100/id_lock_seeds12/lock_manifest.json
test -f "$LOCK" || { echo "REFUSING: seeds 1 and 2 are not both locked; OOD stays sealed" >&2; exit 2; }
OUT=data/results/rlqa32_a100/ood_seed${S}; mkdir -p "$OUT"
RUN=$(ls -d data/results/rlqa32_a100/runs/formal_rl-qa_*_s${S}_*/ | head -1)
CKPT=$(python -c "import json;print(json.load(open('${RUN}best_checkpoint.json'))['path'])")
echo "seed $S locked adapter: $CKPT"
SPECS=(--spec "decoupled=data/results/results_v4f_qwen3-32B.json::data/benchmarks/battery_v4_final.json"
       --spec "compositional=data/results/results_v3f_qwen3-32B.json::data/benchmarks/battery_v3d.json")
python src/experiments/evaluate_memory_rl.py \
  --model Qwen/Qwen3-32B "${SPECS[@]}" --adapter "rl-qa-s${S}=$CKPT" \
  --budgets 2 --skip-qa --skip-no-harm --device cuda --dtype bfloat16 --bootstrap-seed 0 \
  --out "$OUT/ood_auc.json"
python src/experiments/evaluate_memory_rl.py \
  --model Qwen/Qwen3-32B "${SPECS[@]}" --adapter "rl-qa-s${S}=$CKPT" \
  --budgets 2 --max-new-tokens 64 --qa-batch-size 1 \
  --device cuda --dtype bfloat16 --bootstrap-seed 0 \
  --out "$OUT/ood_qa.json"
python src/experiments/measure.py \
  --model Qwen/Qwen3-32B --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
  --adapter "$CKPT" --dtype bfloat16 --device cuda --end-only --no-verbal-raw \
  --battery data/benchmarks/battery_v4_final.json \
  --out "$OUT/workspace_rl-qa-s${S}_decoupled_end_only.json"
echo "RLQA32_OOD_SEED${S}_OK"
