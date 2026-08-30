#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -J rlqa_wsnh
#SBATCH -o logs/rlqa_wsnh_seed%a.out
#SBATCH -e logs/rlqa_wsnh_seed%a.err
#SBATCH --array=0-2
source slurm/a100_env.sh
LOCK=data/results/rlqa_a100/id_lock/lock_manifest.json
test -f "$LOCK" || { echo "REFUSING: no three-seed ID lock" >&2; exit 2; }
s=${SLURM_ARRAY_TASK_ID}
RUN=$(ls -d data/results/rlqa_a100/runs/formal_rl-qa_*_s${s}_*/ 2>/dev/null | head -1)
CKPT=$(python -c "import json;print(json.load(open('${RUN}best_checkpoint.json'))['path'])")
# Same Decoupled battery, same --end-only readout, adapter enabled: does the
# adapter damage the latent workspace it was never trained to touch?
python src/experiments/measure.py \
  --model Qwen/Qwen3-8B --model-revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
  --adapter "$CKPT" --dtype bfloat16 --device cuda --end-only --no-verbal-raw \
  --battery data/benchmarks/battery_v4_final.json \
  --out data/results/rlqa_a100/ood/workspace_rl-qa-s${s}_decoupled_end_only.json
echo "WSNH_SEED${s}_OK"
