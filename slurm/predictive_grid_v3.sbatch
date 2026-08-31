#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=250G
#SBATCH -t 12:00:00
#SBATCH -J grid_v3
#SBATCH -o logs/grid_v3_%a.out
#SBATCH -e logs/grid_v3_%a.err
#SBATCH --array=0-18
source slurm/a100_env.sh
REPOS=(
 Qwen/Qwen3-0.6B
 Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B
 Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct
 allenai/OLMo-2-0425-1B allenai/OLMo-2-0425-1B-SFT allenai/OLMo-2-0425-1B-DPO allenai/OLMo-2-0425-1B-Instruct
 allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-SFT allenai/OLMo-2-1124-7B-Instruct
 mistralai/Mistral-7B-Instruct-v0.3
 openai-community/gpt2
 Qwen/Qwen3-14B
)
R=${REPOS[$SLURM_ARRAY_TASK_ID]}
SLUG=$(echo "$R" | tr '/' '_')
CACHE="$HF_HOME/hub/models--$(echo "$R" | tr '/' '-' | sed 's/-/--/')"
CACHE="$HF_HOME/hub/models--${R//\//--}"
REV=$(cat "$CACHE/refs/main" 2>/dev/null) || { echo "SKIP $R: not downloaded" >&2; exit 0; }
OUT=data/results/a100_next_boundary_campaign/predictive_grid_v3/$SLUG
mkdir -p "$OUT"
echo "model=$R rev=$REV"
# Base checkpoints have no chat template, so V_chat is undefined for them; the
# script detects that and only V_raw is written. V_raw stays enabled throughout.
for spec in "explicit:battery_v1_final" "evoked:battery_v2_final" \
            "decoupled:battery_v4_final" "compositional:battery_v3d"; do
  C=${spec%%:*}; B=${spec##*:}
  [ -e "$OUT/$C.json" ] && continue
  python src/experiments/measure.py --model "$R" --model-revision "$REV" \
    --tokenizer-revision "$REV" --dtype bfloat16 --device cuda --end-only \
    --battery "data/benchmarks/$B.json" --out "$OUT/$C.json" || echo "FAILED $R $C" >&2
done
echo "GRID_V3_OK $R"
