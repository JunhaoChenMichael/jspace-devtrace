# Shared prelude for every A100 Slurm job in this repository.
ulimit -u 65536
set -euo pipefail
export REPO=/rodata/azradonc_dev/m253405/jspace-devtrace
export HF_HOME=/rodata/azradonc_dev/m253405/cache
export HF_HUB_OFFLINE=1            # weights are pre-staged; compute nodes have no outbound internet
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN || true
export TOKENIZERS_PARALLELISM=false
export METACOG_EXPECTED_GPU="${METACOG_EXPECTED_GPU:-NVIDIA A100-SXM4-80GB}"
export MODEL_REV=${MODEL_REV:-b968826d9c46dd6066d109eabc6255188de91218}
source /rodata/azradonc_dev/m253405/myconda/etc/profile.d/conda.sh
conda activate jspace
cd "$REPO"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
export METACOG_EXPECTED_MODEL="${METACOG_EXPECTED_MODEL:-Qwen/Qwen3-8B}"
