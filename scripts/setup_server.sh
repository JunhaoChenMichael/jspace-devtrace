#!/usr/bin/env bash
# One-shot server setup for jspace-devtrace. Run AFTER extracting the tarball:
#   source server_env.sh && bash setup_server.sh
# Idempotent-ish; safe to re-run. Assumes a CUDA box with conda or python3.10-3.13.
set -e
cd "$(dirname "$0")"

echo "==> 1. Python env"
if command -v conda >/dev/null 2>&1; then
  conda create -y -n jspace python=3.11 || true
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate jspace
else
  python3 -m venv .venv && source .venv/bin/activate
fi

echo "==> 2. Dependencies (CUDA torch)"
pip install --upgrade pip
pip install torch transformers accelerate huggingface_hub openai scikit-learn scipy

echo "==> 3. Credentials check"
: "${HF_TOKEN:?source server_env.sh first (HF_TOKEN unset)}"
: "${OPENAI_API_KEY:?source server_env.sh first (OPENAI_API_KEY unset)}"
python - <<'PY'
import os; from huggingface_hub import login
login(os.environ["HF_TOKEN"], add_to_git_credential=False)
print("HF auth OK")
PY

echo "==> 4. Pre-download models (fast with HF_TOKEN)"
python - <<'PY'
from huggingface_hub import snapshot_download
models = ["gpt2","Qwen/Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct",
          "Qwen/Qwen2.5-3B","Qwen/Qwen2.5-3B-Instruct",
          "Qwen/Qwen2.5-7B","Qwen/Qwen2.5-7B-Instruct"]
for m in models:
    snapshot_download(m, allow_patterns=["*.safetensors","*.json","*.txt"])
    print("cached", m)
print("ALL MODELS READY")
PY

echo "==> 5. Smoke test (tiny, ~1 min): measure + analyze on cached 0.5B v2"
python pilot/measure.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --battery pilot/battery_v2.json --out pilot/results_v2_0.5B-Instruct.json
python pilot/analyze.py pilot/results_v2_0.5B-Instruct.json | tail -12

echo "==> DONE. Next: HANDOFF.md section 8 (TODO). Start with 7B:"
echo "    python pilot/measure.py --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \\"
echo "        --battery pilot/battery_v2.json --out pilot/results_v2_7B-Instruct.json"
