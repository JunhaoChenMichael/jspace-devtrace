#!/usr/bin/env bash
# Autonomous Track A (32B, seed 0) chain. Every stage is a fail-closed gate:
# a non-GREEN gate stops the chain and leaves the evidence in place.
ulimit -u 65536
cd /rodata/azradonc_dev/m253405/jspace-devtrace
source /rodata/azradonc_dev/m253405/myconda/etc/profile.d/conda.sh
conda activate jspace
export HF_HOME=/rodata/azradonc_dev/m253405/cache HF_HUB_OFFLINE=1
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-32B"
unset HF_TOKEN
ROOT=data/results/rlqa32_a100
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
wait_for() {  # wait_for <job-name> <max-minutes>
  local n=$1 max=$2 i=0
  while [ $i -lt $((max*2)) ]; do
    [ "$(squeue -u "$USER" -h -n "$n" | wc -l)" -eq 0 ] && return 0
    sleep 30; i=$((i+1))
  done
  say "TIMEOUT waiting for $n"; return 1
}

# ---------------------------------------------------------------- Gate B0
wait_for rlqa32_b0 1440 || exit 1
B0=$ROOT/preflight/b0_s0_k2_g16_t5/summary.json
[ -f "$B0" ] || { say "STOP: B0 produced no summary"; exit 1; }
STATUS=$(python -c "import json;print(json.load(open('$B0'))['gate_b0']['status'])")
say "Gate B0 = $STATUS"
if [ "$STATUS" != "GREEN" ]; then
  say "STOP: Gate B0 is $STATUS; the plan forbids training and forbids searching temperature."
  exit 1
fi

# ------------------------------------------------------- freeze the recipe
if [ ! -f "$ROOT/RECIPE_FREEZE.json" ]; then
python - <<'PY'
import hashlib, json, datetime
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
b0 = json.load(open("data/results/rlqa32_a100/preflight/b0_s0_k2_g16_t5/summary.json"))
freeze = {
 "schema_version": "rlqa-a100-recipe-freeze/v1",
 "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
 "model": "Qwen/Qwen3-32B",
 "model_revision": "9216db5781bf21249d130ec9da846c4624c16137",
 "tokenizer_revision": "9216db5781bf21249d130ec9da846c4624c16137",
 "method": "rl-qa",
 "recipe": {"lambda_qa":1.0,"lambda_w":0.0,"budget":2,"group_size":8,"grpo_epochs":2,
   "lora_rank":32,"dtype":"bfloat16","max_steps":300,"learning_rate":1e-6,"beta":0.03,
   "temperature":5.0,"max_length":2048,"answer_tokens":64,
   "optimisation_seeds":[0],"split_seed":0,"eval_every":100,"save_every":100,
   "checkpoint_selection":"strict first maximum of ID QA on the fixed ID validation split"},
 "temperature_decision": {
   "rule_applied": "README_32B_RLQA_A100 6.3: 5.0 is the first and only default candidate; "
                   "lock it if selection diversity passes, then check reward viability. "
                   "Searching temperature is forbidden.",
   "gate_b0": b0["gate_b0"],
   "decision": "temperature = 5.0",
   "note": "Diversity was evaluated before any QA reward was computed; no OOD data was used."},
 "id_only_evidence": {
   "split_manifest_sha256": json.load(open("data/results/rlqa32_a100/dryrun_s0/split_manifest.json"))["manifest_sha256"],
   "train_episodes":175,"validation_episodes":45,
   "training_sources":["explicit","evoked","evoked_g2"],
   "prompt_leak_audit": json.load(open("data/results/rlqa32_a100/preflight/prompt_leak_audit.json"))["counts"],
   "sealed_until_lock":["decoupled","compositional"]},
 "teacher_artifacts_sha256": {f"data/results/results_{t}_qwen3-32B.json": sha(f"data/results/results_{t}_qwen3-32B.json")
   for t in ("v1f","v2f","v2g2")},
}
with open("data/results/rlqa32_a100/RECIPE_FREEZE.json","x") as h:
    json.dump(freeze,h,indent=2,sort_keys=True); h.write("\n")
print("recipe frozen at temperature", freeze["recipe"]["temperature"])
PY
fi

# ---------------------------------------------------------------- canary
say "submitting canary"
sbatch slurm/rlqa32_canary.sbatch >/dev/null
sleep 20; wait_for rlqa32_canary 480 || exit 1
grep -q "RLQA32_CANARY_OK" logs/rlqa32_canary.out || { say "STOP: canary did not pass"; exit 1; }
say "canary PASS | $(grep PEAK_ALLOCATED_GIB logs/rlqa32_canary.out | tail -1)"

# ---------------------------------------------------------------- formal
say "submitting formal seed 0"
sbatch slurm/rlqa32_formal.sbatch >/dev/null
sleep 20; wait_for rlqa32_formal 2880 || exit 1
grep -q "RLQA32_FORMAL_OK" logs/rlqa32_formal_seed0.out || { say "STOP: formal run failed"; exit 1; }
say "formal seed 0 complete"

# ------------------------------------------------------------- ID lock
RUN=$(ls -d $ROOT/runs/formal_rl-qa_*_s0_*/ | head -1)
python scripts/lock_rlqa_checkpoints.py --run "0=${RUN%/}" --seeds 0 \
  --out $ROOT/id_lock/lock_manifest.json || { say "STOP: lock failed; OOD stays sealed"; exit 1; }
python - <<'PY'
import json, datetime
lock = json.load(open("data/results/rlqa32_a100/id_lock/lock_manifest.json"))
with open("data/results/rlqa32_a100/ID_LOCK_STOP.json","x") as h:
    json.dump({"schema_version":"rlqa-id-lock-stop/v1",
      "stopped_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
      "reason":"id_lock_complete_ood_sealed","seed":0,
      "selected_step":lock["seeds"][0]["step"],
      "checkpoint_tree_sha256":lock["seeds"][0]["checkpoint_tree_sha256"],
      "manifest_sha256":lock["manifest_sha256"],
      "ood_opened":False,"automatic_seed_expansion_authorized":False},h,indent=2,sort_keys=True)
print("ID lock at step", lock["seeds"][0]["step"])
PY

# ------------------------------------------------------- one-shot OOD
say "submitting one-shot OOD"
sbatch slurm/rlqa32_ood.sbatch >/dev/null
sleep 20; wait_for rlqa32_ood 1440 || exit 1
grep -q "RLQA32_OOD_OK" logs/rlqa32_ood.out || { say "STOP: OOD evaluation failed"; exit 1; }
say "OOD complete; chain finished, awaiting report generation"
