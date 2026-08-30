#!/usr/bin/env bash
# Seeds 1/2 for the operator-authorised multi-seed characterisation of the
# seed-0 null. Both seeds must hold an ID lock before EITHER opens OOD.
ulimit -u 65536
cd /rodata/azradonc_dev/m253405/jspace-devtrace
source /rodata/azradonc_dev/m253405/myconda/etc/profile.d/conda.sh
conda activate jspace
export HF_HOME=/rodata/azradonc_dev/m253405/cache HF_HUB_OFFLINE=1
export METACOG_EXPECTED_MODEL="Qwen/Qwen3-32B"
unset HF_TOKEN
ROOT=data/results/rlqa32_a100
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }
wait_for(){ local n=$1 max=$2 i=0
  while [ $i -lt $((max*2)) ]; do
    [ "$(squeue -u "$USER" -h -n "$n" | wc -l)" -eq 0 ] && return 0
    sleep 30; i=$((i+1)); done
  say "TIMEOUT $n"; return 1; }

wait_for rlqa32_formal12 2880 || exit 1
for s in 1 2; do
  grep -q "RLQA32_FORMAL_SEED${s}_OK" logs/rlqa32_formal_seed${s}.out || {
    say "STOP: seed $s training did not finish"; exit 1; }
done
say "seeds 1 and 2 trained"

ARGS=()
for s in 1 2; do
  RUN=$(ls -d $ROOT/runs/formal_rl-qa_*_s${s}_*/ | head -1)
  ARGS+=(--run "${s}=${RUN%/}")
done
python scripts/lock_rlqa_checkpoints.py "${ARGS[@]}" --seeds 1,2 \
  --out $ROOT/id_lock_seeds12/lock_manifest.json || { say "STOP: lock failed"; exit 1; }
say "seeds 1 and 2 locked; opening OOD"

sbatch slurm/rlqa32_ood_seeds12.sbatch >/dev/null
sleep 20; wait_for rlqa32_ood12 1440 || exit 1
for s in 1 2; do
  grep -q "RLQA32_OOD_SEED${s}_OK" logs/rlqa32_ood_seed${s}.out || {
    say "STOP: seed $s OOD failed"; exit 1; }
done
say "seeds 1 and 2 OOD complete"
