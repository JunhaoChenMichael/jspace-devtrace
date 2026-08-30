#!/usr/bin/env bash
# Non-invasive peak-GPU-memory sampler: runs nvidia-smi *inside* an existing
# Slurm allocation with --overlap, so it neither disturbs nor reserves anything.
ulimit -u 65536
JOBID=$1; OUT=$2; INTERVAL=${3:-60}
peak=0
while squeue -h -j "$JOBID" -o %T 2>/dev/null | grep -q RUNNING; do
  used=$(srun --overlap --jobid="$JOBID" -n1 nvidia-smi \
           --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [[ "$used" =~ ^[0-9]+$ ]]; then
    [ "$used" -gt "$peak" ] && peak=$used
    echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"used_mib\":$used,\"peak_mib\":$peak}" >> "$OUT"
  fi
  sleep "$INTERVAL"
done
echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"final_peak_mib\":$peak,\"final_peak_gib\":$(awk "BEGIN{printf \"%.2f\", $peak/1024}")}" >> "$OUT"
