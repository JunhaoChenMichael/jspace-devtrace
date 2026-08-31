#!/usr/bin/env bash
ulimit -u 65536
cd /rodata/azradonc_dev/m253405/jspace-devtrace
for i in $(seq 1 240); do
  grep -q "^PINNED" /rodata/azradonc_dev/m253405/logs/dl_grid.log 2>/dev/null && break
  sleep 60
done
echo "[$(date -u +%H:%M)] downloads finished; submitting the predictive grid"
sbatch slurm/predictive_grid_v3.sbatch
