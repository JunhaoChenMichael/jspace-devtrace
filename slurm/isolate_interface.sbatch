#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=250G
#SBATCH -t 2:00:00
#SBATCH -J iso_iface
#SBATCH -o logs/isolate_interface.out
#SBATCH -e logs/isolate_interface.err
source slurm/a100_env.sh
REV=$(cat "$HF_HOME/hub/models--Qwen--Qwen3-32B/refs/main")
python /rodata/azradonc_dev/m253405/jspace-devtrace/scripts/isolate_interface.py \
  Qwen/Qwen3-32B "$REV" data/benchmarks/battery_v4_final.json 4
