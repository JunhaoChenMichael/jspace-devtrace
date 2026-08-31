#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=250G
#SBATCH -t 1:00:00
#SBATCH -J iso2
#SBATCH -o logs/isolate2.out
#SBATCH -e logs/isolate2.err
source slurm/a100_env.sh
REV=$(cat "$HF_HOME/hub/models--Qwen--Qwen3-32B/refs/main")
python scripts/isolate_interface2.py Qwen/Qwen3-32B "$REV" data/benchmarks/battery_v4_final.json
