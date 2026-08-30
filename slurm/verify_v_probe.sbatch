#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=250G
#SBATCH -t 2:00:00
#SBATCH -J verify_v
#SBATCH -o logs/verify_v_probe.out
#SBATCH -e logs/verify_v_probe.err
source slurm/a100_env.sh
S=scripts/verify_v_readout_independent.py
echo "########## Qwen3-8B (should reproduce V=0.3415 on the full battery)"
python "$S" Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 data/benchmarks/battery_v4_final.json 0
echo "########## Qwen3-32B (should reproduce V=0.6571)"
python "$S" Qwen/Qwen3-32B 9216db5781bf21249d130ec9da846c4624c16137 data/benchmarks/battery_v4_final.json 0
