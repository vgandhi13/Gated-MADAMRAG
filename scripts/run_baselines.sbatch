#!/usr/bin/env bash
#SBATCH --job-name=gated-baselines
#SBATCH --partition=gpu,gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=l4|l40s
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baselines-%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/baselines/env.sh"
mkdir -p "$TMPDIR" "$GATED_WORK/results"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torch-inductor"
cd "$GATED_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -u -m baselines.run "$@"
