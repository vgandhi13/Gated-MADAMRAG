#!/usr/bin/env bash
#SBATCH --job-name=gated-summary
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/summary-%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/baselines/env.sh"
cd "$GATED_ROOT"
python -u -m baselines.summarize "$@"
