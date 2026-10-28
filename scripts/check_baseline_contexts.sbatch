#!/usr/bin/env bash
#SBATCH --job-name=gated-context-check
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --output=logs/context-%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/baselines/env.sh"
python -u "$GATED_ROOT/scripts/check_baseline_contexts.py" --model "$GATED_MODEL" \
    --output "$GATED_WORK/context_lengths.json"
