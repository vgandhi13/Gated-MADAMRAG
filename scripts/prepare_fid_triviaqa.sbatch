#!/bin/bash
#SBATCH --job-name=fid-triviaqa-500
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:15:00
#SBATCH --output=logs/fid-triviaqa-%j.log

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/baselines/env.sh"

python3 -u "$GATED_ROOT/scripts/prepare_fid_triviaqa.py" \
    --cache-dir "$GATED_WORK/triviaqa_fid/sources" \
    --output-dir "$GATED_ROOT/data/triviaqa_top5" \
    --sample-size 500 \
    --seed 42 \
    --top-k 5 \
    --require-answer-string
