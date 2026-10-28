#!/usr/bin/env bash
# Shared environment for jobs and interactive use. Personal/cluster paths go in
# local/env.local.sh (gitignored); copy local/env.local.sh.example to start.
export GATED_ROOT="${GATED_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "$GATED_ROOT/local/env.local.sh" ]]; then
    source "$GATED_ROOT/local/env.local.sh"
fi
export GATED_WORK="${GATED_WORK:-$GATED_ROOT/work}"
export GATED_ENV="${GATED_ENV:-$GATED_WORK/env}"
export GATED_MODEL="${GATED_MODEL:-Qwen/Qwen3-8B}"
export PATH="$GATED_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME="$GATED_WORK/cache/huggingface"
export PIP_CACHE_DIR="$GATED_WORK/cache/pip"
export VLLM_CACHE_ROOT="$GATED_WORK/cache/vllm"
export TMPDIR="$GATED_WORK/tmp/${SLURM_JOB_ID:-interactive}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
