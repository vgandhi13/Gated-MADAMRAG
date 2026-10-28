#!/usr/bin/env bash
# Debate Gating baseline: confidence-gated debate on SiliconFlow (Qwen/Qwen3-8B, no GPU needed).
#
# By default it runs BOTH datasets (RAMDocs, then TriviaQA top-5). `--limit N` takes the
# first N questions from EACH dataset. Extra arguments go to run_debate_gating.py
# (see `python3 -m baselines.DebateGating.run_debate_gating --help`).
#
# Examples (run from anywhere):
#   bash scripts/run_debate_gating_baseline.sh --limit 5            # smoke test: 5 + 5 questions
#   bash scripts/run_debate_gating_baseline.sh                      # full run: 500 + 500 questions
#   bash scripts/run_debate_gating_baseline.sh --datasets ramdocs   # one dataset only
#   bash scripts/run_debate_gating_baseline.sh --threshold 90 --max-rounds 2
#   WORKERS=4 MAX_CONCURRENCY=8 bash scripts/run_debate_gating_baseline.sh   # fewer parallel calls
#   OUTPUT_DIR=/some/dir bash scripts/run_debate_gating_baseline.sh          # custom results folder
#
# Results: baselines/DebateGating/results/Qwen3-8B[-limitN]/<dataset>/debate_gating/
#   (manifest.json, predictions.jsonl, metrics.json) and logs/run-*.log. A --limit run
#   gets its own folder so it never clashes with the full run. Rerunning the same
#   command resumes from saved predictions.
#
# SILICONFLOW_API_KEY may be exported or placed in .env at the repository root.
#
# Parallelism: requests are network-bound, so CPU is not the limit; each in-flight
# request costs a thread (~1 MB measured, 8 MB budgeted). The real limits are free
# RAM (so the machine never swaps or hits the OOM killer) and the API's rate limit
# (HTTP 429, retried with backoff). Override with WORKERS / MAX_CONCURRENCY.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Lenient .env reader: accepts `KEY=value`, `KEY = value`, quotes, and hyphenated
# names (SILICON-FLOW-API-KEY becomes SILICON_FLOW_API_KEY). Values are never echoed.
if [[ -f "$ROOT/.env" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_-]*)[[:space:]]*=[[:space:]]*(.*)$ ]] || continue
        name="${BASH_REMATCH[1]//-/_}"; value="${BASH_REMATCH[2]%"${BASH_REMATCH[2]##*[![:space:]]}"}"
        value="${value#[\"\']}"; value="${value%[\"\']}"
        [[ -z "${!name:-}" ]] && export "$name=$value"
    done < "$ROOT/.env"
fi
export SILICONFLOW_API_KEY="${SILICONFLOW_API_KEY:-${SILICON_FLOW_API_KEY:-}}"
: "${SILICONFLOW_API_KEY:?Set SILICONFLOW_API_KEY (export it or add it to .env)}"
MODEL="${SILICONFLOW_MODEL:-Qwen/Qwen3-8B}"
LIMIT=""
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in
        --limit) LIMIT="-limit${ARGS[$((i + 1))]:-}" ;;
        --limit=*) LIMIT="-limit${ARGS[$i]#--limit=}" ;;
    esac
done
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/baselines/DebateGating/results/${MODEL##*/}${LIMIT}}"

# --- Hardware-aware concurrency -------------------------------------------------
CORES=$(nproc)
AVAIL_MB=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
LOAD=$(awk '{print int($1 + 0.5)}' /proc/loadavg)
RESERVE_MB=1536          # always left for the OS and other services
PER_REQUEST_MB=8         # conservative per in-flight request
API_CAP=6                # 16 hit HTTP 429 constantly on SiliconFlow; 6 stays under the rate limit
BY_MEM=$(( (AVAIL_MB - RESERVE_MB - 200) / PER_REQUEST_MB ))
if (( BY_MEM < 2 )); then
    echo "Only ${AVAIL_MB} MB RAM available; free memory before running." >&2
    exit 1
fi
CONC=$(( BY_MEM < API_CAP ? BY_MEM : API_CAP ))
if (( LOAD >= CORES )); then CONC=$(( CONC / 2 > 2 ? CONC / 2 : 2 )); fi  # machine busy
MAX_CONCURRENCY="${MAX_CONCURRENCY:-$CONC}"
# Several agents of one question are in flight at once, so run half as many questions.
WORKERS="${WORKERS:-$(( MAX_CONCURRENCY / 2 > 1 ? MAX_CONCURRENCY / 2 : 1 ))}"
echo "Hardware: ${CORES} cores, ${AVAIL_MB} MB available RAM, load ${LOAD}" \
     "-> ${WORKERS} question workers, ${MAX_CONCURRENCY} max in-flight requests"

mkdir -p "$OUTPUT_DIR/logs"
LOG="$OUTPUT_DIR/logs/run-$(date +%Y%m%d-%H%M%S).log"
cd "$ROOT"
# nice keeps SSH and other services responsive on a small server.
nice -n 10 python3 -u -m baselines.DebateGating.run_debate_gating \
    --model "$MODEL" --output-dir "$OUTPUT_DIR" \
    --workers "$WORKERS" --max-concurrency "$MAX_CONCURRENCY" "$@" 2>&1 | tee "$LOG"
