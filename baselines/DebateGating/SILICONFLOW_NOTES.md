# SiliconFlow parallelism notes

Measured on 2026-10-28 with `Qwen/Qwen3-8B`, `max_new_tokens=128`, on RAMDocs, from a
Raspberry Pi 5 (4 cores, 8 GB RAM). Rate limits depend on the SiliconFlow account
tier (RPM/TPM), so re-check these numbers if the account or model changes.

| Setting (`WORKERS` / `MAX_CONCURRENCY`) | Throughput | HTTP 429 retries | Deepest retry | Failed calls |
|---|---|---|---|---|
| 8 questions / 16 requests | ~43 questions/min | ~3 per question (587 in 197 questions) | 3rd attempt | 0 |
| **3 questions / 6 requests (default)** | ~17 questions/min | ~0.6 per question (96 in 164 questions), in bursts | 3rd attempt | 0 |

**Safe setting: 3 question workers, 6 in-flight requests** (the script's default).
It runs for long stretches with no 429s, and the occasional burst is absorbed by
the retry backoff (up to 8 attempts, honoring `Retry-After`). This is the setting to
use for unattended runs.

**What the numbers mean:**

- 429 is a soft limit. Even at 8/16, every rate-limited call succeeded within 3
  attempts and nothing was lost. Higher concurrency was about 2.5x faster in
  wall time despite the retries.
- At 8/16 the API rejects about a third of all requests, and the log is mostly
  retry lines. If the provider tightens its limits, the run could exhaust its 8
  retries and stop. (It resumes when rerun, but it would need babysitting.)
- A middle ground, 4–5 questions / 8–10 requests, is likely fine if speed matters.
  Lower to 2 / 4 if retries start reaching the 5th attempt.
- The Pi is never the bottleneck. 32 concurrent HTTPS requests use only ~33 MB
  RAM, and the CPU is idle while waiting on the API.

**How to set it:**

```bash
bash scripts/run_debate_gating_baseline.sh                                 # default 3 / 6
WORKERS=4 MAX_CONCURRENCY=8 bash scripts/run_debate_gating_baseline.sh     # faster
WORKERS=2 MAX_CONCURRENCY=4 bash scripts/run_debate_gating_baseline.sh     # if 429s persist
```

Check the retry depth during a run:
`grep -o 'retry [0-9]*' <log> | sort | uniq -c`
