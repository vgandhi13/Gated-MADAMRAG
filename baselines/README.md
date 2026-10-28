# Fixed-evidence baselines

Run No RAG, Standard RAG, and the original MADAM-RAG on the existing 500 RAMDocs
and 500 FiD TriviaQA examples. The backbone is a local Qwen3-8B checkpoint
(`$GATED_MODEL`); the Python environment is `$GATED_ENV`. Both are configured in
`local/env.local.sh` (see Setup below).

- **No RAG:** one generation using only the question and answer-format instruction.
- **Standard RAG:** one generation using the question and every supplied passage.
- **MADAM-RAG:** calls `MADAMRAG/run_madam_rag.py:multi_agent_debate` directly.
  Every file in `MADAMRAG/` is unchanged. The external `madam_adapter.py` supplies
  the pipeline interface, translates input schemas, and logs calls. Independent
  queries can be batched; each original debate executes its own sequential logic.
  Original prompts, parsing, normalized substring stopping rule, peer-response
  histories, and previous-aggregate reuse on convergence are preserved.

All methods use greedy non-thinking decoding (Qwen3 `enable_thinking=False`), seed
42, BF16 model weights, and at most 128 new tokens per call. The original MADAM-RAG
function hardcodes 128 tokens and needs at least two rounds to return an aggregate;
the runner uses three maximum rounds. The original main() routine's 4-bit loading
is bypassed so all methods use the same model precision and inference backend.
These are local Qwen3-8B experiments, not reproductions of published larger-model scores.

RAMDocs uses all 1–12 documents. TriviaQA uses all 100 passages in their original
order. There is no top-k filtering, reranking, answer-based passage selection,
indexing, or silent context truncation. A prompt exceeding the 24,576-token context
fails explicitly. Only question and passage title/text reach the model; answers,
misinformation labels, disambiguation annotations, and answer-presence labels stay
in evaluation. The TriviaQA sample is already conditioned on answer presence among
100 passages, as documented in `data/triviaqa/README.md`.

## Setup and launch

Personal paths and the Slurm account live in `local/env.local.sh`, which is
gitignored and sourced by `baselines/env.sh`. Run every command from the repository
root. Setup creates `$GATED_ENV` if needed and installs the pinned Qwen3/vLLM stack.
Run installation and GPU inference through Slurm:

```bash
cp local/env.local.sh.example local/env.local.sh   # then edit paths and account
# Download the model once (on a node with internet access), e.g.:
#   huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/models/Qwen3-8B
# Data: RAMDocs ships with MADAM-RAG; TriviaQA is rebuilt by a CPU job.
mkdir -p logs data/ramdocs
cp baselines/MADAMRAG/RAMDocs_test.jsonl data/ramdocs/
source baselines/env.sh                            # exports SBATCH_ACCOUNT, GATED_*
sbatch scripts/prepare_fid_triviaqa.sbatch
sbatch scripts/setup_baselines.sbatch
# After setup completes:
sbatch scripts/run_baselines.sbatch --output-dir "$GATED_WORK/results/qwen3-8b"
```

`data/download_manifest.json` records the SHA-256 of each data file.

For a smoke test, append `--limit 2` and use a distinct output directory. Use
`--baselines no_rag standard_rag` or `--baselines madam_rag` and
`--datasets ramdocs` or `--datasets triviaqa` to select work. Jobs request one L4 or L40S GPU, four CPU cores, and 40 GB host RAM.
Slurm can select an available matching GPU in `gpu`,
`gpu-highvram`, or `gpu-preempt-highvram`; A100s are excluded. Override walltime based on measured runtime.
The partitions and `l4|l40s` constraint match the Unity cluster; adjust them for other clusters.
Small batches and 2,048-token prefill chunks limit memory use; weights remain BF16.
The inference memory budget is capped at 21 GiB even on a larger GPU. To use an
available L40S specifically, pass `--partition=gpu-highvram,gpu-preempt-highvram`
and `--constraint=l40s` to sbatch. To require the smaller L4, pass
`--partition=gpu --constraint=l4`. No A100 is requested.

A CPU audit of every Standard RAG prompt found maxima of 3,063 input tokens on
RAMDocs and 19,496 on TriviaQA. Both fit the 24,576-token budget including 128 output
tokens. The audit is reproducible with `sbatch scripts/check_baseline_contexts.sbatch`.

Large runs may be split with `--num-shards N --shard-index I` (zero based), assigning
examples by input index modulo N. Each query's original document set is unchanged.
Resubmit exactly the same command/output directory to resume saved queries. Run
configuration, input/code/model metadata hashes, package versions, and Slurm job IDs
are recorded. Changes to the protocol require a new output directory. Checkpoints
are flushed per completed query and recover an incomplete final write.

## Evaluation

Answer strings are parsed from `All Correct Answers: [...]`; explanations are
excluded so rejected misinformation mentioned in a rationale does not count as an
answer. Malformed answer formats are retained and score zero. Empty lists or
`unknown` are abstentions. Answers are normalized with Unicode NFD, lowercasing,
ASCII punctuation/article removal, and whitespace normalization, then deduplicated.

RAMDocs accuracy requires an exact normalized match to the complete gold answer
set. Precision/recall/F1 compare predicted and gold sets per query. Extra predictions
are false positives; known misinformation is also logged separately. This is a
strict string-based implementation of the proposal's criterion, not semantic or
LLM-judged matching. Paraphrases and omitted units can be marked incorrect.

TriviaQA aliases are alternatives for one answer. Strict accuracy requires at least
one prediction and no predictions outside the alias set. Recall accepts any alias;
precision penalizes additional non-alias answers. Also report conventional
any-alias EM and best token F1 separately. Dataset metrics are macro averages.

Every call records actual prompt and generated token counts, output text, finish
reason, and prompt hash. Totals include explanations, aggregations, and the
convergence-check round. Prefix caching does not discount logical input-token cost.
Mean input/output/total tokens, call counts, and output-token-limit counts are saved.

```bash
python -m unittest baselines.test_baselines -v
# After all full runs/shards complete, using the shared environment:
source baselines/env.sh
python -m baselines.summarize --output-dir "$GATED_WORK/results/qwen3-8b"
```

Each run writes `manifest.json`, `predictions.jsonl`, `progress.json`, and
`metrics.json`. The summarizer rejects missing/duplicate examples and mismatched
protocols, merges complete shards, and writes `summary.json`, `summary.csv`, and
`summary.md` with token cost relative to MADAM-RAG for each dataset.

All caches, installation files, and outputs live under `$GATED_WORK` (default
`./work`, gitignored); Slurm logs go to `./logs`. Point `$GATED_WORK` at fast
storage on shared clusters.

## Automatic GPU profiles

`--auto-gpu-profile` is enabled by default. A queued job can be eligible for both
L4 and L40S. On a 24 GB GPU the runner resolves the question batch size and active
sequence limit to at most 8, prefill chunks to at most 2,048 tokens, and the memory
budget to at most 21 GiB with 93% device utilization. On L40S the requested larger
settings are retained (the full RAMDocs jobs request batches/sequences of 32,
4,096-token prefill chunks, 40 GiB, and 90% utilization).

Prompts, evidence, 24,576-token context limit, decoding, and debate behavior are
unchanged. Each manifest records requested/resolved settings and the device for
every run attempt, including requeues. Use `--no-auto-gpu-profile` to retain
explicit resource settings without automatic adjustment.
