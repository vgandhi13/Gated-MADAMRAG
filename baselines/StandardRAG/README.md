# Standard RAG

One Qwen3-8B generation over the question and **all supplied passages**, concatenated
in their saved order. Implements the Standard RAG baseline in Section 4 of the
project proposal. No retriever, embeddings, FAISS index, DPR/BART checkpoint,
fine-tuning, or passage conversion is needed.

The implementation is in `run_standard_rag.py`. Data loading, inference, scoring,
and Slurm launchers are shared; see [../README.md](../README.md).

From the repository root:

```bash
sbatch scripts/run_baselines.sbatch --baselines standard_rag \
  --datasets ramdocs triviaqa \
  --output-dir "$GATED_WORK/results/qwen3-8b"
```
