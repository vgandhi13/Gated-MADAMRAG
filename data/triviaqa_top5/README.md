# FiD TriviaQA test sample

500 questions sampled from the 11,313-question FiD test split using Python random seed 42, without replacement. Candidates were drawn in random order; the first 500 questions with a reference answer alias in at least one of their supplied passages were retained. Every retained question has answer-string coverage among its first 5 retrieved passages. This is an answer-present subset, so its retrieval coverage is not representative of the full test set. The FiD test split is selected from the original TriviaQA unfiltered web development split using the official TQA.test.idx.json file.

- `TriviaQA_fid_test_500.jsonl`: one question per line, with the first 5 retrieved officially supplied passages in their original order.
- `TriviaQA_fid_test_500.json`: the same records as a JSON array for FiD-style loaders.
- `TriviaQA_fid_test_500.answer_presence.jsonl`: separate evaluation annotations, joined by `id`.
- `manifest.json`: source URLs and SHA-256 hashes, selected question IDs and indices, sampling procedure, counts, and answer-string coverage.

Each main record contains `id` (original TriviaQA QuestionId), `fid_test_index` (zero-based), `original_dev_index` (zero-based), `question`, `answers` (reference aliases), `target` (official answer, with FiD's uppercase-title conversion), and `ctxs` (ordered passages with `id`, `title`, and `text`). Passage IDs and question IDs are different identifiers.

Passage count distribution in the sample: {5: 500}. Total passage occurrences: 2,500. Unique passage IDs: 2,491.

Answer-string coverage, as questions with at least one matching passage among the first k:

| k | Questions | Percentage |
|---|---:|---:|
| 1 | 389/500 | 77.8% |
| 5 | 500/500 | 100.0% |

Matching follows FiD/DPR token-sequence matching (Unicode NFD, lowercase, all reference aliases), using passage text only. An answer-string match is not a correctness or semantic-support label. There are no RAMDocs-style correct/misinfo/noise annotations. Pass only question and passage title/text to the answering model; keep answers, target, and the answer-presence file for evaluation.

Large downloaded sources are cached on persistent /work storage, at the path in manifest.json. The entire Wikipedia corpus was streamed once; only sampled passages were exported. The full test split is not materialized with passage text in this directory.

Reproduce from the gated repository, through Slurm:

```bash
sbatch scripts/prepare_fid_triviaqa.sbatch
```

Sources and preparation:
- https://github.com/facebookresearch/FiD/blob/main/get-data.sh
- https://github.com/facebookresearch/FiD/blob/main/src/preprocess.py
- https://github.com/facebookresearch/FiD/blob/main/src/evaluation.py
