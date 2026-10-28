# Debate Gating baseline: Qwen3-8B results

Setup: SiliconFlow `Qwen/Qwen3-8B`, temperature 0, thinking off, 128 output tokens per call,
gate threshold 95, at most 3 rounds. 500 questions per dataset. Raw numbers are in
`<dataset>/debate_gating/metrics.json`.

| | RAMDocs | TriviaQA (top 5) |
|---|---:|---:|
| Accuracy (strict exact match) | **23.8%** | **56.0%** |
| Precision / Recall / F1 | 0.63 / 0.71 / 0.65 | 0.69 / 0.84 / 0.73 |
| Any-alias EM (conventional TriviaQA metric) | – | 83.8% |
| Misinformation included | 33.0% | – (no labels) |
| Stopped after round 1 (no debate) | 97.2% | 94.2% |
| Mean LLM calls / tokens per question | 6.9 / 2,906 | 6.5 / 2,895 |
| Calls cut off at 128 tokens | 18 | 24 |
| Total cost | $0.087 | $0.087 |
| Mean time per question | 11.6 s | 10.4 s |

## What "misinformation included" means

RAMDocs plants documents with false answers (`wrong_answers`) in 61% of questions. A
question counts as "misinformation included" when any of those false answers appears in
the final answer list. That happened in **33%** of questions. In 93% of those, the
correct answer is in the list too. So the fuse step usually *finds* the truth but keeps
the planted answer next to it instead of rejecting it, for example `["1902", "1885",
"July 10, 1850"]` when only 1902 is correct. TriviaQA has no misinformation labels, so
its 0 means "not measured".

## Why exact-match accuracy is low

A question is correct only if the predicted answer set exactly equals the gold set after
normalization (lowercase, no punctuation or articles). No partial credit, no paraphrases.

RAMDocs failures (% of all 500 questions; categories are rough string heuristics):

| Reason | Share | Example (gold → predicted) |
|---|---:|---|
| Included a planted false answer | 33.6% | `1902` → `1902, 1885, July 10, 1850` |
| Missed one of several correct answers | 22.2% | `Australia, UK` → `Australia` |
| Right answer, different wording | 12.4% | `3,559 people` → `3,559` |
| All gold answers plus extra non-gold ones | 5.6% | |
| Wrong or empty | 2.4% | |

80% of RAMDocs questions have 2–3 gold answers, and accuracy falls from 30% (1 answer)
to 15.5% (3 answers).

On TriviaQA, aliases are alternative spellings of *one* answer. Strict accuracy fails
whenever the model adds any non-alias answer. That is why accuracy is 56% while the
usual any-alias EM is 83.8%:

| Reason | Share | Example |
|---|---:|---|
| Correct alias plus an extra answer | 27.8% | `Liverpool` + `Middlesbrough` |
| Right answer, no exact alias match | 8.6% | `Caribbean Sea` vs `Caribbean` |
| Wrong | 7.6% | |

## Why ~97% of questions had no debate

The gate debates only if the mean confidence of answering agents is below 95. In round 1,
**99.2% of RAMDocs agents (97.9% on TriviaQA) reported exactly 95**. The prompt's scale
says 95–100 means "stated explicitly in the document", and each agent *does* find its
answer stated explicitly in its own document, even when that document is the planted
one. So the average lands at 95, and the gate passes.

Confidence is judged per document, not across documents. Agents disagreed in round 1 on
90% of RAMDocs questions (59% on TriviaQA), yet none of them could see that from inside
their own document. The gate therefore saves tokens (about 6.5 calls per question,
versus at least `2n + 1` for MADAM-RAG) but almost never runs the debate that would
filter misinformation. Self-reported confidence alone is a weak gate for this task.
