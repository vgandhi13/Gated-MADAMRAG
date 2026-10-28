#!/usr/bin/env python3
"""Build a seeded TriviaQA test sample using FiD's official passage selections.

Run through Slurm: the source Wikipedia corpus is several GB compressed.
Only passages needed by the sample are retained in memory and exported.
"""

import argparse
import collections
import csv
import datetime
import functools
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tarfile
import time
import unicodedata
import urllib.request


SOURCES = {
    "dataindex.tar.gz": "https://dl.fbaipublicfiles.com/FiD/data/dataindex.tar.gz",
    "tqa_passages.tar.gz": "https://dl.fbaipublicfiles.com/FiD/data/tqa_passages.tar.gz",
    "triviaqa-unfiltered.tar.gz": "https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz",
    "psgs_w100.tsv.gz": "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz",
}


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def download(cache, name):
    path = cache / name
    receipt = cache / (name + ".download.json")
    if path.exists() and receipt.exists():
        metadata = json.loads(receipt.read_text())
        if path.stat().st_size == metadata["bytes"]:
            log(f"Reusing {path}")
            return path, metadata
        raise ValueError(f"Cached file size differs from receipt: {path}")
    url = SOURCES[name]
    temporary = path.with_name(path.name + ".partial")
    for attempt in range(1, 4):
        offset = temporary.stat().st_size if temporary.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            log(f"Downloading {name}, attempt {attempt}, offset {offset:,}")
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    offset = 0
                if offset:
                    assert response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                size = response.headers.get("Content-Length")
                expected = offset + int(size) if size else None
                last_report = time.monotonic()
                with temporary.open("ab" if offset else "wb") as stream:
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        stream.write(chunk)
                        if time.monotonic() - last_report > 30:
                            log(f"{name}: {stream.tell():,} bytes downloaded")
                            last_report = time.monotonic()
                if expected is not None and temporary.stat().st_size != expected:
                    raise OSError(f"Incomplete download: expected {expected} bytes")
                metadata = {
                    "source_url": url,
                    "resolved_url": response.url,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    **fingerprint(temporary),
                }
            temporary.replace(path)
            write_json(receipt, metadata)
            log(f"Downloaded {name}: {metadata['bytes']:,} bytes")
            return path, metadata
        except Exception as error:
            log(f"Download attempt {attempt} failed: {error}")
            if attempt == 3:
                raise
            time.sleep(2 * attempt)


def archive_json(path, suffix):
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(suffix):
                log(f"Reading {path.name}:{member.name}")
                with archive.extractfile(member) as stream:
                    return json.load(stream)
    raise ValueError(f"No member ending {suffix!r} in {path}")


@functools.lru_cache(maxsize=8192)
def character_class(character):
    return unicodedata.category(character)[0]


def tokenize(text):
    """Equivalent to FiD/DPR's L/N/M word tokens and single non-Z/non-C tokens.

    Unicode NFD normalization and lowercasing match FiD src/evaluation.py.
    Uses Unicode categories directly so the third-party regex package is optional.
    """
    tokens, word = [], []
    for character in unicodedata.normalize("NFD", text):
        category = character_class(character)
        if category in "LNM":
            word.append(character)
        else:
            if word:
                tokens.append("".join(word).lower())
                word = []
            if category not in "ZC":
                tokens.append(character.lower())
    if word:
        tokens.append("".join(word).lower())
    return tokens


def answer_presence(answers, contexts):
    aliases = ["\0" + "\0".join(tokenize(answer)) + "\0" for answer in answers]
    hits = []
    for context in contexts:
        text = "\0" + "\0".join(tokenize(context["text"])) + "\0"
        hits.append(any(alias in text for alias in aliases))
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-answer-string", action="store_true",
        help="Use rejection sampling to require a reference alias in at least one supplied passage.",
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Submit this download/corpus scan through Slurm.")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name in ("dataindex.tar.gz", "tqa_passages.tar.gz", "triviaqa-unfiltered.tar.gz"):
        _, provenance[name] = download(args.cache_dir, name)
    test_indices = archive_json(args.cache_dir / "dataindex.tar.gz", "TQA.test.idx.json")
    passage_selections = archive_json(args.cache_dir / "tqa_passages.tar.gz", "tqa_passages/test.json")
    source_questions = archive_json(args.cache_dir / "triviaqa-unfiltered.tar.gz", "unfiltered-web-dev.json")["Data"]
    assert len(test_indices) == len(set(test_indices))
    assert 0 < args.sample_size <= len(test_indices)
    rng = random.Random(args.seed)
    sampled_indices = rng.sample(range(len(test_indices)), args.sample_size)
    candidate_indices = sampled_indices.copy()
    if args.require_answer_string:
        # Continue the same uniform random draw without replacement. Accept the
        # first sample_size answer-present questions, preserving existing valid
        # questions when changing an unfiltered run with the same seed.
        initial_indices = set(sampled_indices)
        remaining_indices = [i for i in range(len(test_indices)) if i not in initial_indices]
        candidate_indices += rng.sample(remaining_indices, min(args.sample_size, len(remaining_indices)))
    examples, needed_ids = [], set()
    full_counts = collections.Counter()
    full_question_ids = []
    for source_index in test_indices:
        question_id = source_questions[source_index]["QuestionId"]
        full_question_ids.append(question_id)
        full_counts[len(passage_selections[question_id])] += 1
    assert len(full_question_ids) == len(set(full_question_ids))
    for test_index in candidate_indices:
        original_index = test_indices[test_index]
        source = source_questions[original_index]
        question_id = source["QuestionId"]
        selected_ids = passage_selections[question_id]
        assert selected_ids and len(selected_ids) == len(set(selected_ids))
        target = source["Answer"]["Value"]
        examples.append({
            "id": question_id,
            "fid_test_index": test_index,
            "original_dev_index": original_index,
            "question": source["Question"],
            "answers": source["Answer"]["Aliases"],
            "target": target.title() if target.isupper() else target,
            "ctxs": [{"id": passage_id} for passage_id in selected_ids],
        })
        needed_ids.update(selected_ids)
    del source_questions, passage_selections
    log(f"FiD test: {len(test_indices):,} questions; passage count distribution: {dict(full_counts)}")
    log(f"Selected {len(examples)} candidate questions, requiring {len(needed_ids):,} unique Wikipedia passages")
    corpus, provenance["psgs_w100.tsv.gz"] = download(args.cache_dir, "psgs_w100.tsv.gz")
    passages, remaining = {}, set(needed_ids)
    csv.field_size_limit(sys.maxsize)
    scanned, start = 0, time.monotonic()
    with gzip.open(corpus, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        assert header == ["id", "text", "title"], header
        for row in reader:
            scanned += 1
            passage_id = row[0]
            if passage_id in remaining:
                assert len(row) == 3
                passages[passage_id] = {"id": passage_id, "text": row[1], "title": row[2]}
                remaining.remove(passage_id)
            if scanned % 1_000_000 == 0:
                log(f"Scanned {scanned:,} passages; found {len(passages):,}/{len(needed_ids):,}")
            if not remaining:
                break
    if remaining:
        raise ValueError(f"Missing {len(remaining)} selected passage IDs; examples: {sorted(remaining)[:5]}")
    log(f"Corpus scan finished in {time.monotonic() - start:.1f} seconds")
    diagnostics, coverage = [], collections.Counter()
    sample_counts = collections.Counter()
    for example in examples:
        example["ctxs"] = [passages[context["id"]].copy() for context in example["ctxs"]]
        sample_counts[len(example["ctxs"])] += 1
        hits = answer_presence(example["answers"], example["ctxs"])
        first_hit = next((i + 1 for i, hit in enumerate(hits) if hit), None)
        diagnostics.append({
            "id": example["id"],
            "has_answer_string": any(hits),
            "answer_containing_passages": sum(hits),
            "first_answer_rank": first_hit,
            "passage_has_answer_string": hits,
        })
        for k in (1, 5, 10, 20, 50, 100):
            coverage[k] += any(hits[:k])
    eligible_candidates = sum(diagnostic["has_answer_string"] for diagnostic in diagnostics)
    if args.require_answer_string:
        accepted = [
            (example, diagnostic) for example, diagnostic in zip(examples, diagnostics)
            if diagnostic["has_answer_string"]
        ][:args.sample_size]
        if len(accepted) != args.sample_size:
            raise ValueError(
                f"Only {len(accepted)} eligible questions in {len(candidate_indices)} candidates; "
                "no existing output files have been changed."
            )
        examples, diagnostics = map(list, zip(*accepted))
        sampled_indices = [example["fid_test_index"] for example in examples]
        needed_ids = {context["id"] for example in examples for context in example["ctxs"]}
        sample_counts = collections.Counter(len(example["ctxs"]) for example in examples)
        coverage = collections.Counter({
            k: sum(any(diagnostic["passage_has_answer_string"][:k]) for diagnostic in diagnostics)
            for k in (1, 5, 10, 20, 50, 100)
        })
        assert all(diagnostic["has_answer_string"] for diagnostic in diagnostics)
        log(f"Accepted {len(examples)} of {eligible_candidates} eligible candidates; coverage@100=100%")
    assert len(examples) == len({example["id"] for example in examples}) == args.sample_size
    assert len({example["question"] for example in examples}) == args.sample_size
    assert all(example["answers"] and all(context["text"] for context in example["ctxs"]) for example in examples)
    stem = f"TriviaQA_fid_test_{args.sample_size}"
    previous_archive = None
    previous_manifest = args.output_dir / "manifest.json"
    if previous_manifest.exists():
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = args.output_dir / "archive" / stamp
        archive.mkdir(parents=True)
        old_summary = json.loads(previous_manifest.read_text())
        for name in [*old_summary["files"], "manifest.json", "README.md"]:
            source = args.output_dir / name
            if source.exists():
                os.link(source, archive / name)
        previous_archive = str(archive.relative_to(args.output_dir))
        log(f"Preserved previous sample in {archive}")
    files = {}
    for suffix, records in ((".jsonl", examples), (".answer_presence.jsonl", diagnostics)):
        path = args.output_dir / (stem + suffix)
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
        files[path.name] = fingerprint(path)
    path = args.output_dir / (stem + ".json")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(examples, stream, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)
    files[path.name] = fingerprint(path)
    summary = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": "FiD TriviaQA",
        "split": "FiD test (selected from original TriviaQA unfiltered web dev)",
        "full_test_questions": len(test_indices),
        "full_test_passage_count_distribution": dict(sorted(full_counts.items())),
        "sample_size": args.sample_size,
        "random_seed": args.seed,
        "sampling": (
            "Seeded random draw without replacement: draw sample_size initial indices, then up to "
            "sample_size additional indices uniformly from remaining test indices using the same RNG; "
            "retain the first sample_size candidates containing a reference answer alias in their supplied passages."
            if args.require_answer_string else
            "Python random.Random(seed).sample(range(full_test_questions), sample_size); without replacement; random order; no answer-presence filtering"
        ),
        "require_answer_string": args.require_answer_string,
        "candidate_fid_test_indices": candidate_indices,
        "candidate_questions_evaluated": len(candidate_indices),
        "candidate_questions_with_answer_string": eligible_candidates,
        "previous_sample_archive": previous_archive,
        "sampled_fid_test_indices": sampled_indices,
        "sampled_question_ids": [example["id"] for example in examples],
        "sample_passage_count_distribution": dict(sorted(sample_counts.items())),
        "total_passage_occurrences": sum(len(example["ctxs"]) for example in examples),
        "unique_passages": len(needed_ids),
        "answer_string_coverage": {str(k): {"questions": coverage[k], "fraction": coverage[k] / len(examples)} for k in sorted(coverage)},
        "answer_presence_method": "FiD/DPR Unicode-NFD, lowercase token-sequence matching against answers aliases in passage text only; Unicode L/N/M word tokens and single non-Z/non-C tokens. This checks answer strings, not factual support.",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cache_directory": str(args.cache_dir.resolve()),
        "script_sha256": fingerprint(Path(__file__))["sha256"],
        "upstream_preparation": "https://github.com/facebookresearch/FiD/blob/main/src/preprocess.py",
        "upstream_evaluation": "https://github.com/facebookresearch/FiD/blob/main/src/evaluation.py",
        "sources": provenance,
        "files": files,
    }
    write_json(args.output_dir / "manifest.json", summary)
    sampling_description = (
        f"{len(examples)} questions sampled from the {len(test_indices):,}-question FiD test split using Python "
        f"random seed {args.seed}, without replacement. Candidates were drawn in random order; the first "
        f"{args.sample_size} questions with a reference answer alias in at least one of their supplied passages "
        f"were retained. Every retained question has answer-string coverage among its 100 passages. "
        f"This is an answer-present subset, so its retrieval coverage is not representative of the full test set."
        if args.require_answer_string else
        f"{len(examples)} questions sampled uniformly without replacement from the {len(test_indices):,}-question "
        f"FiD test split using Python random seed {args.seed}. No answer-presence filtering was applied."
    )
    readme = f"""# FiD TriviaQA test sample

{sampling_description} The FiD test split is selected from the original TriviaQA unfiltered web development split using the official TQA.test.idx.json file.

- `{stem}.jsonl`: one question per line, with every officially supplied passage in its original order.
- `{stem}.json`: the same records as a JSON array for FiD-style loaders.
- `{stem}.answer_presence.jsonl`: separate evaluation annotations, joined by `id`.
- `manifest.json`: source URLs and SHA-256 hashes, selected question IDs and indices, sampling procedure, counts, and answer-string coverage.

Each main record contains `id` (original TriviaQA QuestionId), `fid_test_index` (zero-based), `original_dev_index` (zero-based), `question`, `answers` (reference aliases), `target` (official answer, with FiD's uppercase-title conversion), and `ctxs` (ordered passages with `id`, `title`, and `text`). Passage IDs and question IDs are different identifiers.

Passage count distribution in the sample: {dict(sorted(sample_counts.items()))}. Total passage occurrences: {summary['total_passage_occurrences']:,}. Unique passage IDs: {len(needed_ids):,}.

Answer-string coverage, as questions with at least one matching passage among the first k:

| k | Questions | Percentage |
|---|---:|---:|
"""
    for k in sorted(coverage):
        readme += f"| {k} | {coverage[k]}/{len(examples)} | {100 * coverage[k] / len(examples):.1f}% |\n"
    readme += """
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
"""
    if previous_archive:
        readme += f"\nThe previous sample is preserved in `{previous_archive}/`.\n"
    temporary_readme = args.output_dir / "README.md.partial"
    temporary_readme.write_text(readme, encoding="utf-8")
    temporary_readme.replace(args.output_dir / "README.md")
    log(f"Finished: {args.output_dir}; coverage@100={coverage[100]}/{len(examples)}")


if __name__ == "__main__":
    main()
