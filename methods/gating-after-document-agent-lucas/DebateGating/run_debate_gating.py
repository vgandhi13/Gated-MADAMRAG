"""Confidence-gated debate (adapted from DebateCoder, arXiv:2601.21469) over SiliconFlow.

One agent per document gives a short answer, the evidence sentence behind it, and a
confidence score. If the mean confidence of answering agents reaches the gate, the
answers are fused immediately; otherwise agents check each other's evidence and
re-score, up to max rounds.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.data import DATASETS, load_examples, render_document, sha256
from baselines.run import read_checkpoint, write_json
from baselines.scoring import normalize_answer, parse_answers, score_answers, summarize

API_URL = os.environ.get('SILICONFLOW_BASE_URL', 'https://api.siliconflow.com/v1') + '/chat/completions'

CONFIDENCE_SCALE = """Confidence means how likely your answer is correct:
- 95-100: the document states the answer explicitly, and it clearly matches the question.
- 70-94: the answer is supported but implied, partial, or the document may be about a different entity.
- below 70: weak support, or the document may be wrong.
Use 95 or more only when you would bet on the answer."""

REPLY_FORMAT = """Reply in exactly this format:
Answer: <short answer, or unknown if the document does not answer the question>
Evidence: <the sentence from your document that supports the answer, or none>
Confidence: <integer 0-100>"""

ANSWER_PROMPT = """You are an agent reading one retrieved document to answer a question. The document may be relevant, irrelevant, or contain misinformation, and the question may be ambiguous (the same name can refer to different entities).

Question: {question}
Document: {document}

Answer using only this document.
{scale}

{format}"""

DEBATE_PROMPT = """You are an agent reading one retrieved document to answer a question. The document may be relevant, irrelevant, or contain misinformation, and the question may be ambiguous (the same name can refer to different entities).

Question: {question}
Document: {document}

Your previous reply:
{own}

Replies from other agents, each reading a different document:
{peers}

Compare their evidence with yours. Keep your answer if your document supports it, even if others disagree: a different answer may be about a different entity. Change it only if your evidence is weak or clearly contradicted by stronger evidence. Then re-rate your confidence.
{scale}

{format}"""

FUSE_PROMPT = """You are combining replies from agents that each read a different document for the same question.

Question: {question}
Agent replies:
{replies}

Return every distinct correct answer: keep answers about different entities that share the question's name, merge different wordings of the same answer into one, and drop unknown answers and answers whose evidence is contradicted by stronger evidence (misinformation). If no answer is supported, return an empty list.
Reply in exactly this format:
All Correct Answers: ["answer 1", "answer 2"]. Explanation: <one sentence>."""


def parse_reply(text):
    answer = re.search(r'Answer\s*:\s*(.*)', text, flags=re.I)
    evidence = re.search(r'Evidence\s*:\s*(.*)', text, flags=re.I)
    confidence = re.search(r'Confidence\s*:\s*(\d+(?:\.\d+)?)', text, flags=re.I)
    # A missing confidence is treated as 0 so it can only trigger more debate.
    return {'answer': answer.group(1).strip().rstrip('.') if answer else 'unknown',
            'evidence': evidence.group(1).strip() if evidence else '',
            'confidence': min(float(confidence.group(1)), 100.0) if confidence else 0.0}


def is_unknown(answer):
    return normalize_answer(answer) in {'', 'unknown', 'none', 'n/a'}


def format_reply(i, r):
    return (f'Agent {i + 1}: Answer: {r["answer"]}. Evidence: {r["evidence"]} '
            f'Confidence: {r["confidence"]:g}.')


def gate_confidence(replies):
    # Agents whose document does not answer abstain from the gate; if every agent
    # abstains there is nothing to debate.
    scores = [r['confidence'] for r in replies if not is_unknown(r['answer'])]
    return sum(scores) / len(scores) if scores else 100.0


def debate_gating(example, llm, threshold=95.0, max_rounds=3):
    question = example['question']
    docs = [render_document(d) for d in example['documents']]
    calls, rounds = [], []
    outputs = llm([ANSWER_PROMPT.format(question=question, document=d, scale=CONFIDENCE_SCALE,
                                        format=REPLY_FORMAT) for d in docs],
                  'debate_gating/answer')
    calls += outputs
    replies = [parse_reply(o['text']) for o in outputs]
    while True:
        confidence = gate_confidence(replies)
        rounds.append({'replies': replies, 'gate_confidence': confidence})
        if confidence >= threshold or len(rounds) >= max_rounds:
            break
        prompts = [DEBATE_PROMPT.format(
            question=question, document=d, own=format_reply(i, replies[i]),
            peers='\n'.join(format_reply(j, r) for j, r in enumerate(replies) if j != i),
            scale=CONFIDENCE_SCALE, format=REPLY_FORMAT)
            for i, d in enumerate(docs)]
        outputs = llm(prompts, 'debate_gating/debate')
        calls += outputs
        replies = [parse_reply(o['text']) for o in outputs]
    fused = llm([FUSE_PROMPT.format(
        question=question, replies='\n'.join(format_reply(i, r) for i, r in enumerate(replies)))],
        'debate_gating/fuse')[0]
    calls.append(fused)
    return {'prediction': fused['text'], 'calls': calls, 'debate': rounds,
            'rounds': len(rounds), 'converged': rounds[-1]['gate_confidence'] >= threshold}


class ChatAPI:
    """Minimal OpenAI-compatible client. All HTTP requests share one bounded pool."""
    def __init__(self, args):
        self.key = os.environ.get('SILICONFLOW_API_KEY')
        if not self.key:
            raise RuntimeError('SILICONFLOW_API_KEY is not set')
        self.args = args
        self.pool = ThreadPoolExecutor(max_workers=args.max_concurrency)

    def complete(self, prompt, stage):
        a = self.args
        body = {'model': a.model, 'messages': [{'role': 'user', 'content': prompt}],
                'temperature': a.temperature, 'max_tokens': a.max_new_tokens,
                'seed': a.seed, 'enable_thinking': a.enable_thinking}
        if a.enable_thinking:
            body['thinking_budget'] = a.thinking_budget
        request = urllib.request.Request(
            API_URL, data=json.dumps(body).encode(),
            headers={'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'})
        begin = time.monotonic()
        for attempt in range(a.retries):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = json.loads(response.read())
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                status = getattr(exc, 'code', None)
                # Retry rate limits, server errors, and network failures only.
                if (status is not None and status != 429 and status < 500) or \
                        attempt == a.retries - 1:
                    raise RuntimeError(f'{stage}: API error {exc}: '
                                       f'{exc.read().decode() if status else ""}') from exc
                wait = float(exc.headers.get('Retry-After') or 0) if status else 0
                wait = max(wait, min(60, 2 ** attempt))
                print(f'{stage}: retry {attempt + 1} in {wait:.0f}s after {exc}', flush=True)
                time.sleep(wait)
        choice, usage = data['choices'][0], data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        return {'stage': stage, 'text': (choice['message'].get('content') or '').strip(),
                'input_tokens': input_tokens, 'output_tokens': output_tokens,
                'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get(
                    'reasoning_tokens', usage.get('reasoning_tokens', 0)),
                'cost_usd': (input_tokens * a.input_price + output_tokens * a.output_price) / 1e6,
                'seconds': time.monotonic() - begin,
                'finish_reason': choice.get('finish_reason'),
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}

    def __call__(self, prompts, stage):
        return list(self.pool.map(lambda p: self.complete(p, stage), prompts))


def evaluate(args):
    llm = ChatAPI(args)
    for dataset in args.datasets:
        path = Path(DATASETS[dataset]).resolve()
        examples = load_examples(dataset, path)[:args.limit]
        run_dir = args.output_dir / dataset / 'debate_gating'
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {'dataset': dataset, 'baseline': 'debate_gating', 'api_url': API_URL,
                  'model': args.model, 'input_sha256': sha256(path),
                  'threshold': args.threshold, 'max_rounds': args.max_rounds,
                  'max_new_tokens': args.max_new_tokens, 'temperature': args.temperature,
                  'enable_thinking': args.enable_thinking,
                  'thinking_budget': args.thinking_budget if args.enable_thinking else None,
                  'seed': args.seed, 'limit': args.limit,
                  'price_usd_per_million_tokens': {'input': args.input_price,
                                                   'output': args.output_price}}
        manifest_path = run_dir / 'manifest.json'
        manifest = {'config': config, 'sessions': []}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['config'] != config:
                raise ValueError(f'Configuration changed: {manifest_path}; use a new output directory')
        session = {'started_at': datetime.now(timezone.utc).isoformat(),
                   'workers': args.workers, 'max_concurrency': args.max_concurrency}
        manifest['sessions'].append(session)
        write_json(manifest_path, manifest)
        results_path = run_dir / 'predictions.jsonl'
        rows = read_checkpoint(results_path)
        done = {r['id'] for r in rows}
        remaining = [e for e in examples if e['id'] not in done]
        print(f'{dataset}: {len(rows)} saved, {len(remaining)} remaining', flush=True)
        begin = time.monotonic()

        def run_one(example):
            start = time.monotonic()
            return debate_gating(example, llm, args.threshold, args.max_rounds), \
                time.monotonic() - start

        with results_path.open('a') as output, ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(run_one, e): e for e in remaining}
            for future in as_completed(futures):
                e, (result, seconds) = futures[future], future.result()
                answers, valid = parse_answers(result['prediction'])
                calls = result['calls']
                input_tokens = sum(c['input_tokens'] for c in calls)
                output_tokens = sum(c['output_tokens'] for c in calls)
                row = {'id': e['id'], 'index': e['index'], 'question': e['question'],
                       'dataset': dataset, 'baseline': 'debate_gating',
                       'document_count': len(e['documents']),
                       'gold_answers': e['gold_answers'], 'wrong_answers': e['wrong_answers'],
                       **result, 'predicted_answers': answers,
                       'scores': score_answers(answers, e, dataset, valid),
                       'input_tokens': input_tokens, 'output_tokens': output_tokens,
                       'total_tokens': input_tokens + output_tokens,
                       'reasoning_tokens': sum(c['reasoning_tokens'] for c in calls),
                       'llm_calls': len(calls), 'cost_usd': sum(c['cost_usd'] for c in calls),
                       'query_seconds': seconds}
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                rows.append(row)
                print(f'{dataset}: {len(rows)}/{len(examples)} ({e["id"]}, {row["rounds"]} rounds, '
                      f'{row["llm_calls"]} calls, {seconds:.1f}s, ${row["cost_usd"]:.5f})', flush=True)
        session.update(finished_at=datetime.now(timezone.utc).isoformat(),
                       wall_seconds=time.monotonic() - begin, queries=len(remaining))
        write_json(manifest_path, manifest)
        metrics = summarize(rows)
        metrics.update(
            gate_pass_rate=sum(r['converged'] for r in rows) / len(rows),
            early_stop_rate=sum(r['rounds'] == 1 for r in rows) / len(rows),
            mean_reasoning_tokens=sum(r['reasoning_tokens'] for r in rows) / len(rows),
            total_cost_usd=sum(r['cost_usd'] for r in rows),
            mean_cost_usd=sum(r['cost_usd'] for r in rows) / len(rows),
            mean_query_seconds=sum(r['query_seconds'] for r in rows) / len(rows),
            wall_seconds=sum(s.get('wall_seconds', 0) for s in manifest['sessions']))
        write_json(run_dir / 'metrics.json', metrics)
        print(json.dumps({'dataset': dataset, **metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument('--model', default=os.environ.get('SILICONFLOW_MODEL', 'Qwen/Qwen3-8B'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'results' / 'Qwen3-8B')
    parser.add_argument('--threshold', type=float, default=95.0)
    parser.add_argument('--max-rounds', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--enable-thinking', action='store_true',
                        help='Qwen3 thinking mode (off by default, as in the local baselines)')
    parser.add_argument('--thinking-budget', type=int, default=1024)
    parser.add_argument('--input-price', type=float, default=0.06, help='USD per 1M input tokens')
    parser.add_argument('--output-price', type=float, default=0.06, help='USD per 1M output tokens')
    parser.add_argument('--workers', type=int, default=3, help='Questions run concurrently')
    parser.add_argument('--max-concurrency', type=int, default=6,
                        help='Maximum in-flight API requests across all questions')
    parser.add_argument('--retries', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if (args.max_rounds < 1 or args.workers < 1 or args.max_concurrency < 1 or
            not 0 <= args.threshold <= 100):
        parser.error('Invalid rounds, workers, concurrency, or threshold')
    if args.enable_thinking and args.max_new_tokens <= args.thinking_budget:
        parser.error('With thinking, --max-new-tokens must exceed --thinking-budget')
    evaluate(args)


if __name__ == '__main__':
    main()
