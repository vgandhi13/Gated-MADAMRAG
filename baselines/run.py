"""Reproducible, resumable evaluation of all three baselines."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from baselines.data import DATASETS, ROOT, load_examples, sha256
from baselines.scoring import parse_answers, score_answers, summarize

BASELINES = ['no_rag', 'standard_rag', 'madam_rag']


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def code_hash():
    files = sorted(p for p in (ROOT / 'baselines').glob('*.py') if not p.name.startswith('test_'))
    files += [ROOT / 'baselines' / d / f for d, f in [
        ('NoRAG', 'run_no_rag.py'), ('StandardRAG', 'run_standard_rag.py'),
        ('MADAMRAG', 'run_madam_rag.py')]]
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def read_checkpoint(path):
    if not path.exists():
        return []
    rows = []
    # Recover only an incomplete last write after preemption/time limit. Any
    # malformed complete line is an error, not silently skipped data.
    with path.open('r+b') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                stream.truncate(start)
                break
            rows.append(json.loads(line))
    return rows


def evaluate(args):
    from baselines.madam_adapter import run_original_batch
    from baselines.NoRAG.run_no_rag import build_prompt as no_rag_prompt
    from baselines.StandardRAG.run_standard_rag import build_prompt as rag_prompt
    from baselines.inference import Backend, configure_for_available_gpu
    if args.dataset_path and len(args.datasets) != 1:
        raise ValueError('--dataset-path requires exactly one dataset')
    gpu_profile = configure_for_available_gpu(args) if args.auto_gpu_profile else None
    inference_settings = {
        'gpu_memory_gb': args.gpu_memory_gb,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_num_seqs': args.max_num_seqs,
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'batch_size': args.batch_size, 'madam_batch_size': args.madam_batch_size,
        'automatic_gpu_profile': gpu_profile,
    }
    backend = None
    for dataset in args.datasets:
        path = Path(args.dataset_path or DATASETS[dataset]).resolve()
        examples = load_examples(dataset, path)
        if args.limit:
            examples = examples[:args.limit]
        selected = [e for e in examples if e['index'] % args.num_shards == args.shard_index]
        for baseline in args.baselines:
            run_dir = args.output_dir / dataset / baseline
            if args.num_shards > 1:
                run_dir /= f'shard-{args.shard_index:03d}-of-{args.num_shards:03d}'
            run_dir.mkdir(parents=True, exist_ok=True)
            model_dir = Path(args.model)
            config = {
                'schema_version': 1, 'dataset': dataset, 'baseline': baseline,
                'input_path': str(path), 'input_sha256': sha256(path),
                'model': str(model_dir.resolve()),
                'model_config_sha256': sha256(model_dir / 'config.json'),
                'model_weight_index_sha256': sha256(model_dir / 'model.safetensors.index.json'),
                'tokenizer_sha256': sha256(model_dir / 'tokenizer_config.json'),
                'code_sha256': code_hash(), 'seed': args.seed,
                'decoding': {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1,
                             'max_new_tokens': args.max_new_tokens, 'enable_thinking': False},
                'max_model_len': args.max_model_len, 'num_rounds': args.num_rounds,
                'passages': 'all supplied passages, original order; no truncation',
                'num_shards': args.num_shards, 'shard_index': args.shard_index,
                'limit': args.limit, 'expected_examples': len(selected),
            }
            manifest_path = run_dir / 'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest['config'] != config:
                    raise ValueError(f'Configuration changed: {manifest_path}; use a new output directory')
            else:
                import importlib.metadata
                manifest = {'config': config, 'created_at': datetime.now(timezone.utc).isoformat(),
                            'python': platform.python_version(),
                            'inference': inference_settings,
                            'packages': {
                                p: importlib.metadata.version(p)
                                for p in ['vllm', 'torch', 'transformers']}, 'jobs': []}
            job = os.environ.get('SLURM_JOB_ID')
            if job not in manifest['jobs']:
                manifest['jobs'].append(job)
            manifest.setdefault('attempts', []).append({
                'slurm_job_id': job, 'started_at': datetime.now(timezone.utc).isoformat(),
                'inference': inference_settings,
            })
            write_json(manifest_path, manifest)
            results_path = run_dir / 'predictions.jsonl'
            rows = read_checkpoint(results_path)
            ids = [r['id'] for r in rows]
            expected = {e['id'] for e in selected}
            if len(set(ids)) != len(ids) or not set(ids) <= expected:
                raise ValueError(f'Unexpected or duplicate saved IDs in {results_path}')
            remaining = [e for e in selected if e['id'] not in set(ids)]
            print(f'{dataset}/{baseline}: {len(rows)} saved, {len(remaining)} remaining', flush=True)
            if remaining and backend is None:
                backend = Backend(args)
            batch_size = args.madam_batch_size if baseline == 'madam_rag' else args.batch_size
            with results_path.open('a', buffering=1) as output:
                for start in range(0, len(remaining), batch_size):
                    batch = remaining[start:start + batch_size]
                    begin = time.monotonic()
                    if baseline == 'madam_rag':
                        results = run_original_batch(batch, backend, args.num_rounds)
                    else:
                        prompts = [no_rag_prompt(e['question']) if baseline == 'no_rag' else
                                   rag_prompt(e['question'], e['documents']) for e in batch]
                        calls = backend.generate(prompts, baseline)
                        results = ((e, {'prediction': call['text'], 'calls': [call],
                                         'rounds': 0, 'converged': False})
                                   for e, call in zip(batch, calls))
                    for e, result in results:
                        elapsed = time.monotonic() - begin
                        answers, valid = parse_answers(result['prediction'])
                        calls = result['calls']
                        input_tokens = sum(c['input_tokens'] for c in calls)
                        output_tokens = sum(c['output_tokens'] for c in calls)
                        row = {
                            'id': e['id'], 'index': e['index'], 'question': e['question'],
                            'dataset': dataset, 'baseline': baseline,
                            'document_count': len(e['documents']),
                            'documents_used': 0 if baseline == 'no_rag' else len(e['documents']),
                            'gold_answers': e['gold_answers'], 'wrong_answers': e['wrong_answers'],
                            **result, 'predicted_answers': answers,
                            'scores': score_answers(answers, e, dataset, valid),
                            'input_tokens': input_tokens, 'output_tokens': output_tokens,
                            'total_tokens': input_tokens + output_tokens, 'llm_calls': len(calls),
                            'batch_seconds_per_example': elapsed / len(batch), 'slurm_job_id': job,
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + '\n')
                        output.flush()
                        os.fsync(output.fileno())
                        rows.append(row)
                        write_json(run_dir / 'progress.json', {
                            'completed': len(rows), 'expected': len(selected),
                            'updated_at': datetime.now(timezone.utc).isoformat(),
                        })
                        if baseline == 'madam_rag':
                            print(f'{dataset}/{baseline}: saved {len(rows)}/{len(selected)} '
                                  f'({e["id"]}, {len(calls)} calls)', flush=True)
                    print(f'{dataset}/{baseline}: {len(rows)}/{len(selected)}, '
                          f'{elapsed:.1f}s batch, last input/output tokens '
                          f'{input_tokens}/{output_tokens}', flush=True)
                    # Also make partially completed jobs inspectable.
                    write_json(run_dir / 'progress.json', {
                        'completed': len(rows), 'expected': len(selected),
                        'updated_at': datetime.now(timezone.utc).isoformat(),
                        'metrics': summarize(rows),
                    })
            if len(rows) != len(selected) or {r['id'] for r in rows} != expected:
                raise ValueError(f'Incomplete run: {run_dir}')
            metrics = summarize(rows)
            write_json(run_dir / 'metrics.json', metrics)
            print(json.dumps({'dataset': dataset, 'baseline': baseline, **metrics}), flush=True)


def main(default_baseline=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baselines', nargs='+', choices=BASELINES,
                        default=[default_baseline] if default_baseline else BASELINES)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument('--dataset-path', type=Path)
    parser.add_argument('--model', default=os.environ.get('GATED_MODEL', 'Qwen/Qwen3-8B'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-rounds', '--num_rounds', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--max-model-len', type=int, default=24576)
    parser.add_argument('--max-num-seqs', type=int, default=8)
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.93)
    parser.add_argument('--gpu-memory-gb', type=float, default=21.0)
    parser.add_argument('--auto-gpu-profile', action=argparse.BooleanOptionalAction, default=True,
                        help='Automatically reduce batch/memory settings on 24 GB GPUs')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--madam-batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--enforce-eager', action='store_true')
    args = parser.parse_args()
    if (not 0 <= args.shard_index < args.num_shards or args.num_rounds < 1 or
            args.max_new_tokens < 1 or args.batch_size < 1 or args.madam_batch_size < 1 or
            (args.limit is not None and args.limit < 1)):
        parser.error('Invalid shard, rounds, batch size, token limit, or example limit')
    if 'madam_rag' in args.baselines and (args.num_rounds < 2 or args.max_new_tokens != 128):
        parser.error('Unmodified MADAM-RAG requires >=2 rounds and uses 128 tokens per call')
    evaluate(args)


if __name__ == '__main__':
    main()
