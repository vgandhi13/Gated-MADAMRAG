"""Merge complete shards and report quality and cost relative to MADAM-RAG."""
import argparse
import csv
import json
from pathlib import Path

from baselines.data import load_examples
from baselines.run import BASELINES, write_json
from baselines.scoring import summarize


def collect(root, dataset, baseline):
    directory = root / dataset / baseline
    shards = sorted(directory.glob('shard-*/manifest.json'))
    manifests = shards or [directory / 'manifest.json']
    rows, configs = [], []
    for manifest in manifests:
        config = json.loads(manifest.read_text())['config']
        rows_path = manifest.parent / 'predictions.jsonl'
        shard_rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        if len(shard_rows) != config['expected_examples']:
            raise ValueError(f'Incomplete run: {manifest.parent}')
        rows.extend(shard_rows)
        configs.append(config)
    reference = {k: v for k, v in configs[0].items()
                 if k not in {'shard_index', 'expected_examples'}}
    for config in configs:
        if {k: v for k, v in config.items()
                if k not in {'shard_index', 'expected_examples'}} != reference:
            raise ValueError(f'Inconsistent shard configuration: {directory}')
    if len(configs) != reference['num_shards']:
        raise ValueError(f'Missing shards: {directory}')
    expected = load_examples(dataset, reference['input_path'])
    if reference['limit']:
        expected = expected[:reference['limit']]
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != {e['id'] for e in expected}:
        raise ValueError(f'Missing or duplicate examples: {directory}')
    rows.sort(key=lambda r: r['index'])
    if shards:
        with (directory / 'predictions.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        write_json(directory / 'metrics.json', summarize(rows))
    return summarize(rows), reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--datasets', nargs='+', choices=['ramdocs', 'triviaqa'],
                        default=['ramdocs', 'triviaqa'])
    args = parser.parse_args()
    summaries = []
    for dataset in args.datasets:
        collected = {b: collect(args.output_dir, dataset, b) for b in BASELINES}
        # All systems must use the same data, model, code, and decoding protocol.
        checks = ['input_sha256', 'model', 'model_config_sha256', 'model_weight_index_sha256',
                  'tokenizer_sha256', 'code_sha256', 'decoding', 'seed', 'max_model_len', 'limit']
        for key in checks:
            if any(collected[b][1][key] != collected['madam_rag'][1][key] for b in BASELINES):
                raise ValueError(f'Baseline protocol mismatch for {dataset}: {key}')
        madam_cost = collected['madam_rag'][0]['mean_total_tokens']
        for baseline in BASELINES:
            metrics = collected[baseline][0]
            summaries.append({'dataset': dataset, 'baseline': baseline, **metrics,
                              'relative_tokens_to_madam': metrics['mean_total_tokens'] / madam_cost})
    write_json(args.output_dir / 'summary.json', summaries)
    with (args.output_dir / 'summary.csv').open('w') as stream:
        fields = sorted(set().union(*(s.keys() for s in summaries)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        '| Dataset | Baseline | N | Acc. | Prec. | Rec. | F1 | Mean input tokens | Mean output tokens | Relative cost |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for s in summaries:
        lines.append(f"| {s['dataset']} | {s['baseline']} | {s['examples']} | "
                     f"{100*s['accuracy']:.2f} | {100*s['precision']:.2f} | "
                     f"{100*s['recall']:.2f} | {100*s['f1']:.2f} | "
                     f"{s['mean_input_tokens']:.1f} | {s['mean_output_tokens']:.1f} | "
                     f"{s['relative_tokens_to_madam']:.4f} |")
    report = '\n'.join(lines) + '\n'
    (args.output_dir / 'summary.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
