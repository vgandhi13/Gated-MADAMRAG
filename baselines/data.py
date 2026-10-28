"""Keep model-visible evidence separate from evaluation annotations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    'ramdocs': ROOT / 'data/ramdocs/RAMDocs_test.jsonl',
    'triviaqa': ROOT / 'data/triviaqa_top5/TriviaQA_fid_test_500.jsonl',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_examples(dataset, path=None):
    path = Path(path or DATASETS[dataset])
    examples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index = len(examples)
        source_docs = row['documents'] if dataset == 'ramdocs' else row['ctxs']
        docs = [{'id': str(d.get('id', i)), 'title': d.get('title', ''), 'text': d['text']}
                for i, d in enumerate(source_docs)]
        gold = row['gold_answers'] if dataset == 'ramdocs' else row['answers']
        if not row['question'].strip() or not gold or not docs:
            raise ValueError(f'Incomplete example {index} in {path}')
        examples.append({
            'id': str(row.get('id', f'{dataset}:{index:04d}')), 'index': index,
            'question': row['question'], 'documents': docs,
            'gold_answers': gold, 'wrong_answers': row.get('wrong_answers', []),
        })
    if len({e['id'] for e in examples}) != len(examples):
        raise ValueError('Duplicate example IDs')
    return examples


def render_document(document):
    title = document['title']
    return (f'Title: {title}\n' if title else '') + document['text']
