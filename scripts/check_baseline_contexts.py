"""Audit saved evidence prompt lengths without loading model weights."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.data import load_examples
from baselines.StandardRAG.run_standard_rag import build_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report = {}
    for dataset in ['ramdocs', 'triviaqa']:
        lengths = []
        for example in load_examples(dataset):
            ids = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': build_prompt(example['question'], example['documents'])}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            lengths.append({'id': example['id'], 'input_tokens': len(ids)})
        report[dataset] = {'examples': len(lengths), 'max_input_tokens': max(r['input_tokens'] for r in lengths),
                           'mean_input_tokens': sum(r['input_tokens'] for r in lengths) / len(lengths),
                           'examples_exceeding_24576_with_128_output_tokens':
                           [r for r in lengths if r['input_tokens'] + 128 > 24576]}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
