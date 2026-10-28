"""CPU-only checks for evidence isolation, evaluation, and debate bookkeeping."""
import ast
import re
import string
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.data import load_examples
from baselines.madam_adapter import run_original_debate, run_original_batch
from baselines.NoRAG.run_no_rag import build_prompt as no_rag_prompt
from baselines.StandardRAG.run_standard_rag import build_prompt as rag_prompt
from baselines.run import read_checkpoint
from baselines.inference import apply_gpu_profile
from baselines.scoring import parse_answers, score_answers


class FakeBackend:
    max_new_tokens = 128

    def __init__(self, changed=False):
        self.changed = changed
        self.agent_calls = 0

    def generate(self, prompts, stage):
        calls = []
        for prompt in prompts:
            if prompt.startswith('You are an aggregator'):
                text = 'All Correct Answers: ["Paris"]. Explanation: Not London.'
            else:
                answer = f'answer-{self.agent_calls // 2}' if self.changed else 'Paris'
                self.agent_calls += 1
                text = f'Answer: {answer}. Explanation: Evidence supports this.'
            calls.append({'text': text, 'stage': stage, 'input_tokens': 10,
                          'output_tokens': 5, 'finish_reason': 'stop'})
        return calls


class BaselineTests(unittest.TestCase):
    def setUp(self):
        # Execute the exact original function definitions without importing its
        # GPU dependencies, so the bookkeeping tests remain CPU/stdlib-only.
        source = Path(__file__).parent / 'MADAMRAG/run_madam_rag.py'
        tree = ast.parse(source.read_text())
        functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                               type_ignores=[])
        module = types.ModuleType('baselines.MADAMRAG.run_madam_rag')
        module.__dict__.update(re=re, string=string, List=list)
        exec(compile(functions, str(source), 'exec'), module.__dict__)
        self.module_patch = patch.dict(sys.modules, {module.__name__: module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_ramdocs_requires_all_gold(self):
        example = {'gold_answers': ['Paris', 'Rome'], 'wrong_answers': ['London']}
        self.assertEqual(score_answers(['Paris'], example, 'ramdocs')['accuracy'], 0)
        self.assertEqual(score_answers(['Paris'], example, 'ramdocs')['recall'], .5)
        self.assertEqual(score_answers(['Paris', 'Rome'], example, 'ramdocs')['accuracy'], 1)
        wrong = score_answers(['Paris', 'Rome', 'London'], example, 'ramdocs')
        self.assertEqual(wrong['accuracy'], 0)
        self.assertEqual(wrong['misinformation'], 1)
        self.assertAlmostEqual(wrong['precision'], 2/3)

    def test_triviaqa_aliases_are_alternatives(self):
        example = {'gold_answers': ['Winston Churchill', 'Sir Winston Churchill'], 'wrong_answers': []}
        score = score_answers(['Winston Churchill'], example, 'triviaqa')
        self.assertEqual(score['accuracy'], 1)
        self.assertEqual(score['recall'], 1)
        score = score_answers(['Winston Churchill', 'Attlee'], example, 'triviaqa')
        self.assertEqual(score['accuracy'], 0)
        self.assertEqual(score['precision'], .5)
        self.assertEqual(score['triviaqa_em'], 1)

    def test_score_only_answer_field(self):
        parsed, valid = parse_answers(
            'All Correct Answers: ["Paris"]. Explanation: London is misinformation.')
        self.assertTrue(valid)
        self.assertEqual(parsed, ['Paris'])

    def test_parse_failures_and_unknown(self):
        for text in ['All Correct Answers: ["Paris"', 'The answer is Paris', '[1963]']:
            self.assertEqual(parse_answers(text), ([], False))
        for text in ['unknown', 'All Correct Answers: []', 'All Correct Answers: ["unknown"]']:
            self.assertEqual(parse_answers(text), ([], True))
        self.assertEqual(parse_answers('["Paris", "The Paris"]'), (['Paris'], True))
        self.assertEqual(parse_answers("All Correct Answers: ['Paris']. Explanation: ..."), (['Paris'], True))

    def test_no_gold_or_annotation_leakage(self):
        docs = [{'id': 'private-id', 'title': 'Title', 'text': 'Visible passage',
                 'type': 'misinfo', 'answer': 'SECRET', 'has_answer': True}]
        prompt = rag_prompt('Visible question?', docs)
        self.assertIn('Visible passage', prompt)
        for private in ['private-id', 'SECRET', 'misinfo', 'has_answer']:
            self.assertNotIn(private, prompt)
        self.assertNotIn('Visible passage', no_rag_prompt('Visible question?'))

    def test_every_supplied_document_is_used_in_order(self):
        docs = [{'title': '', 'text': f'passage-{i}'} for i in range(100)]
        prompt = rag_prompt('Question?', docs)
        positions = [prompt.index(f'Document {i+1}: passage-{i}\n')
                     if i < 99 else prompt.index('Document 100: passage-99') for i in range(100)]
        self.assertEqual(positions, sorted(positions))

    def test_original_madam_one_round_limitation_is_preserved(self):
        docs = [{'title': '', 'text': 'Paris'}] * 2
        with self.assertRaisesRegex(ValueError, 'Original MADAM-RAG'):
            run_original_debate({'question': 'Capital?', 'documents': docs},
                                FakeBackend(), num_rounds=1)

    def test_madam_convergence_counts_check_round(self):
        docs = [{'title': '', 'text': 'Paris'}] * 2
        result = run_original_debate({'question': 'Capital?', 'documents': docs},
                                     FakeBackend(), num_rounds=3)
        self.assertTrue(result['converged'])
        self.assertEqual(result['rounds'], 2)
        self.assertEqual(len(result['calls']), 5)
        self.assertEqual(sum(c['input_tokens'] for c in result['calls']), 50)

    def test_madam_runs_all_rounds_when_answers_change(self):
        docs = [{'title': '', 'text': 'Paris'}] * 2
        result = run_original_debate({'question': 'Capital?', 'documents': docs},
                                     FakeBackend(changed=True), num_rounds=3)
        self.assertFalse(result['converged'])
        self.assertEqual(result['rounds'], 3)
        self.assertEqual(len(result['calls']), 9)

    def test_concurrent_original_queries_remain_independent(self):
        examples = [{'id': str(i), 'question': 'Capital?',
                     'documents': [{'title': '', 'text': 'Paris'}] * 2} for i in range(3)]
        results = list(run_original_batch(examples, FakeBackend(), 3))
        self.assertEqual({e['id'] for e, _ in results}, {'0', '1', '2'})
        for _, result in results:
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(len(result['calls']), 5)
            self.assertEqual(result['prediction'],
                             'All Correct Answers: ["Paris"]. Explanation: Not London.')

    def test_concurrent_queries_propagate_inference_errors(self):
        class FailedBackend(FakeBackend):
            def generate(self, prompts, stage):
                raise RuntimeError('test inference failure')
        examples = [{'question': 'Capital?',
                     'documents': [{'title': '', 'text': 'Paris'}]}] * 3
        with self.assertRaisesRegex(RuntimeError, 'test inference failure'):
            list(run_original_batch(examples, FailedBackend(), 3))

    def test_resume_recovers_partial_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.jsonl'
            path.write_bytes(b'{"id": "first"}\n{"id": "unfinished')
            self.assertEqual(read_checkpoint(path), [{'id': 'first'}])
            self.assertEqual(path.read_bytes(), b'{"id": "first"}\n')
            path.write_bytes(b'broken complete line\n')
            with self.assertRaises(ValueError):
                read_checkpoint(path)

    def test_real_data_schema_and_counts(self):
        for dataset in ['ramdocs', 'triviaqa']:
            examples = load_examples(dataset)
            self.assertEqual(len(examples), 500)
            for example in examples:
                for doc in example['documents']:
                    self.assertEqual(set(doc), {'id', 'title', 'text'})
                if dataset == 'triviaqa':
                    self.assertEqual(len(example['documents']), 5)

    def test_l4_profile_reduces_only_scheduling_settings(self):
        args = types.SimpleNamespace(batch_size=32, madam_batch_size=32, max_num_seqs=32,
                                     max_num_batched_tokens=4096, gpu_memory_gb=40.0,
                                     gpu_memory_utilization=.90, max_new_tokens=128,
                                     num_rounds=3, max_model_len=24576)
        profile = apply_gpu_profile(args, 22.4)
        self.assertEqual((args.batch_size, args.madam_batch_size, args.max_num_seqs), (8, 8, 8))
        self.assertEqual(args.max_num_batched_tokens, 2048)
        self.assertEqual(args.gpu_memory_gb, 21)
        self.assertEqual(args.gpu_memory_utilization, .93)
        self.assertEqual((args.max_new_tokens, args.num_rounds, args.max_model_len), (128, 3, 24576))
        self.assertEqual(profile['requested']['batch_size'], 32)

    def test_l40s_profile_preserves_large_batches(self):
        args = types.SimpleNamespace(batch_size=32, madam_batch_size=32, max_num_seqs=32,
                                     max_num_batched_tokens=4096, gpu_memory_gb=40.0,
                                     gpu_memory_utilization=.90)
        profile = apply_gpu_profile(args, 44.4)
        self.assertEqual(profile['requested'], profile['resolved'])


if __name__ == '__main__':
    unittest.main()
