"""Deterministic normalized exact answer matching; no LLM judge."""
import ast
from collections import Counter
import json
import re
import string
import unicodedata


def normalize_answer(text):
    text = unicodedata.normalize('NFD', text).lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def parse_answers(text):
    # Only the answer field is scored, never the explanation (which may quote
    # rejected misinformation). A malformed answer is a scored format failure.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    match = re.search(r'All Correct Answers\s*:\s*(.*)', text, flags=re.I | re.S)
    field = match.group(1).strip() if match else text
    if field.startswith('```'):
        field = re.sub(r'^```(?:json)?\s*', '', field)
    if re.match(r'^unknown\b', field, flags=re.I):
        return [], True
    if not field.startswith('['):
        return [], False
    # raw_decode tolerates the trailing explanation without silently accepting
    # an incomplete (output-token-limited) list.
    try:
        answers, _ = json.JSONDecoder().raw_decode(field)
    except (ValueError, TypeError):
        end = field.find(']')
        if end < 0:
            return [], False
        try:
            answers = ast.literal_eval(field[:end + 1])
        except (ValueError, SyntaxError):
            return [], False
    if not isinstance(answers, list) or any(not isinstance(a, str) for a in answers):
        return [], False
    result = []
    seen = set()
    for answer in answers:
        key = normalize_answer(answer)
        if key and key not in {'unknown', 'none', 'n/a'} and key not in seen:
            result.append(answer.strip())
            seen.add(key)
    return result, True


def token_f1(prediction, gold):
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    return 2 * common / (len(pred) + len(ref))


def score_answers(answers, example, dataset, format_valid=True):
    pred = {normalize_answer(a) for a in answers}
    gold = {normalize_answer(a) for a in example['gold_answers']}
    wrong = {normalize_answer(a) for a in example['wrong_answers']} - gold
    if dataset == 'ramdocs':
        true_positive = len(pred & gold)
        precision = true_positive / len(pred) if pred else 0.0
        recall = true_positive / len(gold)
        accuracy = float(pred == gold)
    else:
        # TriviaQA aliases describe ONE answer, not independent required facts.
        true_positive = int(bool(pred & gold))
        false_positive = len(pred - gold)
        precision = true_positive / (true_positive + false_positive) if pred else 0.0
        recall = float(true_positive)
        accuracy = float(bool(pred) and pred <= gold)
    if not format_valid:
        precision = recall = accuracy = 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'misinformation': float(bool(pred & wrong)), 'format_valid': bool(format_valid),
        'triviaqa_em': float(bool(pred & gold)) if dataset == 'triviaqa' else None,
        'triviaqa_token_f1': max((token_f1(a, g) for a in answers
                                for g in example['gold_answers']), default=0.0)
        if dataset == 'triviaqa' else None,
    }


def summarize(rows):
    if not rows:
        raise ValueError('No results to summarize')
    result = {'examples': len(rows)}
    for key in ['accuracy', 'precision', 'recall', 'f1', 'misinformation',
                'format_valid', 'triviaqa_em', 'triviaqa_token_f1']:
        values = [r['scores'][key] for r in rows if r['scores'][key] is not None]
        if values:
            result[key] = sum(values) / len(values)
    for key in ['input_tokens', 'output_tokens', 'total_tokens', 'llm_calls', 'rounds']:
        result['mean_' + key] = sum(r[key] for r in rows) / len(rows)
    result['total_input_tokens'] = sum(r['input_tokens'] for r in rows)
    result['total_output_tokens'] = sum(r['output_tokens'] for r in rows)
    result['length_limited_calls'] = sum(
        c['finish_reason'] == 'length' for r in rows for c in r['calls'])
    return result
