"""Run the unmodified upstream MADAM-RAG function with shared inference.

No code, prompts, parsing, convergence checks, or aggregation logic inside
baselines/MADAMRAG is patched or replaced.
"""
from concurrent.futures import Future, ThreadPoolExecutor
import queue
import threading
import time

from baselines.data import render_document


class PipelineAdapter:
    """Implement only the text-generation pipeline interface used by upstream."""
    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    def __call__(self, messages, max_new_tokens=128, top_p=None, do_sample=False):
        if max_new_tokens != self.backend.max_new_tokens or do_sample:
            raise ValueError('Upstream generation settings must match shared greedy decoding')
        if len(messages) != 1 or messages[0]['role'] != 'user':
            raise ValueError('Unexpected upstream message format')
        prompt = messages[0]['content']
        stage = 'aggregator' if prompt.startswith('You are an aggregator') else 'agent'
        call = self.backend.generate([prompt], f'madam_rag/{stage}')[0]
        self.calls.append(call)
        return [{'generated_text': messages + [{'role': 'assistant', 'content': call['text']}]}]


def run_original_debate(example, backend, num_rounds=3):
    # Import upstream directly. Its main() model-loading routine is not invoked.
    from baselines.MADAMRAG.run_madam_rag import multi_agent_debate
    generator = PipelineAdapter(backend)
    begin = time.monotonic()
    records = multi_agent_debate(
        example['question'], [render_document(d) for d in example['documents']],
        generator, num_rounds=num_rounds)
    if records['final_aggregation'] is None:
        raise ValueError('Original MADAM-RAG returned no final aggregate. '
                         'The released implementation requires at least two rounds.')
    round_keys = [k for k in records if k.startswith('round')]
    # The original skips aggregation in the final round only on convergence.
    last = records[round_keys[-1]]
    return {'prediction': records['final_aggregation'], 'calls': generator.calls,
            'debate': records, 'rounds': len(round_keys),
            'converged': 'aggregation' not in last,
            'query_wall_seconds': time.monotonic() - begin}


class QueuedBackend:
    def __init__(self, requests, max_new_tokens):
        self.requests = requests
        self.max_new_tokens = max_new_tokens
        self.error = None
        self.lock = threading.Lock()

    def generate(self, prompts, stage):
        future = Future()
        with self.lock:
            if self.error is not None:
                raise self.error
            self.requests.put((prompts, stage, future))
        return future.result()

    def fail(self, error):
        with self.lock:
            self.error = error
            while True:
                try:
                    _, _, future = self.requests.get_nowait()
                    future.set_exception(error)
                except queue.Empty:
                    break


def run_original_batch(examples, backend, num_rounds=3):
    """Batch independent queries while retaining each original sequential debate.

    The main thread alone calls vLLM. Each question's original function blocks on
    its own next generation, so no agent sees another question's responses and
    dependencies within a debate remain exactly as in upstream.
    """
    requests = queue.Queue()
    queued = QueuedBackend(requests, backend.max_new_tokens)
    pool = ThreadPoolExecutor(max_workers=len(examples))
    futures = {pool.submit(run_original_debate, e, queued, num_rounds): e for e in examples}
    pending_requests = []
    try:
        while futures:
            done = [future for future in futures if future.done()]
            for future in done:
                example = futures.pop(future)
                yield example, future.result()
            if not futures:
                break
            try:
                pending_requests = [requests.get(timeout=0.01)]
            except queue.Empty:
                continue
            # Briefly collect independent queries' next calls into one GPU batch.
            deadline = time.monotonic() + 0.005
            while len(pending_requests) < len(futures):
                try:
                    pending_requests.append(requests.get(timeout=max(0.0, deadline-time.monotonic())))
                except queue.Empty:
                    break
            prompts = [p for ps, _, _ in pending_requests for p in ps]
            calls = backend.generate(prompts, 'madam_rag')
            offset = 0
            for ps, stage, future in pending_requests:
                subset = calls[offset:offset + len(ps)]
                for call in subset:
                    call['stage'] = f'{stage}/{offset}'
                future.set_result(subset)
                offset += len(ps)
            pending_requests = []
    except BaseException as exc:
        # Unblock all original-debate threads on an inference failure.
        queued.fail(exc)
        for _, _, future in pending_requests:
            if not future.done():
                future.set_exception(exc)
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
