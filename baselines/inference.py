"""Shared greedy Qwen3 backend, with actual token accounting for every call."""
import hashlib
import json
import os


def apply_gpu_profile(args, total_memory_gib):
    """Resolve scheduling settings without changing prompts or decoding."""
    fields = ['batch_size', 'madam_batch_size', 'max_num_seqs',
              'max_num_batched_tokens', 'gpu_memory_gb', 'gpu_memory_utilization']
    requested = {key: getattr(args, key) for key in fields}
    if total_memory_gib <= 26:
        args.batch_size = min(args.batch_size, 8)
        args.madam_batch_size = min(args.madam_batch_size, 8)
        args.max_num_seqs = min(args.max_num_seqs, 8)
        args.max_num_batched_tokens = min(args.max_num_batched_tokens, 2048)
        args.gpu_memory_gb = min(args.gpu_memory_gb, 21.0)
        # L4 needs sufficient KV cache for the unchanged 24,576-token limit.
        args.gpu_memory_utilization = 0.93
        profile = 'L4 / 24 GB'
    else:
        profile = 'L40S / larger GPU'
    return {'profile': profile, 'device_memory_gib': total_memory_gib,
            'requested': requested,
            'resolved': {key: getattr(args, key) for key in fields}}


def configure_for_available_gpu(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU profiling must run inside a Slurm allocation')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.cuda.get_device_properties(0)
    profile = apply_gpu_profile(args, device.total_memory / 1024**3)
    profile['device_name'] = device.name
    print(json.dumps({'automatic_gpu_profile': profile}), flush=True)
    return profile


class Backend:
    def __init__(self, args):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('GPU inference must run inside a Slurm allocation')
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.max_model_len = args.max_model_len
        self.max_new_tokens = args.max_new_tokens
        device = torch.cuda.get_device_properties(0)
        utilization = min(args.gpu_memory_utilization,
                          args.gpu_memory_gb * 1024**3 / device.total_memory)
        print(json.dumps({'gpu': device.name,
                          'device_memory_gib': device.total_memory / 1024**3,
                          'inference_memory_budget_gib': utilization * device.total_memory / 1024**3}),
              flush=True)
        self.llm = LLM(
            model=args.model, tokenizer=args.model, dtype='bfloat16',
            seed=args.seed, tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=utilization,
            enable_prefix_caching=True, enforce_eager=args.enforce_eager,
            disable_log_stats=True, generation_config='vllm',
        )
        self.sampling = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1,
            max_tokens=args.max_new_tokens, seed=args.seed,
        )

    def generate(self, prompts, stage):
        token_ids = [self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False) for prompt in prompts]
        for ids in token_ids:
            if len(ids) + self.max_new_tokens > self.max_model_len:
                raise ValueError(f'{stage}: {len(ids)} prompt tokens plus '
                                 f'{self.max_new_tokens} output exceed context '
                                 f'{self.max_model_len}; evidence is never silently truncated')
        outputs = self.llm.generate([{'prompt_token_ids': ids} for ids in token_ids],
                                    self.sampling, use_tqdm=False)
        calls = []
        for prompt, ids, output in zip(prompts, token_ids, outputs):
            completion = output.outputs[0]
            calls.append({
                'stage': stage, 'text': completion.text.strip(),
                'input_tokens': len(output.prompt_token_ids),
                'output_tokens': len(completion.token_ids),
                'finish_reason': completion.finish_reason,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            })
        return calls
