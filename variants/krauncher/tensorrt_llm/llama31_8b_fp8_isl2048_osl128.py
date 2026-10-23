"""Krauncher variant of tasks/tensorrt_llm/llama31_8b_fp8_isl2048_osl128.py (the neutral
form, with the source and the reduction it documents); the expected forecast
and its sources are in tasks/tensorrt_llm/llama31_8b_fp8_isl2048_osl128.json.

The task function is the neutral one, unchanged. FUNC is the task, OPTIONS
the arguments of its @client.task decorator, KWARGS the arguments of the
call.
"""

OPTIONS = dict(
    timeout=7200,
    data_urls=['hf://models/nvidia/Llama-3.1-8B-Instruct-FP8'],
    pip=['tensorrt_llm', 'transformers'],
    stream_stderr=True,
)

KWARGS = dict(num_requests=3000, isl=2048, osl=128, tp=1, warmup=2)


def bench_llama31_8b_fp8_isl2048_osl128(num_requests: int = 3000, isl: int = 2048, osl: int = 128, tp: int = 1,
                                        warmup: int = 2):
    # requires: pip install tensorrt_llm (the v0.21 benchmarks, perf-overview.md:238)
    import random
    import time

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    from transformers import AutoTokenizer

    model = "nvidia/Llama-3.1-8B-Instruct-FP8"
    vocab = AutoTokenizer.from_pretrained(model).vocab_size
    rng = random.Random(0)
    prompts = [[rng.randrange(vocab) for _ in range(isl)] for _ in range(num_requests)]

    llm = LLM(model=model, backend="pytorch", tensor_parallel_size=tp,
              skip_tokenizer_init=True,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.90),
              cuda_graph_config=CudaGraphConfig(
                  enable_padding=True,
                  batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512,
                               1024, 2048, 4096, 8192]))
    params = SamplingParams(max_tokens=osl, end_id=-1, pad_id=-1)

    llm.generate(prompts[:warmup], params)
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {"output_tokens": tokens, "elapsed_sec": round(elapsed, 1),
            "output_tokens_per_sec": round(tokens / elapsed, 1)}



FUNC = bench_llama31_8b_fp8_isl2048_osl128
