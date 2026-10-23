"""TensorRT-LLM offline max-throughput benchmark: nvidia/Llama-3.1-8B-Instruct-FP8, TP 1,
3000 synthetic requests of ISL 2048 / OSL 128 tokens (trtllm-bench throughput, PyTorch backend).

Neutral form of the task: a plain script, runnable on a machine with 1 GPU
(`python tasks/tensorrt_llm/llama31_8b_fp8_isl2048_osl128.py`); the model by its public id. KWARGS are
the call arguments. The expected forecast and its sources are in llama31_8b_fp8_isl2048_osl128.json;
the form a service takes is in variants/<service>/tensorrt_llm/llama31_8b_fp8_isl2048_osl128.py.

Source (the measured workload):
- github.com/NVIDIA/TensorRT-LLM, commit 0c9430e (2025-09-15), docs/source/performance/perf-overview.md: section "Throughput Measurements" (lines 17-24), table "Llama 3.1 8B FP8"
  (lines 90-107), row ISL 2048, OSL 128, TP 1; hardware list (lines 26-32).
- Commands (lines 177-178, 200, 235; TP via $tp_size, line 186, and
  trtllm-bench --tp, throughput.py:193; options file lines 240-259):
  python benchmarks/cpp/prepare_dataset.py --tokenizer=nvidia/Llama-3.1-8B-Instruct-FP8 --stdout
      token-norm-dist --num-requests=3000 --input-mean=2048 --output-mean=128
      --input-stdev=0 --output-stdev=0 > dataset.txt
  trtllm-bench --model nvidia/Llama-3.1-8B-Instruct-FP8 throughput --dataset dataset.txt
      --backend pytorch --tp 1 --extra_llm_api_options llm_options.yml
- num_requests 3000 for ISL 2048 / OSL 128: table at lines 215-226.

This file is the smallest faithful reduction of that run to one file; every
choice follows the source (paths relative to the repository at 0c9430e):
- requests: 3000 prompts of exactly 2048 random token ids from the tokenizer's
  vocabulary, each generating exactly 128 tokens (prepare_dataset.py with
  stdev 0; benchmarks/cpp/utils/utils.py:156-161 samples ids from
  range(vocab_size)); sampling_params.max_tokens = the request's output
  length (tensorrt_llm/bench/benchmark/utils/asynchronous.py:53); end_id and
  pad_id -1, i.e. no early stop on EOS (throughput.py:129-133, 414-417);
- engine: tensorrt_llm.LLM with the PyTorch backend (bench/benchmark/
  __init__.py:7, 110-112), tensor_parallel_size 1, tokenizer init skipped,
  KV cache free_gpu_memory_fraction 0.90 (throughput.py:98-103 default; the
  page notes 0.95 "in many cases", line 262), CUDA graphs with padding
  and the batch sizes of llm_options.yml (lines 240-259);
- all requests submitted at once, no concurrency limit (throughput.py:
  220-224 default); 2 warmup requests first (throughput.py:171-175,
  426-445).

Differences from the source, none of which changes the measured work:
max_batch_size and max_num_tokens are left to the LLM API defaults instead of
trtllm-bench's heuristics (bench/benchmark/utils/general.py:77-160) — they
change the batching, not the number of tokens computed; the requests are
generated in the script instead of read from a dataset file; no reports
written.
"""

KWARGS = {"num_requests": 3000, "isl": 2048, "osl": 128, "tp": 1, "warmup": 2}


def bench_llama31_8b_fp8_isl2048_osl128(num_requests: int = 3000, isl: int = 2048, osl: int = 128, tp: int = 1,
                                        warmup: int = 2):
    # requires: pip install tensorrt_llm (the v0.21 benchmarks, perf-overview.md:239)
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


if __name__ == "__main__":
    print(bench_llama31_8b_fp8_isl2048_osl128(**KWARGS))
