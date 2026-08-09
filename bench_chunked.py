"""Chunked prefill benchmark — adapted from NKNaN/nano-vllm-chunked-prefill.

Runs 8 requests with ~950-token prompts, compares TTFT/TBT with chunked
prefill ON vs OFF. Uses enforce_eager for fair comparison.
"""
import gc
import os
import statistics
import time

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

LONG_PASSAGE = """\
Large language models (LLMs) represent one of the most significant advances in \
artificial intelligence over the past decade. These models, built on transformer \
architectures first introduced in the seminal "Attention Is All You Need" paper by \
Vaswani et al. in 2017, have demonstrated remarkable capabilities across a wide range \
of natural language understanding and generation tasks. From answering complex questions \
to writing code, translating languages, and engaging in nuanced dialogue, modern LLMs \
have transformed how humans interact with computer systems.

The key innovation behind transformers is the self-attention mechanism, which allows \
the model to weigh the importance of different tokens in a sequence when producing \
its output. This mechanism, combined with multi-head attention, positional encodings, \
and feed-forward neural networks, enables transformers to capture long-range \
dependencies and parallelize computation effectively. The scaling properties of \
transformers have proven exceptional, with models ranging from hundreds of millions \
to over a trillion parameters showing consistent improvements in capability as model \
size, data, and compute increase.

Inference optimization has become a critical area of research as LLMs grow in size \
and complexity. KV caching, which stores previously computed key-value pairs to avoid \
recomputation during autoregressive generation, is fundamental to efficient inference. \
PagedAttention, inspired by operating system virtual memory management, organizes KV \
cache into fixed-size blocks, reducing fragmentation and enabling flexible memory \
allocation. Continuous batching, another key technique, dynamically inserts new \
requests into the processing batch as others complete, maximizing GPU utilization.

Chunked prefill addresses a specific challenge: when a long prompt arrives, its \
prefill computation can block the decode of already-running requests, causing \
head-of-line blocking. By splitting prefill into smaller chunks and interleaving \
them with decode steps, chunked prefill reduces latency for queued requests while \
maintaining high throughput. This technique is particularly valuable in serving \
scenarios with mixed prompt lengths and concurrent request arrivals.

Tensor parallelism splits model weights across multiple GPUs, enabling inference of \
models too large for a single device. Pipeline parallelism, by contrast, partitions \
model layers across devices, with micro-batches flowing through the pipeline. Both \
approaches have trade-offs in communication overhead, latency, and throughput, and \
the choice between them depends on the specific deployment scenario and hardware \
configuration available.
"""

QUESTIONS = [
    "Please summarize the following article about LLM serving systems in 3 sentences",
    "What is KV caching and why is it important according to the following text",
    "Explain the concept of continuous batching as described in the following article",
    "What is chunked prefill and what problem does it solve based on the following text",
    "List all the optimization techniques mentioned in the following article",
    "What is PagedAttention according to the following text",
    "How does tensor parallelism differ from pipeline parallelism based on the following article",
    "What trend has characterized LLM development according to the following text",
]


def make_prompts(tokenizer):
    raw = [f"{q}:\n\n{LONG_PASSAGE}" for q in QUESTIONS]
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in raw
    ]


def run_benchmark(path, prompts, sampling_params, enable_chunked_prefill, prefill_chunk_size=512):
    tag = (f"Chunked Prefill ON (chunk_size={prefill_chunk_size})"
           if enable_chunked_prefill else "Chunked Prefill OFF")
    print(f"\n{'='*56}\n {tag}\n{'='*56}")

    llm = LLM(path, enforce_eager=True,
              enable_chunked_prefill=enable_chunked_prefill,
              prefill_chunk_size=prefill_chunk_size)

    n = len(prompts)
    ttfts = [None] * n
    last_times = [None] * n
    tbts = [[] for _ in range(n)]

    t_start = time.perf_counter()
    for index, new_token_ids, finished in llm.generate_stream(prompts, sampling_params, use_tqdm=True):
        t = time.perf_counter()
        if ttfts[index] is None:
            ttfts[index] = (t - t_start) * 1000
        else:
            tbts[index].append((t - last_times[index]) * 1000)
        last_times[index] = t

    all_tbts = [x for seq_tbts in tbts for x in seq_tbts]
    print(f"\n TTFT (ms): mean={statistics.mean(ttfts):7.1f} "
          f"p50={statistics.median(ttfts):7.1f} "
          f"p99={sorted(ttfts)[-1]:7.1f}")
    if all_tbts:
        print(f" TPOT (ms): mean={statistics.mean(all_tbts):7.1f} "
              f"p50={statistics.median(all_tbts):7.1f} "
              f"p99={sorted(all_tbts)[-1]:7.1f}")
    print()

    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main():
    path = os.path.expanduser("/root/model/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompts = make_prompts(tokenizer)
    token_counts = [len(tokenizer.encode(p)) for p in prompts]
    print(f"Batch : {len(prompts)} requests")
    print(f"Tokens: min={min(token_counts)} max={max(token_counts)} "
          f"mean={statistics.mean(token_counts):.0f}")

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    run_benchmark(path, prompts, sampling_params, enable_chunked_prefill=False)
    run_benchmark(path, prompts, sampling_params,
                  enable_chunked_prefill=True, prefill_chunk_size=512)


if __name__ == "__main__":
    main()
