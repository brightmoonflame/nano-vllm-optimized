"""Prefill CUDA Graph benchmark — single-request, no queueing.

Submits one request at a time and measures end-to-end latency (≈ prefill time
since output is 1 token and there is no queueing). Compares prefill CUDA Graph
ON vs OFF across multiple prompt lengths.
"""
import gc
import os
import statistics
import time

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

PROMPT_LENGTHS = [1024, 2048, 4096, 8192]
NUM_WARMUP = 2
NUM_REPEATS = 10


def run_benchmark(path, prompt_len, enable_prefill_cudagraph):
    tag = f"Prefill Graph {'ON' if enable_prefill_cudagraph else 'OFF'}"
    print(f"\n{'='*48}\n {tag} — prompt {prompt_len} tokens\n{'='*48}")

    llm = LLM(path, enforce_eager=True,
              enable_prefill_cudagraph=enable_prefill_cudagraph,
              max_model_len=prompt_len + 64)

    prompt = [0] * prompt_len
    sp = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)

    # Warmup
    for _ in range(NUM_WARMUP):
        llm.generate([prompt], sp)

    # Measure
    latencies = []
    for _ in range(NUM_REPEATS):
        t = time.perf_counter()
        llm.generate([prompt], sp, use_tqdm=False)
        latencies.append((time.perf_counter() - t) * 1000)

    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    print(f" Latency (ms): mean={mean_ms:7.1f}  median={median_ms:7.1f}  "
          f"min={min(latencies):7.1f}  max={max(latencies):7.1f}")

    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return median_ms


def main():
    path = os.path.expanduser("/root/model/Qwen3-0.6B/")
    _ = AutoTokenizer.from_pretrained(path)  # ensure model path exists

    results = {}
    for prompt_len in PROMPT_LENGTHS:
        off = run_benchmark(path, prompt_len, enable_prefill_cudagraph=False)
        on = run_benchmark(path, prompt_len, enable_prefill_cudagraph=True)
        speedup = (off - on) / off * 100 if off > 0 else 0
        results[prompt_len] = (off, on, speedup)

    print(f"\n{'='*48}\n Summary\n{'='*48}")
    print(f"{'Prompt':>8}  {'OFF (ms)':>10}  {'ON (ms)':>10}  {'Delta':>8}")
    for plen, (off, on, sp) in results.items():
        delta = on - off
        sign = '+' if delta >= 0 else ''
        print(f"{plen:>8}  {off:>10.1f}  {on:>10.1f}  {sign}{delta:>7.1f}")


if __name__ == "__main__":
    main()
