"""Single-request Dense Prefill CUDA Graph A/B benchmark.

The benchmark intentionally covers only the static prefill case supported by
the engine: one first-pass prompt without a prefix cache, chunked prefill,
mixed batch, or speculative decoding. It compares the same model path eager
vs CUDA Graph across prompt-length buckets.

Use ``--use-triton-attn`` to benchmark the self-written Dense Prefill kernel;
``--kv-quant`` additionally verifies graph compatibility with INT8 KV writes.
The attention calculation of a first Dense Prefill still consumes its freshly
projected BF16 K/V; the INT8 cache is written for subsequent decode steps.
"""
from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare eager and CUDA-Graph Dense Prefill.")
    parser.add_argument("--model", required=True, help="Path to a local Hugging Face model.")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--use-triton-attn", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the self-written Triton Dense Prefill kernel (default: true).")
    parser.add_argument("--kv-quant", action="store_true",
                        help="Use INT8 KV Cache; this benchmarks online INT8 writes during Dense Prefill.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if not Path(args.model).expanduser().is_dir():
        parser.error(f"model directory does not exist: {Path(args.model).expanduser()}")
    if not args.prompt_lengths or any(length <= 0 for length in args.prompt_lengths):
        parser.error("--prompt-lengths must contain positive values")
    if args.num_warmup < 0 or args.num_repeats <= 0:
        parser.error("--num-warmup must be non-negative and --num-repeats must be positive")
    return args


def run_benchmark(args: argparse.Namespace, prompt_len: int, graph_enabled: bool) -> float:
    mode = "CUDA Graph" if graph_enabled else "eager"
    dtype = "INT8 KV write" if args.kv_quant else "BF16 KV write"
    backend = "Triton" if args.use_triton_attn else "flash-attn"
    print(f"\n{'=' * 64}\nDense Prefill {mode} — {backend}, {dtype}, prompt={prompt_len}\n{'=' * 64}")

    # Graph capture is enabled only for the ON side. The measured request emits
    # one token, so its latency is dominated by the long prefill forward.
    llm = LLM(
        args.model,
        enforce_eager=not graph_enabled,
        enable_prefill_cudagraph=graph_enabled,
        use_triton_attn=args.use_triton_attn,
        kv_quant=args.kv_quant,
        max_model_len=prompt_len + 64,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    captured = hasattr(llm.model_runner, "prefill_graphs")
    if graph_enabled and not captured:
        raise RuntimeError("Prefill CUDA Graph was requested but was not captured for this configuration")

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)

    def fresh_prompt(index: int) -> list[int]:
        # Reusing an identical prompt can hit Prefix Cache after the first
        # round, turning the workload into Paged Prefill and bypassing this
        # graph. Change the first token while retaining the same shape.
        return [index + 1] + [0] * (prompt_len - 1)

    for index in range(args.num_warmup):
        llm.generate([fresh_prompt(index)], sampling_params, use_tqdm=False)

    latencies = []
    for index in range(args.num_repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        llm.generate([fresh_prompt(args.num_warmup + index)], sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1e3)

    median = statistics.median(latencies)
    print(
        f"Latency: median={median:.2f}ms  mean={statistics.mean(latencies):.2f}ms  "
        f"min={min(latencies):.2f}ms  max={max(latencies):.2f}ms  graph captured={captured}"
    )
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return median


def main() -> None:
    args = parse_args()
    print(
        "Scope: single-request first Dense Prefill only; Prefix Cache, Paged/Chunked/mixed "
        "Prefill, and speculative decoding intentionally remain eager."
    )
    print(f"model={args.model}  triton={args.use_triton_attn}  kv_quant={args.kv_quant}")
    print(f"{'Prompt':>8}  {'eager (ms)':>12}  {'graph (ms)':>12}  {'speedup':>9}")
    results = []
    for prompt_len in args.prompt_lengths:
        eager_ms = run_benchmark(args, prompt_len, graph_enabled=False)
        graph_ms = run_benchmark(args, prompt_len, graph_enabled=True)
        results.append((prompt_len, eager_ms, graph_ms))
        print(f"{prompt_len:>8}  {eager_ms:>12.2f}  {graph_ms:>12.2f}  {eager_ms / graph_ms:>8.2f}x")

    print(f"\n{'=' * 48}\nSummary\n{'=' * 48}")
    for prompt_len, eager_ms, graph_ms in results:
        print(f"prompt={prompt_len:>5}: {eager_ms:.2f}ms → {graph_ms:.2f}ms ({eager_ms / graph_ms:.2f}x)")


if __name__ == "__main__":
    main()
