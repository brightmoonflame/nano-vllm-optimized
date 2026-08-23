"""Integration correctness checks for CUDA Graph inference paths.

Compares greedy eager and CUDA-Graph token outputs for the fused Triton INT8
KV configuration. ``--decode-only`` skips Prefill graph capture entirely, so
Decode correctness can be validated independently of Prefill experiments.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


PREFILL_BOUNDARIES = (255, 256, 257)  # default prefill graph boundary: 256 tokens
DECODE_BATCH_SIZES = (3, 5, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check eager/Graph correctness at CUDA-Graph bucket boundaries.")
    parser.add_argument("--model", required=True, help="Path to a local Hugging Face model.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Do not capture or test Dense Prefill CUDA Graphs.",
    )
    args = parser.parse_args()
    if not Path(args.model).expanduser().is_dir():
        parser.error(f"model directory does not exist: {Path(args.model).expanduser()}")
    return args


def prompt(length: int, tag: int) -> list[int]:
    """Same shape, distinct prefix: prevents Prefix Cache from changing the case."""
    return [tag] + [0] * (length - 1)


def run_cases(
    args: argparse.Namespace, graph_enabled: bool, test_prefill: bool,
) -> tuple[dict, dict, tuple[int, int]]:
    llm = LLM(
        args.model,
        enforce_eager=not graph_enabled,
        enable_prefill_cudagraph=graph_enabled and test_prefill,
        use_triton_attn=True,
        kv_quant=True,
        max_model_len=512,
        max_num_seqs=32,
        max_num_batched_tokens=512,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        if graph_enabled and not hasattr(llm.model_runner, "graphs"):
            raise AssertionError("Decode CUDA Graph was not captured")
        if graph_enabled and test_prefill and not hasattr(llm.model_runner, "prefill_graphs"):
            raise AssertionError("Prefill CUDA Graph was not captured")

        greedy_one = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        prefill_outputs = {}
        if test_prefill:
            for index, length in enumerate(PREFILL_BOUNDARIES):
                prefill_outputs[length] = llm.generate(
                    [prompt(length, 10 + index)], greedy_one, use_tqdm=False
                )[0]["token_ids"]

        greedy_decode = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
        decode_outputs = {}
        for offset, batch_size in enumerate(DECODE_BATCH_SIZES):
            prompts = [prompt(32, 100 + offset * 32 + index) for index in range(batch_size)]
            decode_outputs[batch_size] = [item["token_ids"] for item in llm.generate(
                prompts, greedy_decode, use_tqdm=False
            )]

        replays = (
            llm.model_runner.prefill_graph_replay_count,
            llm.model_runner.decode_graph_replay_count,
        )
        return prefill_outputs, decode_outputs, replays
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    test_prefill = not args.decode_only
    print("[1/2] eager reference ...")
    eager_prefill, eager_decode, eager_replays = run_cases(
        args, graph_enabled=False, test_prefill=test_prefill,
    )
    print("[2/2] CUDA Graph ...")
    graph_prefill, graph_decode, graph_replays = run_cases(
        args, graph_enabled=True, test_prefill=test_prefill,
    )

    if test_prefill and eager_prefill != graph_prefill:
        print("Dense Prefill mismatch (eager → graph):")
        for length in PREFILL_BOUNDARIES:
            print(f"  {length}: {eager_prefill[length]} → {graph_prefill[length]}")
        raise AssertionError("Dense Prefill greedy outputs differ: eager vs CUDA Graph")
    if eager_decode != graph_decode:
        print("Decode mismatch (eager → graph):")
        for batch_size in DECODE_BATCH_SIZES:
            if eager_decode[batch_size] != graph_decode[batch_size]:
                print(f"  batch={batch_size}: {eager_decode[batch_size]} → {graph_decode[batch_size]}")
        raise AssertionError("Decode greedy outputs differ: eager vs CUDA Graph")
    if test_prefill:
        assert graph_replays[0] >= len(PREFILL_BOUNDARIES), "Prefill CUDA Graph did not replay every boundary case"
    assert graph_replays[1] > 0, "Decode CUDA Graph did not replay"

    print("=" * 72)
    print("CUDA Graph correctness: PASS")
    if test_prefill:
        print(f"Prefill boundaries: {PREFILL_BOUNDARIES}  → outputs match; graph replays={graph_replays[0]}")
    print(f"Decode batch sizes: {DECODE_BATCH_SIZES}  → outputs match; graph replays={graph_replays[1]}")
    print(f"Eager replay counters: prefill={eager_replays[0]}, decode={eager_replays[1]}")


if __name__ == "__main__":
    main()
