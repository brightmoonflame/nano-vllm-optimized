"""Integration correctness checks for fused Triton INT8 Decode CUDA Graph."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


DECODE_BATCH_SIZES = (3, 5, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check eager/Graph correctness across Decode CUDA-Graph batch buckets.")
    parser.add_argument("--model", required=True, help="Path to a local Hugging Face model.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if not Path(args.model).expanduser().is_dir():
        parser.error(f"model directory does not exist: {Path(args.model).expanduser()}")
    return args


def prompt(length: int, tag: int) -> list[int]:
    """Same shape, distinct prefix: prevents Prefix Cache from changing the case."""
    return [tag] + [0] * (length - 1)


def run_cases(args: argparse.Namespace, graph_enabled: bool) -> tuple[dict, int]:
    llm = LLM(
        args.model,
        enforce_eager=not graph_enabled,
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
        greedy_decode = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
        decode_outputs = {}
        for offset, batch_size in enumerate(DECODE_BATCH_SIZES):
            prompts = [prompt(32, 100 + offset * 32 + index) for index in range(batch_size)]
            decode_outputs[batch_size] = [item["token_ids"] for item in llm.generate(
                prompts, greedy_decode, use_tqdm=False
            )]

        return decode_outputs, llm.model_runner.decode_graph_replay_count
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    print("[1/2] eager reference ...")
    eager_decode, eager_replays = run_cases(args, graph_enabled=False)
    print("[2/2] CUDA Graph ...")
    graph_decode, graph_replays = run_cases(args, graph_enabled=True)
    if eager_decode != graph_decode:
        print("Decode mismatch (eager → graph):")
        for batch_size in DECODE_BATCH_SIZES:
            if eager_decode[batch_size] != graph_decode[batch_size]:
                print(f"  batch={batch_size}: {eager_decode[batch_size]} → {graph_decode[batch_size]}")
        raise AssertionError("Decode greedy outputs differ: eager vs CUDA Graph")
    assert graph_replays > 0, "Decode CUDA Graph did not replay"

    print("=" * 72)
    print("CUDA Graph correctness: PASS")
    print(f"Decode batch sizes: {DECODE_BATCH_SIZES}  → outputs match; graph replays={graph_replays}")
    print(f"Eager Decode replay counter: {eager_replays}")


if __name__ == "__main__":
    main()
