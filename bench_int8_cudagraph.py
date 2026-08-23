"""End-to-end Decode CUDA Graph A/B benchmark for fused INT8 KV attention.

Runs the same workload in two isolated processes so CUDA Graph capture state
and GPU allocations cannot leak between modes:
  1. INT8 KV Cache + Triton attention, eager Decode
  2. INT8 KV Cache + Triton attention, Decode CUDA Graph enabled

The graph mode is valid only for models whose target attention layers all use
the fused Triton path (for example Llama/Qwen global attention). The child
benchmark records this capability in its JSON metadata.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SERVING_BENCH = ROOT / "serving_bench.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare eager and CUDA-Graph Decode for fused INT8 KV Triton attention."
    )
    parser.add_argument("--model", required=True, help="Path to a local Hugging Face model.")
    parser.add_argument("--mode", choices=["offline", "serving"], default="offline")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--request-rate", type=float, default=32.0)
    parser.add_argument("--length-distribution", choices=["fixed", "uniform"], default="fixed")
    parser.add_argument("--min-input-len", type=int, default=1024)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=128)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, help="Optional A/B summary JSON path.")
    args = parser.parse_args()
    if not Path(args.model).expanduser().is_dir():
        parser.error(f"model directory does not exist: {Path(args.model).expanduser()}")
    if args.min_input_len <= 0 or args.min_input_len > args.max_input_len:
        parser.error("input lengths must satisfy 0 < min <= max")
    if args.min_output_len <= 0 or args.min_output_len > args.max_output_len:
        parser.error("output lengths must satisfy 0 < min <= max")
    if args.min_input_len + args.min_output_len > args.max_model_len:
        parser.error("minimum input and output lengths exceed --max-model-len")
    return args


def child_command(args: argparse.Namespace, enforce_eager: bool, output_json: Path) -> list[str]:
    command = [
        sys.executable, str(SERVING_BENCH),
        "--model", args.model,
        "--mode", args.mode,
        "--num-requests", str(args.num_requests),
        "--request-rate", str(args.request_rate),
        "--length-distribution", args.length_distribution,
        "--min-input-len", str(args.min_input_len),
        "--max-input-len", str(args.max_input_len),
        "--min-output-len", str(args.min_output_len),
        "--max-output-len", str(args.max_output_len),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--warmup-requests", str(args.warmup_requests),
        "--seed", str(args.seed),
        "--kv-quant", "--use-triton-attn",
        "--output-json", str(output_json),
    ]
    if enforce_eager:
        command.append("--enforce-eager")
    return command


def ms(summary: dict | None) -> float | None:
    return None if summary is None else summary["p50"] * 1e3


def ratio(faster: float | None, slower: float | None) -> str:
    if faster is None or slower is None or faster == 0:
        return "n/a"
    return f"{slower / faster:.2f}x"


def main() -> None:
    args = parse_args()
    cases = (("INT8 Triton eager", True), ("INT8 Triton + Decode CUDA Graph", False))
    results = []
    with tempfile.TemporaryDirectory(prefix="nano-vllm-int8-graph-") as directory:
        directory_path = Path(directory)
        for index, (label, enforce_eager) in enumerate(cases):
            path = directory_path / f"{index}.json"
            print(f"\n{'#' * 80}\n[{index + 1}/2] {label}\n{'#' * 80}")
            subprocess.run(child_command(args, enforce_eager, path), cwd=ROOT, check=True)
            results.append({"label": label, **json.loads(path.read_text(encoding="utf-8"))})

    eager, graph = results
    eager_agg, graph_agg = eager["aggregate"], graph["aggregate"]
    eager_tpot, graph_tpot = ms(eager_agg["tpot_s"]), ms(graph_agg["tpot_s"])
    eager_tps, graph_tps = eager_agg["output_throughput_tps"], graph_agg["output_throughput_tps"]
    print("\n" + "=" * 96)
    print("Fused INT8 KV Triton Decode: eager vs CUDA Graph")
    print("=" * 96)
    print(f"{'mode':<36} {'output tok/s':>14} {'TPOT p50':>14} {'graph captured':>16}")
    print("-" * 96)
    for result in results:
        aggregate, metadata = result["aggregate"], result["metadata"]
        print(
            f"{result['label']:<36} {aggregate['output_throughput_tps']:>14.1f} "
            f"{ms(aggregate['tpot_s']) if aggregate['tpot_s'] else float('nan'):>13.2f}ms "
            f"{str(metadata['decode_cudagraph_captured']):>16}"
        )
    print("-" * 96)
    print(
        "CUDA Graph vs eager: "
        f"throughput {graph_tps / eager_tps:.2f}x, TPOT {ratio(graph_tpot, eager_tpot)}."
    )
    if not graph["metadata"]["int8_decode_cudagraph_safe"]:
        print("WARNING: this model has a non-fused Attention fallback; graph results are not valid for INT8 Decode.")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"arguments": vars(args), "results": results}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"summary JSON: {args.output_json}")


if __name__ == "__main__":
    main()
