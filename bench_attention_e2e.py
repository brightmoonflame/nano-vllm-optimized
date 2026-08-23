"""End-to-end attention backend comparison.

Runs the same serving_bench workload in four isolated processes:
  1. BF16 + flash-attn
  2. BF16 + self-researched Triton attention
  3. INT8 KV cache + flash-attn compatibility fallback
  4. INT8 KV cache + fused Triton attention

Each child writes serving_bench's machine-readable JSON to a temporary file.
This driver reads those files and prints a directly comparable throughput / TTFT
/ TPOT table. Separate processes ensure a mode's model, KV cache, and CUDA
Graph state are destroyed before the next mode begins.

Run (eager is the fair raw-backend comparison):
  python bench_attention_e2e.py --model /root/model/Llama-3.2-3B-Instruct

Use --no-enforce-eager to measure the production configuration, including
CUDA Graph where a mode supports it.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
SERVING_BENCH = ROOT / "serving_bench.py"


CASES = (
    ("BF16 flash-attn", False, False),
    ("BF16 Triton", False, True),
    ("INT8 flash fallback", True, False),
    ("INT8 Triton fused", True, True),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare flash-attn and Triton end-to-end attention paths.")
    p.add_argument("--model", required=True, help="Path to the local Hugging Face model directory.")
    p.add_argument("--mode", choices=["offline", "serving"], default="offline")
    p.add_argument("--num-requests", type=int, default=128)
    p.add_argument("--request-rate", type=float, default=32.0,
                   help="Poisson arrival rate for --mode serving.")
    p.add_argument("--length-distribution", choices=["fixed", "uniform"], default="fixed")
    p.add_argument("--min-input-len", type=int, default=1024)
    p.add_argument("--max-input-len", type=int, default=1024)
    p.add_argument("--min-output-len", type=int, default=128)
    p.add_argument("--max-output-len", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True,
                   help="default: true for a fair raw-backend comparison; use --no-enforce-eager for CUDA Graph")
    p.add_argument("--enable-chunked-prefill", action="store_true")
    p.add_argument("--prefill-chunk-size", type=int, default=1024)
    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path,
                   help="Optional path for this driver's four-case summary JSON.")
    args = p.parse_args()
    if args.num_requests <= 0:
        p.error("--num-requests must be positive")
    if args.min_input_len <= 0 or args.min_input_len > args.max_input_len:
        p.error("input lengths must satisfy 0 < --min-input-len <= --max-input-len")
    if args.min_output_len <= 0 or args.min_output_len > args.max_output_len:
        p.error("output lengths must satisfy 0 < --min-output-len <= --max-output-len")
    if args.min_input_len + args.min_output_len > args.max_model_len:
        p.error("minimum input and output lengths exceed --max-model-len")
    if args.mode == "serving" and args.request_rate <= 0:
        p.error("--request-rate must be positive in serving mode")
    return args


def build_command(args: argparse.Namespace, kv_quant: bool, use_triton: bool, result_path: Path) -> list[str]:
    cmd = [
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
        "--output-json", str(result_path),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.enable_chunked_prefill:
        cmd.extend(["--enable-chunked-prefill", "--prefill-chunk-size", str(args.prefill_chunk_size)])
    if kv_quant:
        cmd.append("--kv-quant")
    if use_triton:
        cmd.append("--use-triton-attn")
    return cmd


def milliseconds(summary: Optional[dict]) -> Optional[float]:
    return None if summary is None else summary["p50"] * 1e3


def speedup(value: Optional[float], baseline: Optional[float], higher_is_better: bool) -> str:
    if value is None or baseline is None or value == 0 or baseline == 0:
        return "n/a"
    ratio = value / baseline if higher_is_better else baseline / value
    return f"{ratio:.2f}x"


def print_summary(results: list[dict]) -> None:
    baseline = results[0]["aggregate"]
    base_tps = baseline["output_throughput_tps"]
    base_ttft = milliseconds(baseline["ttft_s"])
    base_tpot = milliseconds(baseline["tpot_s"])

    print("\n" + "=" * 116)
    print("End-to-end attention backend comparison  (relative to BF16 flash-attn)")
    print("=" * 116)
    print(f"{'mode':<24} {'output tok/s':>13} {'throughput':>12} {'TTFT p50':>12} {'TPOT p50':>12} {'TPOT gain':>12}")
    print("-" * 116)
    for result in results:
        aggregate = result["aggregate"]
        tps = aggregate["output_throughput_tps"]
        ttft = milliseconds(aggregate["ttft_s"])
        tpot = milliseconds(aggregate["tpot_s"])
        print(f"{result['label']:<24} {tps:>13.1f} {speedup(tps, base_tps, True):>12} "
              f"{ttft if ttft is not None else float('nan'):>11.2f}ms "
              f"{tpot if tpot is not None else float('nan'):>11.2f}ms "
              f"{speedup(tpot, base_tpot, False):>12}")
    print("=" * 116)
    int8_fallback, int8_fused = results[2]["aggregate"], results[3]["aggregate"]
    print("INT8 fused vs INT8 fallback: "
          f"output throughput {speedup(int8_fused['output_throughput_tps'], int8_fallback['output_throughput_tps'], True)}, "
          f"TPOT {speedup(milliseconds(int8_fused['tpot_s']), milliseconds(int8_fallback['tpot_s']), False)}.")


def main() -> None:
    args = parse_args()
    model = Path(args.model).expanduser()
    if not model.is_dir():
        raise SystemExit(f"model directory does not exist: {model}")

    results = []
    with tempfile.TemporaryDirectory(prefix="nano-vllm-attn-e2e-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for index, (label, kv_quant, use_triton) in enumerate(CASES):
            result_path = tmpdir_path / f"{index}.json"
            print("\n" + "#" * 80)
            print(f"[{index + 1}/{len(CASES)}] {label}")
            print("#" * 80)
            subprocess.run(build_command(args, kv_quant, use_triton, result_path), cwd=ROOT, check=True)
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            results.append({
                "label": label,
                "kv_quant": kv_quant,
                "use_triton_attn": use_triton,
                "aggregate": raw["aggregate"],
                "metadata": raw["metadata"],
            })

    print_summary(results)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({
            "arguments": vars(args),
            "results": results,
        }, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"summary JSON: {args.output_json}")


if __name__ == "__main__":
    main()
