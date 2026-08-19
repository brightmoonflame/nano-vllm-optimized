"""End-to-end BF16 vs online INT8 W8A16 benchmark.

Uses the Llama-3.2-3B path documented in README by default:

    python bench_w8a16.py --model /root/model/Llama-3.2-3B-Instruct

The two modes run in separate engine instances, so the BF16 baseline and the
quantized model never coexist on the GPU. KV quantization is intentionally off
to isolate Linear-weight memory and throughput.
"""
import argparse
import gc
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.layers.linear import LinearBase


def linear_weight_bytes(llm: LLM) -> int:
    total = 0
    for module in llm.model_runner.model.modules():
        if isinstance(module, LinearBase):
            total += module.weight.numel() * module.weight.element_size()
            if module.weight_quant is not None:
                total += module.weight_scale.numel() * module.weight_scale.element_size()
    return total


def run(model: str, weight_quant: str | None, prompts: list[list[int]], max_tokens: int) -> tuple[int, float]:
    llm = LLM(
        model,
        weight_quant=weight_quant,
        kv_quant=False,
        use_triton_attn=True,
        enforce_eager=True,
    )
    weight_bytes = linear_weight_bytes(llm)
    sp = SamplingParams(temperature=0, ignore_eos=True, max_tokens=max_tokens)
    # Compile Triton / warm caches outside the timed section.
    llm.generate([prompts[0]], sp, use_tqdm=False)
    torch.cuda.synchronize()
    start = perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return weight_bytes, seconds


def main():
    parser = argparse.ArgumentParser(description="BF16 vs online INT8 W8A16 benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    model = str(Path(args.model).expanduser().resolve())

    rng = Random(0)
    prompts = [[rng.randrange(10_000) for _ in range(args.input_len)] for _ in range(args.num_prompts)]
    bf16_bytes, bf16_seconds = run(model, None, prompts, args.max_tokens)
    int8_bytes, int8_seconds = run(model, "int8_w8a16", prompts, args.max_tokens)
    total_tokens = args.num_prompts * args.max_tokens

    print("=" * 68)
    print(f"device: {torch.cuda.get_device_name(0)}   model: {model}")
    print("=" * 68)
    print(f"{'mode':<12} {'linear weights':>16} {'time':>10} {'throughput':>16}")
    print(f"{'BF16':<12} {bf16_bytes / 2**20:>13.1f} MiB {bf16_seconds:>8.2f}s {total_tokens / bf16_seconds:>12.1f} tok/s")
    print(f"{'INT8 W8A16':<12} {int8_bytes / 2**20:>13.1f} MiB {int8_seconds:>8.2f}s {total_tokens / int8_seconds:>12.1f} tok/s")
    print("-" * 68)
    print(f"Linear-weight memory: {int8_bytes / bf16_bytes * 100:.1f}% of BF16 "
          f"({(1 - int8_bytes / bf16_bytes) * 100:.1f}% saved)")
    print(f"End-to-end throughput: {(bf16_seconds / int8_seconds):.2f}x of BF16")


if __name__ == "__main__":
    main()
