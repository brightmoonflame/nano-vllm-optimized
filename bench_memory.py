"""INT8 KV-cache memory/capacity benchmark: kv_quant=True vs False.

Measures, under the same GPU and gpu_memory_utilization, how many KV-cache
blocks each mode can allocate and the resulting total token capacity (a proxy
for max concurrent requests / max context length). This is the "memory half"
of the INT8 value proposition.

Run:
    python bench_memory.py --model /root/model/Llama-3.2-3B-Instruct
"""
import argparse
import gc
from pathlib import Path

import torch

from nanovllm import LLM


def measure(model: str, kv_quant: bool, gpu_memory_utilization: float) -> dict:
    llm = LLM(model, kv_quant=kv_quant, use_triton_attn=True, enforce_eager=True,
              gpu_memory_utilization=gpu_memory_utilization)
    cfg = llm.config
    hf = cfg.hf_config
    block_size = cfg.kvcache_block_size
    num_blocks = cfg.num_kvcache_blocks
    num_layers = hf.num_hidden_layers
    num_kv_heads = hf.num_key_value_heads
    head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    elem_size = 1 if kv_quant else hf.dtype.itemsize

    block_mb = 2 * num_layers * block_size * num_kv_heads * head_dim * elem_size / 1e6
    if kv_quant:
        # Group-wise FP32 scales (both K and V): NUM_GROUPS per token per head.
        block_mb += 2 * num_layers * block_size * num_kv_heads * 8 * 4 / 1e6
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "blocks": num_blocks,
        "block_mb": block_mb,
        "total_mb": block_mb * num_blocks,
        "total_tokens": num_blocks * block_size,
    }


def main():
    p = argparse.ArgumentParser(description="INT8 KV-cache memory/capacity benchmark.")
    p.add_argument("--model", required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = p.parse_args()
    model = str(Path(args.model).expanduser().resolve())

    bf16 = measure(model, kv_quant=False, gpu_memory_utilization=args.gpu_memory_utilization)
    int8 = measure(model, kv_quant=True, gpu_memory_utilization=args.gpu_memory_utilization)

    capacity_ratio = int8["blocks"] / bf16["blocks"]
    print("=" * 56)
    print(f"device: {torch.cuda.get_device_name(0)}   gpu_mem_util={args.gpu_memory_utilization}")
    print("=" * 56)
    print(f"{'mode':<6} {'block_mb':>9} {'blocks':>7} {'total_mb':>9} {'tokens':>10}")
    print(f"{'BF16':<6} {bf16['block_mb']:>9.1f} {bf16['blocks']:>7} {bf16['total_mb']:>9.0f} {bf16['total_tokens']:>10}")
    print(f"{'INT8':<6} {int8['block_mb']:>9.1f} {int8['blocks']:>7} {int8['total_mb']:>9.0f} {int8['total_tokens']:>10}")
    print("-" * 56)
    print(f"per-block memory: {int8['block_mb'] / bf16['block_mb'] * 100:.1f}% of BF16 "
          f"({(1 - int8['block_mb'] / bf16['block_mb']) * 100:.0f}% saved)")
    print(f"KV-cache capacity: {capacity_ratio:.2f}x more blocks/tokens (concurrency/context doubles)")


if __name__ == "__main__":
    main()
