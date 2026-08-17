"""Decode paged-attention kernel benchmark: triton_paged_attention vs flash_attn_with_kvcache.

Decode is memory-bandwidth-bound, so we sweep two orthogonal axes:
  * context_len  — how much KV history each query reads (bandwidth demand)
  * num_seqs     — batch size (parallelism / latency hiding)

ratio = triton_time / flash_attn_time. ratio > 1 means Triton is slower.
Run:
    python bench_triton_decode.py
"""
import statistics
import time

import torch
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.triton_attn import triton_paged_attention

DEVICE = "cuda"
BLOCK_SIZE = 256
WARMUP = 5
REPEATS = 30


def _make_inputs(ctx_len, num_seqs, num_heads, num_kv_heads, head_dim=128, dtype=torch.bfloat16):
    torch.manual_seed(0)
    blocks_per_seq = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * blocks_per_seq
    # Sequential physical blocks (shuffling doesn't change bandwidth; correctness
    # of arbitrary block_table is already covered by tests/test_triton_attn.py).
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(num_seqs, blocks_per_seq)
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    q = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=DEVICE)
    context_lens = torch.full((num_seqs,), ctx_len, dtype=torch.int32, device=DEVICE)
    scale = head_dim ** -0.5
    return q, k_cache, v_cache, block_tables, context_lens, scale


def _median_ms(fn):
    for _ in range(WARMUP):   # includes Triton's one-time JIT compile
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def bench(ctx_len, num_seqs, num_heads=24, num_kv_heads=8):
    q, k_cache, v_cache, block_tables, context_lens, scale = _make_inputs(
        ctx_len, num_seqs, num_heads, num_kv_heads)

    ref = lambda: flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context_lens, block_table=block_tables,
        softmax_scale=scale, causal=True,
    )
    tri = lambda: triton_paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale)

    t_ref = _median_ms(ref)
    t_tri = _median_ms(tri)
    ratio = t_tri / t_ref
    pct = 100.0 / ratio
    print(f"ctx={ctx_len:>5}  batch={num_seqs:>4}  "
          f"flash_attn={t_ref:7.3f}ms  triton={t_tri:7.3f}ms  "
          f"ratio={ratio:5.2f}x  (triton={pct:5.1f}% of flash_attn)")
    return ratio


def main():
    print(f"device: {torch.cuda.get_device_name(0)}  (Llama-3.2-3B: 24/8 heads)")
    print("--- sweep context_len (batch=32) ---")
    for ctx_len in [512, 1024, 2048, 4096, 8192]:
        bench(ctx_len, num_seqs=32)
    print("--- sweep batch (ctx_len=2048) ---")
    for num_seqs in [1, 8, 32, 64, 128]:
        bench(2048, num_seqs)


if __name__ == "__main__":
    main()
