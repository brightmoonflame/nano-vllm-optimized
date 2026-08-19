"""Decode paged-attention kernel benchmark.

BF16 part: triton_paged_attention vs flash_attn_with_kvcache.
INT8 part: fused triton_paged_attention_int8 vs the default kv_quant=True path
           (whole-cache dequant + flash_attn) vs the BF16 ceiling.

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

from nanovllm.layers.triton_attn import (
    triton_paged_attention, _num_splits_for,
)
from nanovllm.layers.kv_quant import dequant_kvcache

DEVICE = "cuda"
BLOCK_SIZE = 256
WARMUP = 5
REPEATS = 30


NUM_GROUPS = 8


def _quantize_groupwise(bf16_cache: torch.Tensor):
    """Per-(token, head, group) symmetric quantization (8 groups of 16 dims)."""
    n, t, h, d = bf16_cache.shape
    g = d // NUM_GROUPS
    x = bf16_cache.float().reshape(n, t, h, NUM_GROUPS, g)
    sc = x.abs().amax(dim=-1) / 127.0                                  # (n, t, h, NUM_GROUPS)
    sc = sc.clamp(min=1e-6)
    sc_full = sc[..., None].expand(n, t, h, NUM_GROUPS, g).reshape(n, t, h, d)
    i8 = torch.round(bf16_cache.float() / sc_full).clamp(-127, 127).to(torch.int8)
    return i8, sc


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
    splits = _num_splits_for(num_seqs, num_heads)
    print(f"ctx={ctx_len:>5}  batch={num_seqs:>4}  "
          f"flash_attn={t_ref:7.3f}ms  triton={t_tri:7.3f}ms  "
          f"ratio={ratio:5.2f}x  (triton={pct:5.1f}% of flash_attn)  [splits={splits}]")
    return ratio


def bench_int8(ctx_len, num_seqs, num_heads=24, num_kv_heads=8):
    """Four-way comparison at fixed (ctx_len, batch):
      1. dequant(whole cache) + flash_attn — the default kv_quant=True path
         (dequantizes EVERY block every layer every step; reads ctx-independent)
      2. flash_attn BF16                   — the kv_quant=False ceiling
      3. triton BF16                       — stage-2 kernel
      4. triton INT8 fused                 — stage-3 kernel (this work)
    """
    q, k_cache, v_cache, block_tables, context_lens, scale = _make_inputs(
        ctx_len, num_seqs, num_heads, num_kv_heads)
    k_i8, k_sc = _quantize_groupwise(k_cache)
    v_i8, v_sc = _quantize_groupwise(v_cache)

    def dequant_flash():
        # Mirrors attention.py's default kv_quant branch: full-cache dequant
        # (all blocks, not just the ctx window) then flash_attn.
        kb = dequant_kvcache(k_i8, k_sc)
        vb = dequant_kvcache(v_i8, v_sc)
        return flash_attn_with_kvcache(
            q.unsqueeze(1), kb, vb,
            cache_seqlens=context_lens, block_table=block_tables,
            softmax_scale=scale, causal=True,
        )

    def flash_bf16():
        return flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context_lens, block_table=block_tables,
            softmax_scale=scale, causal=True,
        )

    def tri_bf16():
        return triton_paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale)

    def tri_int8():
        return triton_paged_attention(q, k_i8, v_i8, block_tables, context_lens, scale,
                                      k_scale=k_sc, v_scale=v_sc)

    t_dequant = _median_ms(dequant_flash)
    t_flash = _median_ms(flash_bf16)
    t_tri = _median_ms(tri_bf16)
    t_int8 = _median_ms(tri_int8)
    print(f"ctx={ctx_len:>5}  batch={num_seqs:>4}  "
          f"dequant+flash={t_dequant:7.3f}ms  flash_bf16={t_flash:7.3f}ms  "
          f"triton_bf16={t_tri:7.3f}ms  triton_int8={t_int8:7.3f}ms  "
          f"| int8 vs dequant: {t_dequant / t_int8:5.2f}x  int8 vs flash_bf16: {t_flash / t_int8:5.2f}x")


def main():
    print(f"device: {torch.cuda.get_device_name(0)}  (Llama-3.2-3B: 24/8 heads)")
    print("--- sweep context_len (batch=32) ---")
    for ctx_len in [512, 1024, 2048, 4096, 8192]:
        bench(ctx_len, num_seqs=32)
    print("--- sweep batch (ctx_len=2048) ---")
    for num_seqs in [1, 8, 32, 64, 128]:
        bench(2048, num_seqs)
    print("--- INT8 fused (batch=32) ---")
    for ctx_len in [1024, 2048, 4096, 8192]:
        bench_int8(ctx_len, num_seqs=32)


if __name__ == "__main__":
    main()
