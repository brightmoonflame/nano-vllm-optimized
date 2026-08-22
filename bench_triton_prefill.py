"""Prefill attention kernel benchmark.

Measures raw kernel latency in isolation (no scheduler / KV-cache / sampling
overhead) so the number reflects only the attention implementation. Covers
dense BF16 Prefill plus paged BF16 / INT8 Prefix-or-Chunked Prefill. Run:
    python bench_triton_prefill.py

For BF16, ratio = triton_time / flash_attn_time; a ratio > 1 means Triton is
slower. INT8 additionally compares fused Triton against the compatible
whole-cache dequant + flash-attn fallback.
"""
import statistics
import time

import torch
from flash_attn import flash_attn_varlen_func

from nanovllm.layers.triton_attn import triton_flash_attn_varlen
from nanovllm.layers.kv_quant import NUM_GROUPS, dequant_kvcache

DEVICE = "cuda"
BLOCK_SIZE = 256
WARMUP = 5
REPEATS = 30


def _make_inputs(seqlen, num_heads, num_kv_heads, head_dim=128, num_seqs=1, dtype=torch.bfloat16):
    torch.manual_seed(0)
    lens = torch.full((num_seqs,), seqlen, dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0).to(torch.int32)]).to(DEVICE)
    n = num_seqs * seqlen
    scale = head_dim ** -0.5
    q = torch.randn(n, num_heads, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(n, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(n, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    return q, k, v, cu, seqlen, scale


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


def _quantize_groupwise(bf16_cache: torch.Tensor):
    """Synthetic INT8 cache with the production per-group scale layout."""
    num_blocks, block_size, num_kv_heads, head_dim = bf16_cache.shape
    group_size = head_dim // NUM_GROUPS
    x = bf16_cache.float().reshape(num_blocks, block_size, num_kv_heads, NUM_GROUPS, group_size)
    scales = x.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    expanded_scales = scales[..., None].expand(
        num_blocks, block_size, num_kv_heads, NUM_GROUPS, group_size,
    ).reshape_as(bf16_cache)
    quantized = torch.round(bf16_cache.float() / expanded_scales).clamp(-127, 127).to(torch.int8)
    return quantized, scales


def bench(seqlen, num_heads, num_kv_heads, num_seqs=1):
    q, k, v, cu, max_seqlen, scale = _make_inputs(seqlen, num_heads, num_kv_heads, num_seqs=num_seqs)

    ref = lambda: flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=scale, causal=True,
    )
    tri = lambda: triton_flash_attn_varlen(q, k, v, cu, max_seqlen=max_seqlen, scale=scale)

    t_ref = _median_ms(ref)
    t_tri = _median_ms(tri)
    ratio = t_tri / t_ref
    pct = 100.0 / ratio
    print(f"seqlen={seqlen:>5}  heads={num_heads:>2}/{num_kv_heads:<2}  "
          f"flash_attn={t_ref:7.3f}ms  triton={t_tri:7.3f}ms  "
          f"ratio={ratio:5.2f}x  (triton={pct:5.1f}% of flash_attn)")
    return ratio


def _make_paged_inputs(prefix_len, new_len, num_heads, num_kv_heads, num_seqs=2,
                       head_dim=128, dtype=torch.bfloat16):
    """Build a varlen batch whose queries are new tokens and K/V are paged."""
    torch.manual_seed(0)
    context_len = prefix_len + new_len
    blocks_per_seq = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * blocks_per_seq
    # Shuffled ids ensure the measured path performs block-table indirection.
    block_tables = torch.randperm(num_blocks, dtype=torch.int32, device=DEVICE).view(num_seqs, blocks_per_seq)
    q_lens = torch.full((num_seqs,), new_len, dtype=torch.int32)
    k_lens = torch.full((num_seqs,), context_len, dtype=torch.int32)
    cu_q = torch.cat([torch.zeros(1, dtype=torch.int32), q_lens.cumsum(0)]).to(DEVICE)
    cu_k = torch.cat([torch.zeros(1, dtype=torch.int32), k_lens.cumsum(0)]).to(DEVICE)
    q = torch.randn(num_seqs * new_len, num_heads, head_dim, dtype=dtype, device=DEVICE)
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    return q, k_cache, v_cache, cu_q, cu_k, block_tables, head_dim ** -0.5


def bench_paged(prefix_len, new_len, num_heads, num_kv_heads, num_seqs=2):
    """Compare paged Prefill BF16 and INT8 fused paths against flash-attn."""
    q, k_cache, v_cache, cu_q, cu_k, block_tables, scale = _make_paged_inputs(
        prefix_len, new_len, num_heads, num_kv_heads, num_seqs=num_seqs,
    )
    max_q, max_k = new_len, prefix_len + new_len

    flash_bf16 = lambda: flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max_q, max_seqlen_k=max_k,
        softmax_scale=scale, causal=True, block_table=block_tables,
    )
    triton_bf16 = lambda: triton_flash_attn_varlen(
        q, k_cache, v_cache, cu_q,
        max_seqlen_q=max_q, scale=scale,
        cu_seqlens_k=cu_k, block_tables=block_tables,
    )

    k_i8, k_scales = _quantize_groupwise(k_cache)
    v_i8, v_scales = _quantize_groupwise(v_cache)

    def dequant_flash():
        k_bf16 = dequant_kvcache(k_i8, k_scales)
        v_bf16 = dequant_kvcache(v_i8, v_scales)
        return flash_attn_varlen_func(
            q, k_bf16, v_bf16,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            softmax_scale=scale, causal=True, block_table=block_tables,
        )

    triton_int8 = lambda: triton_flash_attn_varlen(
        q, k_i8, v_i8, cu_q,
        max_seqlen_q=max_q, scale=scale,
        cu_seqlens_k=cu_k, block_tables=block_tables,
        k_scale=k_scales, v_scale=v_scales,
    )

    t_flash = _median_ms(flash_bf16)
    t_triton = _median_ms(triton_bf16)
    t_dequant = _median_ms(dequant_flash)
    t_int8 = _median_ms(triton_int8)
    print(f"prefix={prefix_len:>5}  new={new_len:>5}  batch={num_seqs:>2}  "
          f"flash_bf16={t_flash:7.3f}ms  triton_bf16={t_triton:7.3f}ms  "
          f"| bf16 ratio={t_triton / t_flash:5.2f}x")
    print(f"{'':>31}dequant+flash={t_dequant:7.3f}ms  triton_int8={t_int8:7.3f}ms  "
          f"| int8 vs dequant={t_dequant / t_int8:5.2f}x  int8 vs flash_bf16={t_flash / t_int8:5.2f}x")


def main():
    print(f"device: {torch.cuda.get_device_name(0)}")
    # Llama-3.2-3B: 24 attention heads / 8 KV heads (GQA 3:1), head_dim 128.
    for seqlen in [512, 1024, 2048, 4096, 8192]:
        bench(seqlen, num_heads=24, num_kv_heads=8)
    print("--- paged prefill: prefix cache / chunked continuation (batch=2) ---")
    for total_len in [1024, 2048, 4096]:
        bench_paged(total_len // 2, total_len // 2, num_heads=24, num_kv_heads=8)


if __name__ == "__main__":
    main()
