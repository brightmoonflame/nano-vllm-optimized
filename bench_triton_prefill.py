"""Prefill attention kernel benchmark: triton_flash_attn_varlen vs flash_attn_varlen_func.

Measures raw kernel latency in isolation (no scheduler / KV-cache / sampling
overhead) so the number reflects only the attention implementation. Run:
    python bench_triton_prefill.py

ratio = triton_time / flash_attn_time. A ratio > 1 means Triton is slower;
the stage-1 expectation is 1.05 ~ 1.25 (i.e. Triton at 80% ~ 95% of flash_attn).
"""
import statistics
import time

import torch
from flash_attn import flash_attn_varlen_func

from nanovllm.layers.triton_attn import triton_flash_attn_varlen

DEVICE = "cuda"
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


def main():
    print(f"device: {torch.cuda.get_device_name(0)}")
    # Llama-3.2-3B: 24 attention heads / 8 KV heads (GQA 3:1), head_dim 128.
    for seqlen in [512, 1024, 2048, 4096, 8192]:
        bench(seqlen, num_heads=24, num_kv_heads=8)


if __name__ == "__main__":
    main()
