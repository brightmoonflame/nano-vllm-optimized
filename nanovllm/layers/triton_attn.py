"""Self-researched Triton attention kernels (forward-only).

Stage 1 — prefill: a dense (non-paged) varlen FlashAttention-2 kernel, a
drop-in alternative to `flash_attn_varlen_func` for the common case
(`context.block_tables is None`, i.e. no prefix-cache hit).

Stage 2 — decode: a single-query paged attention kernel, a drop-in
alternative to `flash_attn_with_kvcache` (BF16 path), reading K/V directly
from the paged KV cache via `block_tables`.

Both are gated behind `Attention.use_triton_attn` in `attention.py` —
default behavior (flash_attn package) is untouched.

Algorithm (standard FlashAttention-2 forward, see Dao et al. 2023):
for each query block, iterate over key/value blocks up to the causal
boundary while keeping a running `(row_max, row_sumexp, weighted_acc)`
triple and rescaling it as each new block arrives — the full (seqlen x
seqlen) attention matrix is never materialized, and softmax normalization
only happens once at the very end ("online softmax").
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    cu_seqlens_ptr,
    stride_qn, stride_qh,
    stride_kn, stride_kh,
    stride_vn, stride_vh,
    stride_on, stride_oh,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m_block = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seqlen = seq_end - seq_start

    m_start = m_block * BLOCK_M
    if m_start >= seqlen:
        return    # this query block lies entirely past this sequence's end

    # GQA: each group of (num_heads // num_kv_heads) query heads shares one KV head.
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen

    q_ptrs = q_ptr + (seq_start + offs_m)[:, None] * stride_qn + head_idx * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Running online-softmax state: row-wise max, row-wise sumexp, weighted accumulator.
    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Causal: query block [m_start, m_start+BLOCK_M) only needs keys up to its own last row.
    n_end = m_start + BLOCK_M
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen

        k_ptrs = k_ptr + (seq_start + offs_n)[:, None] * stride_kn + kv_head_idx * stride_kh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask & mask_n[None, :], qk, float("-inf"))

        # Rescale the running state to the new (larger) row max, then fold in this block.
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_ptr + (seq_start + offs_n)[:, None] * stride_vn + kv_head_idx * stride_vh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + (seq_start + offs_m)[:, None] * stride_on + head_idx * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None])


def triton_flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
) -> torch.Tensor:
    """FlashAttention-2 forward, causal, GQA, dense varlen-packed layout.

    q: (total_tokens, num_heads, head_dim)
    k, v: (total_tokens, num_kv_heads, head_dim)
    cu_seqlens: (num_seqs + 1,) int32 — shared boundary for q and k/v (this
        kernel only covers the no-prefix-cache case, where cu_seqlens_q ==
        cu_seqlens_k; prefix-cache/paged prefill still falls back to
        `flash_attn_varlen_func`, see attention.py).

    Returns a tensor shaped like q.
    """
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads (GQA)"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "head_dim must be contiguous"
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of 2 (tl.arange requirement)"

    o = torch.empty_like(q)
    num_seqs = cu_seqlens.numel() - 1
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(max_seqlen, BLOCK_M), num_heads, num_seqs)

    _fwd_kernel[grid](
        q, k, v, o,
        cu_seqlens,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return o


@triton.jit
def _paged_attn_decode_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
    block_tables_ptr, context_lens_ptr,
    scale,
    stride_qn, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_on, stride_oh,
    stride_bt,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # tokens per cache block (kvcache_block_size)
    BLOCK_N: tl.constexpr,      # kv tokens per iteration; must divide BLOCK_SIZE
):
    # One program per (sequence, query head): decode has a single query token,
    # so softmax state is scalar and the accumulator is a HEAD_DIM vector.
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    context_len = tl.load(context_lens_ptr + seq_idx)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + seq_idx * stride_qn + head_idx * stride_qh + offs_d)

    # Running online-softmax state: scalar max / sumexp + vector accumulator.
    m_i = tl.full([], float("-inf"), dtype=tl.float32)
    l_i = tl.full([], 0.0, dtype=tl.float32)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # The query is the last token, so every key position [0, context_len) is
    # visible — no causal mask needed, only the context_len boundary.
    # BLOCK_N divides BLOCK_SIZE, so each tile lies inside exactly one cache
    # block and a single block_table lookup per tile suffices.
    for n_start in range(0, context_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < context_len

        physical_block = tl.load(block_tables_ptr + seq_idx * stride_bt + n_start // BLOCK_SIZE)

        k_off = (physical_block.to(tl.int64) * stride_kb
                 + offs_n[:, None] * stride_kt + kv_head_idx * stride_kh + offs_d[None, :])
        k = tl.load(k_cache_ptr + k_off, mask=mask_n[:, None], other=0.0)

        qk = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1) * scale
        qk = tl.where(mask_n, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)

        v_off = (physical_block.to(tl.int64) * stride_vb
                 + offs_n[:, None] * stride_vt + kv_head_idx * stride_vh + offs_d[None, :])
        v = tl.load(v_cache_ptr + v_off, mask=mask_n[:, None], other=0.0)

        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(o_ptr + seq_idx * stride_on + head_idx * stride_oh + offs_d,
             acc.to(o_ptr.dtype.element_ty))


def triton_paged_attention(
    q: torch.Tensor,             # (num_seqs, num_heads, head_dim)
    k_cache: torch.Tensor,       # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,       # same layout as k_cache
    block_tables: torch.Tensor,  # (num_seqs, max_num_blocks) int32, logical→physical block map
    context_lens: torch.Tensor,  # (num_seqs,) int32, total tokens cached per seq
    scale: float,
) -> torch.Tensor:
    """Single-query paged attention over the paged KV cache (decode path).

    A drop-in alternative to `flash_attn_with_kvcache` for the BF16 decode
    case. Returns a tensor shaped like q.
    """
    num_seqs, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads (GQA)"
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of 2 (tl.arange requirement)"
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1

    BLOCK_N = 64    # divides block_size (256), so one tile never spans two cache blocks
    assert block_size % BLOCK_N == 0

    o = torch.empty_like(q)
    grid = (num_seqs, num_heads)
    _paged_attn_decode_kernel[grid](
        q, k_cache, v_cache, o,
        block_tables, context_lens,
        scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        o.stride(0), o.stride(1),
        block_tables.stride(0),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return o
