"""Self-researched Triton implementation of FlashAttention-2 (forward-only).

Stage 1 of TRITON_ATTN_PLAN.md: a dense (non-paged) varlen prefill kernel that
is a drop-in alternative to `flash_attn_varlen_func` for the common case
(`context.block_tables is None`, i.e. no prefix-cache hit). Gated behind
`Attention.use_triton_attn` in `attention.py` — default behavior is untouched.

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
    BLOCK_M = BLOCK_N = 64
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
    )
    return o
