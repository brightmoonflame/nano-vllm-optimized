"""Self-researched Triton attention kernels (forward-only).

Stage 1 — prefill: a dense (non-paged) varlen FlashAttention-2 kernel, a
drop-in alternative to `flash_attn_varlen_func` for the common case
(`context.block_tables is None`, i.e. no prefix-cache hit).

Stage 2 — decode: a single-query paged attention kernel, a drop-in
alternative to `flash_attn_with_kvcache` (BF16 path), reading K/V directly
from the paged KV cache via `block_tables`.

Stage 3 — INT8 fused decode: reads the INT8 paged cache directly and
dequantizes in-register (no whole-cache dequant pass).

Flash-Decoding (Dao et al., Stanford CRFM 2023) on both decode kernels:
the KV dimension is additionally partitioned into splits so small batches
still saturate the GPU. Each split computes an unnormalized local
(m, l, acc); a shared reduce kernel merges them with the standard
max-rescale formula. Large batches degenerate to the single-split path
(WRITE_MID=False), which is identical to the non-split kernel.

All kernels are gated behind `Attention.use_triton_attn` in `attention.py`
— default behavior (flash_attn package) is untouched.

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


_NUM_SMS = None   # cached lazily: torch.cuda.get_device_properties is too
                  # expensive to call on every decode launch (10-30% overhead).


def _num_splits_for(num_seqs: int, num_heads: int) -> int:
    """Pick the KV split count (flash-decoding) for a decode batch.

    A single-query decode grid is (seqs × heads) programs; small batches
    leave most SMs idle (e.g. batch=1 × 24 heads on 128 SMs uses 19%).
    Splitting the KV dimension restores occupancy. Rounded up to a power
    of two for the reduce kernel's `tl.arange`. Static inputs only — no
    GPU→CPU sync, so this stays CUDA-graph friendly.
    """
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    want = max(1, -(-(_NUM_SMS * 4) // (num_seqs * num_heads)))
    return 1 << (want - 1).bit_length()


@triton.jit
def _split_reduce_kernel(
    mid_ptr, o_ptr,
    stride_mid_sh, stride_mid_ss,
    stride_on, stride_oh,
    num_heads: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Merge the per-split softmax states written by a decode kernel.

    Each split s stored its unnormalized (m_s, l_s, acc_s); the merged
    output uses the standard rescale formula:
        m_g   = max_s m_s
        r_s   = exp(m_s - m_g)          # empty splits: m_s=-inf → r_s=0
        out   = (Σ_s r_s·acc_s) / (Σ_s r_s·l_s)
    Shared by the BF16 and INT8 decode kernels (same mid layout).
    """
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, HEAD_DIM)

    base = mid_ptr + (seq_idx * num_heads + head_idx) * stride_mid_sh + offs_s * stride_mid_ss
    m_s = tl.load(base + 0)                                   # (SPLITS,)
    l_s = tl.load(base + 1)                                   # (SPLITS,)
    acc_s = tl.load(base[:, None] + 2 + offs_d[None, :])      # (SPLITS, HEAD_DIM)

    m_g = tl.max(m_s[None, :], axis=1)                        # (1,)
    r = tl.exp(m_s - m_g)                                     # (SPLITS,)
    l_g = tl.sum((l_s * r)[None, :], axis=1)                  # (1,)
    acc_g = tl.sum(acc_s * r[:, None], axis=0)                # (HEAD_DIM,)

    out = acc_g / l_g
    tl.store(o_ptr + seq_idx * stride_on + head_idx * stride_oh + offs_d,
             out.to(o_ptr.dtype.element_ty))


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
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr, mid_ptr,
    block_tables_ptr, context_lens_ptr,
    scale,
    stride_qn, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_on, stride_oh,
    stride_mid_sh, stride_mid_ss,
    stride_bt,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # tokens per cache block (kvcache_block_size)
    BLOCK_N: tl.constexpr,      # kv tokens per iteration; must divide BLOCK_SIZE
    NUM_SPLITS: tl.constexpr,   # flash-decoding KV splits (1 = single-query path)
    WRITE_MID: tl.constexpr,    # True: write (m,l,acc) to mid (multi-split);
                                # False: write final output directly (single split)
):
    # One program per (sequence, query head, KV split): decode has a single
    # query token, so softmax state is scalar and the accumulator is a
    # HEAD_DIM vector.
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    context_len = tl.load(context_lens_ptr + seq_idx)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + seq_idx * stride_qn + head_idx * stride_qh + offs_d)

    # Running online-softmax state. Kept as shape-(1,) tensors on purpose:
    # 0-d (shape-[]) scalars mixed with 1-D/2-D operands in broadcasts are
    # unreliable across Triton versions (produced all-NaN output here), while
    # this (1,)-state layout is structurally identical to the verified
    # prefill kernel's (BLOCK_M,)-state — same code path, just BLOCK_M == 1.
    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Flash-decoding: partition KV into NUM_SPLITS segments, aligned to
    # BLOCK_N so a tile never spans two segments. Empty segments (n_lo past
    # the end) keep the initial (-inf, 0, 0) state — a zero contribution in
    # the reduce kernel, so no special-casing is needed.
    # NUM_SPLITS == 1 keeps the original constant lower bound: a runtime
    # n_lo (even though always 0) measurably blocks loop optimizations.
    if NUM_SPLITS == 1:
        n_lo = 0
        n_hi = context_len
    else:
        seg_len = tl.cdiv(tl.cdiv(context_len, BLOCK_N), NUM_SPLITS) * BLOCK_N
        n_lo = split_idx * seg_len
        n_hi = tl.minimum(n_lo + seg_len, context_len)

    # The query is the last token, so every key position [0, context_len) is
    # visible — no causal mask needed, only the context_len boundary.
    # BLOCK_N divides BLOCK_SIZE, so each tile lies inside exactly one cache
    # block and a single block_table lookup per tile suffices. NOTE: the mask
    # uses GLOBAL logical positions, but addressing must use WITHIN-BLOCK
    # offsets — conflating the two reads past the block boundary once a
    # sequence spans its second block (offsets >= block_size).
    for n_start in range(n_lo, n_hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)                       # global positions (mask)
        mask_n = offs_n < context_len
        offs_in_block = n_start % BLOCK_SIZE + tl.arange(0, BLOCK_N)   # within-block offsets (addressing)

        physical_block = tl.load(block_tables_ptr + seq_idx * stride_bt + n_start // BLOCK_SIZE)

        k_off = (physical_block.to(tl.int64) * stride_kb
                 + offs_in_block[:, None] * stride_kt + kv_head_idx * stride_kh + offs_d[None, :])
        k = tl.load(k_cache_ptr + k_off, mask=mask_n[:, None], other=0.0)

        qk = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1) * scale
        qk = tl.where(mask_n, qk, float("-inf"))

        m_ij = tl.max(qk[None, :], axis=1)          # (1,)
        m_new = tl.maximum(m_i, m_ij)               # (1,)
        p = tl.exp(qk - m_new)                      # (BLOCK_N,) - (1,) -> (BLOCK_N,)
        alpha = tl.exp(m_i - m_new)                 # (1,)

        v_off = (physical_block.to(tl.int64) * stride_vb
                 + offs_in_block[:, None] * stride_vt + kv_head_idx * stride_vh + offs_d[None, :])
        v = tl.load(v_cache_ptr + v_off, mask=mask_n[:, None], other=0.0)

        l_i = l_i * alpha + tl.sum(p[None, :], axis=1)          # (1,)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    if WRITE_MID:
        # Store the UNNORMALIZED partial state; _split_reduce_kernel applies
        # the final softmax normalization across splits.
        base = mid_ptr + (seq_idx * num_heads + head_idx) * stride_mid_sh + split_idx * stride_mid_ss
        one = tl.arange(0, 1)
        tl.store(base + one, m_i)                       # (1,) at offset 0
        tl.store(base + 1 + one, l_i)                   # (1,) at offset 1
        tl.store(base + 2 + offs_d, acc)                # (HEAD_DIM,)
    else:
        acc = acc / l_i                                 # (HEAD_DIM,) / (1,) -> (HEAD_DIM,)
        tl.store(o_ptr + seq_idx * stride_on + head_idx * stride_oh + offs_d,
                 acc.to(o_ptr.dtype.element_ty))


def triton_paged_attention(
    q: torch.Tensor,             # (num_seqs, num_heads, head_dim)
    k_cache: torch.Tensor,       # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,       # same layout as k_cache
    block_tables: torch.Tensor,  # (num_seqs, max_num_blocks) int32, logical→physical block map
    context_lens: torch.Tensor,  # (num_seqs,) int32, total tokens cached per seq
    scale: float,
    num_splits: int | None = None,   # flash-decoding splits; None = auto (by batch size)
) -> torch.Tensor:
    """Single-query paged attention over the paged BF16 KV cache (decode path).

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
    if num_splits is None:
        num_splits = _num_splits_for(num_seqs, num_heads)

    o = torch.empty_like(q)
    if num_splits == 1:
        # Large batches already saturate the GPU: single-pass, direct output.
        _paged_attn_decode_kernel[(num_seqs, num_heads, 1)](
            q, k_cache, v_cache, o, o,        # mid_ptr unused (WRITE_MID=False)
            block_tables, context_lens,
            scale,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            o.stride(0), o.stride(1),
            0, 0,                             # mid strides unused
            block_tables.stride(0),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            BLOCK_N=BLOCK_N,
            NUM_SPLITS=1,
            WRITE_MID=False,
            num_warps=4,
        )
        return o

    # Small batches: partition KV across splits, then reduce. Every program
    # writes its split slot unconditionally (no early return in the kernel),
    # so uninitialized reads cannot happen and empty is safe.
    mid = torch.empty(num_seqs * num_heads, num_splits, head_dim + 2,
                      dtype=torch.float32, device=q.device)
    _paged_attn_decode_kernel[(num_seqs, num_heads, num_splits)](
        q, k_cache, v_cache, o, mid,
        block_tables, context_lens,
        scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        o.stride(0), o.stride(1),
        mid.stride(0), mid.stride(1),
        block_tables.stride(0),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_N=BLOCK_N,
        NUM_SPLITS=num_splits,
        WRITE_MID=True,
        num_warps=4,
    )
    _split_reduce_kernel[(num_seqs, num_heads)](
        mid, o,
        mid.stride(0), mid.stride(1),
        o.stride(0), o.stride(1),
        num_heads=num_heads,
        NUM_SPLITS=num_splits,
        HEAD_DIM=head_dim,
    )
    return o


@triton.jit
def _paged_attn_decode_int8_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, o_ptr, mid_ptr,
    block_tables_ptr, context_lens_ptr,
    scale,
    stride_qn, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_ksc_b, stride_ksc_t,
    stride_vsc_b, stride_vsc_t,
    stride_on, stride_oh,
    stride_mid_sh, stride_mid_ss,
    stride_bt,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # tokens per cache block (kvcache_block_size)
    BLOCK_N: tl.constexpr,      # kv tokens per iteration; must divide BLOCK_SIZE
    NUM_SPLITS: tl.constexpr,   # flash-decoding KV splits (1 = single-query path)
    WRITE_MID: tl.constexpr,    # True: write (m,l,acc) to mid; False: direct output
):
    """Fused INT8 decode attention: reads the INT8 paged cache directly and
    dequantizes in-register — no whole-cache dequant pass, no intermediate
    BF16 buffer. Used when kv_quant=True and use_triton_attn=True.

    Per-(token, head) symmetric quantization lets the scale float out of the
    dot product (scale is constant along head_dim), so we compute in int8 and
    post-multiply the scale once per token instead of rescaling every element:
        qk[t]  = k_scale[t] * softmax_scale * Σ_d q[d]·k_int8[t,d]
        acc[d] += (p[t]·v_scale[t]) * v_int8[t,d]
    """
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    context_len = tl.load(context_lens_ptr + seq_idx)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + seq_idx * stride_qn + head_idx * stride_qh + offs_d).to(tl.float32)

    # Running online-softmax state (shape (1,) — see BF16 kernel note).
    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Flash-decoding segment partition (see BF16 kernel note). The constexpr
    # single-split branch keeps the original constant loop lower bound.
    if NUM_SPLITS == 1:
        n_lo = 0
        n_hi = context_len
    else:
        seg_len = tl.cdiv(tl.cdiv(context_len, BLOCK_N), NUM_SPLITS) * BLOCK_N
        n_lo = split_idx * seg_len
        n_hi = tl.minimum(n_lo + seg_len, context_len)

    for n_start in range(n_lo, n_hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)                       # global (mask)
        mask_n = offs_n < context_len
        offs_in_block = n_start % BLOCK_SIZE + tl.arange(0, BLOCK_N)   # within-block (addressing)

        physical_block = tl.load(block_tables_ptr + seq_idx * stride_bt + n_start // BLOCK_SIZE)

        # --- K: int8 dot product in fp32, then per-(token, head) scale ---
        k_off = (physical_block.to(tl.int64) * stride_kb
                 + offs_in_block[:, None] * stride_kt + kv_head_idx * stride_kh + offs_d[None, :])
        k_i8 = tl.load(k_cache_ptr + k_off, mask=mask_n[:, None], other=0.0)
        qk = tl.sum(q[None, :] * k_i8.to(tl.float32), axis=1)

        k_sc = tl.load(k_scale_ptr + physical_block.to(tl.int64) * stride_ksc_b
                       + offs_in_block * stride_ksc_t + kv_head_idx,
                       mask=mask_n, other=0.0)   # 0.0: unmasked garbage may be NaN → 0*x stays safe
        qk = qk * k_sc * scale
        qk = tl.where(mask_n, qk, float("-inf"))

        # --- online softmax ---
        m_ij = tl.max(qk[None, :], axis=1)          # (1,)
        m_new = tl.maximum(m_i, m_ij)               # (1,)
        p = tl.exp(qk - m_new)                      # (BLOCK_N,)
        alpha = tl.exp(m_i - m_new)                 # (1,)

        # --- V: merge p with v_scale into one per-token weight, then int8 FMA ---
        v_off = (physical_block.to(tl.int64) * stride_vb
                 + offs_in_block[:, None] * stride_vt + kv_head_idx * stride_vh + offs_d[None, :])
        v_i8 = tl.load(v_cache_ptr + v_off, mask=mask_n[:, None], other=0.0)
        v_sc = tl.load(v_scale_ptr + physical_block.to(tl.int64) * stride_vsc_b
                       + offs_in_block * stride_vsc_t + kv_head_idx,
                       mask=mask_n, other=0.0)
        w = p * v_sc                                # both are 0 on padding → no NaN leak

        l_i = l_i * alpha + tl.sum(p[None, :], axis=1)
        acc = acc * alpha + tl.sum(w[:, None] * v_i8.to(tl.float32), axis=0)
        m_i = m_new

    if WRITE_MID:
        base = mid_ptr + (seq_idx * num_heads + head_idx) * stride_mid_sh + split_idx * stride_mid_ss
        one = tl.arange(0, 1)
        tl.store(base + one, m_i)
        tl.store(base + 1 + one, l_i)
        tl.store(base + 2 + offs_d, acc)
    else:
        acc = acc / l_i
        tl.store(o_ptr + seq_idx * stride_on + head_idx * stride_oh + offs_d,
                 acc.to(o_ptr.dtype.element_ty))


def triton_paged_attention_int8(
    q: torch.Tensor,             # (num_seqs, num_heads, head_dim) BF16
    k_cache: torch.Tensor,       # (num_blocks, block_size, num_kv_heads, head_dim) INT8
    v_cache: torch.Tensor,       # same layout as k_cache, INT8
    k_scale: torch.Tensor,       # (num_blocks, block_size, num_kv_heads) FP32
    v_scale: torch.Tensor,       # same layout as k_scale
    block_tables: torch.Tensor,  # (num_seqs, max_num_blocks) int32
    context_lens: torch.Tensor,  # (num_seqs,) int32
    scale: float,
    num_splits: int | None = None,   # flash-decoding splits; None = auto (by batch size)
) -> torch.Tensor:
    """Single-query paged attention over the paged INT8 KV cache (decode path).

    A fused alternative to `dequant_kvcache + flash_attn_with_kvcache`:
    reads INT8 once and dequantizes in-register instead of materializing a
    whole-cache BF16 buffer every decode step. Returns a tensor shaped like q.
    """
    num_seqs, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads (GQA)"
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of 2 (tl.arange requirement)"
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
    assert k_scale.stride(-1) == 1 and v_scale.stride(-1) == 1

    BLOCK_N = 64    # divides block_size (256), so one tile never spans two cache blocks
    assert block_size % BLOCK_N == 0
    if num_splits is None:
        num_splits = _num_splits_for(num_seqs, num_heads)

    o = torch.empty_like(q)
    if num_splits == 1:
        _paged_attn_decode_int8_kernel[(num_seqs, num_heads, 1)](
            q, k_cache, v_cache, k_scale, v_scale, o, o,   # mid_ptr unused
            block_tables, context_lens,
            scale,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            k_scale.stride(0), k_scale.stride(1),
            v_scale.stride(0), v_scale.stride(1),
            o.stride(0), o.stride(1),
            0, 0,                                         # mid strides unused
            block_tables.stride(0),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            BLOCK_N=BLOCK_N,
            NUM_SPLITS=1,
            WRITE_MID=False,
            num_warps=4,
        )
        return o

    # See the BF16 wrapper: every split slot is written unconditionally.
    mid = torch.empty(num_seqs * num_heads, num_splits, head_dim + 2,
                      dtype=torch.float32, device=q.device)
    _paged_attn_decode_int8_kernel[(num_seqs, num_heads, num_splits)](
        q, k_cache, v_cache, k_scale, v_scale, o, mid,
        block_tables, context_lens,
        scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        v_scale.stride(0), v_scale.stride(1),
        o.stride(0), o.stride(1),
        mid.stride(0), mid.stride(1),
        block_tables.stride(0),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_N=BLOCK_N,
        NUM_SPLITS=num_splits,
        WRITE_MID=True,
        num_warps=4,
    )
    _split_reduce_kernel[(num_seqs, num_heads)](
        mid, o,
        mid.stride(0), mid.stride(1),
        o.stride(0), o.stride(1),
        num_heads=num_heads,
        NUM_SPLITS=num_splits,
        HEAD_DIM=head_dim,
    )
    return o
