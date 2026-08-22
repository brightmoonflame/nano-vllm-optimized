"""Self-researched Triton attention kernels (forward-only).

Two unified entry points, each a drop-in alternative to the corresponding
flash_attn function, selected by optional arguments via compile-time switches:

  triton_flash_attn_varlen  — FlashAttention-2 forward (prefill).
      IS_PAGED=False: dense varlen; True: paged (prefix-cache hit, K/V read
      from the paged cache via block_tables + dual cu_seqlens_q/k + offset
      causal). IS_INT8=True additionally dequantizes the INT8 cache in-register.

  triton_paged_attention     — single-query paged attention (decode), BF16 by
      default; IS_INT8=True reads the INT8 paged cache directly and dequantizes
      in-register (no whole-cache dequant pass).

Flash-Decoding (Dao et al., Stanford CRFM 2023) on the decode kernel:
the KV dimension is additionally partitioned into splits so small batches
still saturate the GPU. Each split computes an unnormalized local
(m, l, acc); a shared reduce kernel merges them with the standard
max-rescale formula. Large batches degenerate to the single-split path
(WRITE_MID=False), which is identical to the non-split path.

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

# Keep the group count in one place (shared with kv_quant / model_runner).
NUM_GROUPS = 8   # scale groups along head_dim (128 / 8 = 16 dims per group)


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
    # Target 2 programs/SM (not 4): each split then handles 2+ tiles, giving the
    # loop enough iterations to pipeline loads (1-tile splits expose full load
    # latency). Fewer, fatter splits measured faster than many 1-tile splits.
    want = max(1, -(-(_NUM_SMS * 2) // (num_seqs * num_heads)))
    return 1 << (want - 1).bit_length()


def mid_buffer_size(num_heads: int, head_dim: int, max_bs: int) -> int:
    """Element count for a shared flash-decoding mid buffer that covers any
    decode batch size: max over bs of (bs*heads) * splits(bs) * (head_dim+2),
    counting only splits > 1 batches (single-split never touches mid).

    One buffer can be shared by all attention layers — layers execute
    sequentially within a forward, so at most one layer uses it at a time.
    """
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    target = _NUM_SMS * 2
    best = 0
    for bs in range(1, max_bs + 1):
        p = bs * num_heads
        splits = 1 << (max(1, -(-target // p)) - 1).bit_length()
        if splits > 1:
            best = max(best, p * splits * (head_dim + 2))
    return best


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
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    block_tables_ptr,
    k_scale_ptr, v_scale_ptr,
    stride_qn, stride_qh,
    stride_kt, stride_kh, stride_kb,
    stride_vt, stride_vh, stride_vb,
    stride_ksc_b, stride_ksc_t,
    stride_vsc_b, stride_vsc_t,
    stride_on, stride_oh,
    stride_bt,
    scale,
    IS_PAGED: tl.constexpr,
    IS_INT8: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Unified FlashAttention-2 forward (prefill) with two compile-time switches:

      IS_PAGED  False: dense varlen K/V (stride_kt is the token stride);
                True:  paged K/V via block_tables (stride_kb/stride_kt).
      IS_INT8   False: K/V are BF16; True: INT8 + group-wise scale, dequantized
                in-register back to the query dtype before the dot products.

    Dense is IS_PAGED=False with cu_seqlens_k == cu_seqlens_q, so seqlen_k ==
    seqlen_q and start == 0 — the offset causal reduces to standard causal. The
    INT8 variant is only ever used paged (dense prefill consumes fresh BF16 K/V).
    """
    m_block = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    seq_start_q = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end_q = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_start_k = tl.load(cu_seqlens_k_ptr + seq_idx)
    seq_end_k = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
    seqlen_q = seq_end_q - seq_start_q
    seqlen_k = seq_end_k - seq_start_k
    start = seqlen_k - seqlen_q   # cached prefix length (0 for dense)

    m_start = m_block * BLOCK_M
    if m_start >= seqlen_q:
        return    # this query block lies entirely past this sequence's end

    # GQA: each group of (num_heads // num_kv_heads) query heads shares one KV head.
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q

    q_ptrs = q_ptr + (seq_start_q + offs_m)[:, None] * stride_qn + head_idx * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Running online-softmax state: row-wise max, row-wise sumexp, weighted accumulator.
    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Offset causal: query row at global position (start + offs_m) attends keys
    # [0, start + offs_m]; the exclusive loop bound is clamped to the key length.
    n_end = tl.minimum(start + m_start + BLOCK_M, seqlen_k)
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)                     # seq-local key positions
        mask_n = offs_n < seqlen_k

        if IS_PAGED:
            offs_in_block = n_start % BLOCK_SIZE + tl.arange(0, BLOCK_N)
            physical_block = tl.load(block_tables_ptr + seq_idx * stride_bt + n_start // BLOCK_SIZE)
            k_off = (physical_block.to(tl.int64) * stride_kb
                     + offs_in_block[:, None] * stride_kt + kv_head_idx * stride_kh + offs_d[None, :])
            v_off = (physical_block.to(tl.int64) * stride_vb
                     + offs_in_block[:, None] * stride_vt + kv_head_idx * stride_vh + offs_d[None, :])
        else:
            k_off = (seq_start_k + offs_n)[:, None] * stride_kt + kv_head_idx * stride_kh + offs_d[None, :]
            v_off = (seq_start_k + offs_n)[:, None] * stride_vt + kv_head_idx * stride_vh + offs_d[None, :]

        if IS_INT8:
            # INT8 -> group-scale dequant -> query dtype, then standard dot. Uses
            # physical_block/offs_in_block from the IS_PAGED branch (INT8 is paged-only).
            offs_g = tl.arange(0, NUM_GROUPS)
            k_i8 = tl.load(k_ptr + k_off, mask=mask_n[:, None], other=0.0)
            ks = tl.load(k_scale_ptr + physical_block.to(tl.int64) * stride_ksc_b
                         + offs_in_block[:, None] * stride_ksc_t + kv_head_idx * NUM_GROUPS + offs_g[None, :],
                         mask=mask_n[:, None], other=0.0)
            ks_full = tl.reshape(tl.broadcast_to(ks[:, :, None], (BLOCK_N, NUM_GROUPS, GROUP_SIZE)),
                                 (BLOCK_N, HEAD_DIM))
            k = (k_i8.to(tl.float32) * ks_full).to(q_ptr.dtype.element_ty)
        else:
            k = tl.load(k_ptr + k_off, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        causal_mask = (start + offs_m)[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask & mask_n[None, :], qk, float("-inf"))

        # Rescale the running state to the new (larger) row max, then fold in this block.
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        if IS_INT8:
            v_i8 = tl.load(v_ptr + v_off, mask=mask_n[:, None], other=0.0)
            vs = tl.load(v_scale_ptr + physical_block.to(tl.int64) * stride_vsc_b
                         + offs_in_block[:, None] * stride_vsc_t + kv_head_idx * NUM_GROUPS + offs_g[None, :],
                         mask=mask_n[:, None], other=0.0)
            vs_full = tl.reshape(tl.broadcast_to(vs[:, :, None], (BLOCK_N, NUM_GROUPS, GROUP_SIZE)),
                                 (BLOCK_N, HEAD_DIM))
            v = (v_i8.to(tl.float32) * vs_full).to(q_ptr.dtype.element_ty)
        else:
            v = tl.load(v_ptr + v_off, mask=mask_n[:, None], other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + (seq_start_q + offs_m)[:, None] * stride_on + head_idx * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None])


def triton_flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    scale: float,
    cu_seqlens_k: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unified FlashAttention-2 forward (prefill), causal, GQA.

    One entry point for all prefill variants, selected by which optional
    arguments are passed:

      dense (no prefix cache):
        k/v are (total_tokens, num_kv_heads, head_dim); omit cu_seqlens_k,
        block_tables, k_scale, v_scale.
      paged BF16 (prefix-cache hit):
        k/v are the (num_blocks, block_size, num_kv_heads, head_dim) cache;
        pass cu_seqlens_k (prefix + new tokens) and block_tables.
      paged INT8 (prefix-cache hit + kv_quant):
        additionally pass k_scale/v_scale.

    Returns a tensor shaped like q.
    """
    IS_PAGED = block_tables is not None
    IS_INT8 = k_scale is not None
    assert (k_scale is None) == (v_scale is None), "k_scale and v_scale must be passed together"
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    total_q, num_heads, head_dim = q.shape
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of 2 (tl.arange requirement)"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "head_dim must be contiguous"

    if IS_PAGED:
        num_kv_heads = k.shape[2]
        block_size = k.shape[1]
        assert block_size % 64 == 0, "block_size must be a multiple of BLOCK_N=64 (single tile per block)"
        stride_kt, stride_kh, stride_kb = k.stride(1), k.stride(2), k.stride(0)
        stride_vt, stride_vh, stride_vb = v.stride(1), v.stride(2), v.stride(0)
        bt = block_tables
        bt_stride = block_tables.stride(0)
    else:
        num_kv_heads = k.shape[1]
        block_size = 1                       # unused when IS_PAGED=False
        stride_kt, stride_kh, stride_kb = k.stride(0), k.stride(1), 0
        stride_vt, stride_vh, stride_vb = v.stride(0), v.stride(1), 0
        bt = k                               # dummy (IS_PAGED=False never dereferences)
        bt_stride = 0

    assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads (GQA)"

    if IS_INT8:
        assert head_dim % NUM_GROUPS == 0
        ksc_b, ksc_t = k_scale.stride(0), k_scale.stride(1)
        vsc_b, vsc_t = v_scale.stride(0), v_scale.stride(1)
    else:
        k_scale = v_scale = k               # dummy (IS_INT8=False never dereferences)
        ksc_b = ksc_t = vsc_b = vsc_t = 0

    o = torch.empty_like(q)
    num_seqs = cu_seqlens_q.numel() - 1
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), num_heads, num_seqs)

    _fwd_kernel[grid](
        q, k, v, o,
        cu_seqlens_q, cu_seqlens_k,
        bt,
        k_scale, v_scale,
        q.stride(0), q.stride(1),
        stride_kt, stride_kh, stride_kb,
        stride_vt, stride_vh, stride_vb,
        ksc_b, ksc_t, vsc_b, vsc_t,
        o.stride(0), o.stride(1),
        bt_stride,
        scale,
        IS_PAGED=IS_PAGED,
        IS_INT8=IS_INT8,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        HEAD_DIM=head_dim,
        GROUP_SIZE=head_dim // NUM_GROUPS,
        NUM_GROUPS=NUM_GROUPS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return o


@triton.jit
def _paged_attn_decode_kernel(
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
    IS_INT8: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,   # HEAD_DIM // NUM_GROUPS
    NUM_GROUPS: tl.constexpr,   # scale groups along head_dim
    BLOCK_SIZE: tl.constexpr,   # tokens per cache block (kvcache_block_size)
    BLOCK_N: tl.constexpr,      # kv tokens per iteration; must divide BLOCK_SIZE
    NUM_SPLITS: tl.constexpr,   # flash-decoding KV splits (1 = single-query path)
    WRITE_MID: tl.constexpr,    # True: write (m,l,acc) to mid (multi-split);
                                # False: write final output directly (single split)
):
    """Single-query paged attention over the paged KV cache (decode path).

    One compile-time switch:
      IS_INT8  False: K/V are BF16; True: INT8 + group-wise scale, dequantized
               in-register (element-wise fp32 sum, since BLOCK_M == 1).

    One program per (sequence, query head, KV split): decode has a single
    query token, so softmax state is scalar and the accumulator is a
    HEAD_DIM vector.
    """
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    context_len = tl.load(context_lens_ptr + seq_idx)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_g = tl.arange(0, NUM_GROUPS)   # used only by the IS_INT8 branch
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
        v_off = (physical_block.to(tl.int64) * stride_vb
                 + offs_in_block[:, None] * stride_vt + kv_head_idx * stride_vh + offs_d[None, :])

        if IS_INT8:
            # INT8 -> group-scale dequantization in registers. Keep the
            # dequantized tile in BF16 so the following tl.dot can use the
            # same matrix-multiply path as the Prefill kernel.
            k_i8 = tl.load(k_cache_ptr + k_off, mask=mask_n[:, None], other=0.0)
            ks = tl.load(k_scale_ptr + physical_block.to(tl.int64) * stride_ksc_b
                         + offs_in_block[:, None] * stride_ksc_t + kv_head_idx * NUM_GROUPS + offs_g[None, :],
                         mask=mask_n[:, None], other=0.0)               # (BLOCK_N, NUM_GROUPS)
            ks_full = tl.reshape(tl.broadcast_to(ks[:, :, None], (BLOCK_N, NUM_GROUPS, GROUP_SIZE)),
                                 (BLOCK_N, HEAD_DIM))
            k = (k_i8.to(tl.float32) * ks_full).to(q_ptr.dtype.element_ty)
        else:
            k = tl.load(k_cache_ptr + k_off, mask=mask_n[:, None], other=0.0)

        # A (1, HEAD_DIM) × (HEAD_DIM, BLOCK_N) tile is small but maps to the
        # same tensor-core-friendly dot primitive used by Prefill. The former
        # element-wise FP32 reduction compiled to scalar FMA instructions.
        qk = tl.reshape(tl.dot(q[None, :], tl.trans(k)), (BLOCK_N,)) * scale

        qk = tl.where(mask_n, qk, float("-inf"))

        m_ij = tl.max(qk[None, :], axis=1)          # (1,)
        m_new = tl.maximum(m_i, m_ij)               # (1,)
        p = tl.exp(qk - m_new)                      # (BLOCK_N,) - (1,) -> (BLOCK_N,)
        alpha = tl.exp(m_i - m_new)                 # (1,)

        if IS_INT8:
            v_i8 = tl.load(v_cache_ptr + v_off, mask=mask_n[:, None], other=0.0)
            vs = tl.load(v_scale_ptr + physical_block.to(tl.int64) * stride_vsc_b
                         + offs_in_block[:, None] * stride_vsc_t + kv_head_idx * NUM_GROUPS + offs_g[None, :],
                         mask=mask_n[:, None], other=0.0)
            vs_full = tl.reshape(tl.broadcast_to(vs[:, :, None], (BLOCK_N, NUM_GROUPS, GROUP_SIZE)),
                                 (BLOCK_N, HEAD_DIM))
            v = (v_i8.to(tl.float32) * vs_full).to(q_ptr.dtype.element_ty)
        else:
            v = tl.load(v_cache_ptr + v_off, mask=mask_n[:, None], other=0.0)

        # Match Prefill's PV accumulation as well; p is cast to BF16 for the
        # tensor-core input while tl.dot still accumulates into FP32.
        pv = tl.reshape(tl.dot(p[None, :].to(v.dtype), v), (HEAD_DIM,))
        acc = acc * alpha + pv

        l_i = l_i * alpha + tl.sum(p[None, :], axis=1)          # (1,)
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


def _get_mid(mid: torch.Tensor | None, num_seqs: int, num_heads: int,
             num_splits: int, head_dim: int, device) -> torch.Tensor:
    """View of a pre-allocated mid buffer, or a fresh one if not provided.

    A pre-allocated buffer keeps the split path free of dynamic allocation —
    required for CUDA-graph capture and saves a per-step empty() in eager.
    narrow+view share storage (no alloc), so they are capture-safe.
    """
    need = num_seqs * num_heads * num_splits * (head_dim + 2)
    if mid is None:
        return torch.empty(num_seqs * num_heads, num_splits, head_dim + 2,
                           dtype=torch.float32, device=device)
    assert mid.numel() >= need, "mid buffer too small for this batch/split"
    return mid.narrow(0, 0, need).view(num_seqs * num_heads, num_splits, head_dim + 2)


def triton_paged_attention(
    q: torch.Tensor,             # (num_seqs, num_heads, head_dim)
    k_cache: torch.Tensor,       # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,       # same layout as k_cache
    block_tables: torch.Tensor,  # (num_seqs, max_num_blocks) int32, logical→physical block map
    context_lens: torch.Tensor,  # (num_seqs,) int32, total tokens cached per seq
    scale: float,
    num_splits: int | None = None,   # flash-decoding splits; None = auto (by batch size)
    mid: torch.Tensor | None = None, # pre-allocated split buffer (CUDA-graph friendly)
    k_scale: torch.Tensor | None = None,  # INT8 group scales (kv_quant=True)
    v_scale: torch.Tensor | None = None,  # same layout as k_scale
) -> torch.Tensor:
    """Unified single-query paged attention over the paged KV cache (decode).

    BF16 by default; pass k_scale/v_scale for the fused INT8 path (reads the
    INT8 cache once and dequantizes in-register — no whole-cache BF16 buffer).
    A drop-in alternative to `flash_attn_with_kvcache` for both dtypes.
    Returns a tensor shaped like q.
    """
    IS_INT8 = k_scale is not None
    assert (k_scale is None) == (v_scale is None), "k_scale and v_scale must be passed together"

    num_seqs, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads (GQA)"
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of 2 (tl.arange requirement)"
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1

    if IS_INT8:
        assert head_dim % NUM_GROUPS == 0
        ksc_b, ksc_t = k_scale.stride(0), k_scale.stride(1)
        vsc_b, vsc_t = v_scale.stride(0), v_scale.stride(1)
    else:
        k_scale = v_scale = k_cache         # dummy (IS_INT8=False never dereferences)
        ksc_b = ksc_t = vsc_b = vsc_t = 0

    BLOCK_N = 64    # divides block_size (256), so one tile never spans two cache blocks
    assert block_size % BLOCK_N == 0
    if num_splits is None:
        num_splits = _num_splits_for(num_seqs, num_heads)

    o = torch.empty_like(q)
    if num_splits == 1:
        # Large batches already saturate the GPU: single-pass, direct output.
        _paged_attn_decode_kernel[(num_seqs, num_heads, 1)](
            q, k_cache, v_cache, k_scale, v_scale, o, o,   # mid unused (WRITE_MID=False)
            block_tables, context_lens,
            scale,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            ksc_b, ksc_t, vsc_b, vsc_t,
            o.stride(0), o.stride(1),
            0, 0,                                           # mid strides unused
            block_tables.stride(0),
            IS_INT8=IS_INT8,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=head_dim,
            GROUP_SIZE=head_dim // NUM_GROUPS,
            NUM_GROUPS=NUM_GROUPS,
            BLOCK_SIZE=block_size,
            BLOCK_N=BLOCK_N,
            NUM_SPLITS=1,
            WRITE_MID=False,
            num_warps=4,
        )
        return o

    # Small batches: partition KV across splits, then reduce. Every program
    # writes its split slot unconditionally (no early return in the kernel),
    # so an empty buffer is safe; `mid` may be a shared pre-allocated buffer.
    mid = _get_mid(mid, num_seqs, num_heads, num_splits, head_dim, q.device)
    _paged_attn_decode_kernel[(num_seqs, num_heads, num_splits)](
        q, k_cache, v_cache, k_scale, v_scale, o, mid,
        block_tables, context_lens,
        scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        ksc_b, ksc_t, vsc_b, vsc_t,
        o.stride(0), o.stride(1),
        mid.stride(0), mid.stride(1),
        block_tables.stride(0),
        IS_INT8=IS_INT8,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        HEAD_DIM=head_dim,
        GROUP_SIZE=head_dim // NUM_GROUPS,
        NUM_GROUPS=NUM_GROUPS,
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


