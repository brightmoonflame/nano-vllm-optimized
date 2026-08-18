"""INT8 KV cache quantization — per-(token, head, group) symmetric Min-Max.

Both K and V quantize in groups along head_dim (e.g. 8 groups of 16 dims):
a per-(token, head) scale lets a single outlier dimension dominate the scale
and crush the precision of the other 127 dims, while a per-head static scale
requires calibration and wastes most of INT8's uniform range on small-magnitude
tokens. Group-wise scales isolate outliers in their own group and need no
calibration — scales are computed dynamically at store time.

Store: BF16 K/V → INT8 cache + FP32 scale (num_groups per token per head).
Decode: INT8 cache → BF16 buffer (fallback path) or dequantized in-register
        by the fused Triton kernel (main path).
Prefill: unaffected (uses freshly computed K/V directly).
"""

import torch
import triton
import triton.language as tl

NUM_GROUPS = 8   # groups along head_dim; head_dim must be divisible by this


@triton.jit
def store_kvcache_int8_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    k_scale_ptr, v_scale_ptr,
    slot_mapping_ptr,
    HEAD_DIM: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,    # HEAD_DIM // NUM_GROUPS
    NUM_GROUPS: tl.constexpr,
):
    """Quantize BF16 K/V to INT8 at store time. One program per (token, head).

    Each of the NUM_GROUPS slices along head_dim gets its own scale, computed
    dynamically from that slice's abs-max — no calibration, no overflow.
    """
    idx = tl.program_id(0)      # token index
    head = tl.program_id(1)     # kv head index
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    col = tl.arange(0, HEAD_DIM)
    base_off = slot * NUM_KV_HEADS * HEAD_DIM + head * HEAD_DIM

    # --- Key ---
    k_in = tl.load(key_ptr + idx * key_stride + head * HEAD_DIM + col).to(tl.float32)
    # (HEAD_DIM,) → (NUM_GROUPS, GROUP_SIZE): per-group abs-max.
    k_g = tl.reshape(k_in, (NUM_GROUPS, GROUP_SIZE))
    k_scale = tl.max(tl.abs(k_g), axis=1) / 127.0                    # (NUM_GROUPS,)
    k_int8 = (k_in / tl.reshape(tl.broadcast_to(
        k_scale[:, None], (NUM_GROUPS, GROUP_SIZE)), (HEAD_DIM,))).to(tl.int8)
    tl.store(k_cache_ptr + base_off + col, k_int8)
    tl.store(k_scale_ptr + slot * NUM_KV_HEADS * NUM_GROUPS + head * NUM_GROUPS
             + tl.arange(0, NUM_GROUPS), k_scale)

    # --- Value (same recipe) ---
    v_in = tl.load(value_ptr + idx * value_stride + head * HEAD_DIM + col).to(tl.float32)
    v_g = tl.reshape(v_in, (NUM_GROUPS, GROUP_SIZE))
    v_scale = tl.max(tl.abs(v_g), axis=1) / 127.0
    v_int8 = (v_in / tl.reshape(tl.broadcast_to(
        v_scale[:, None], (NUM_GROUPS, GROUP_SIZE)), (HEAD_DIM,))).to(tl.int8)
    tl.store(v_cache_ptr + base_off + col, v_int8)
    tl.store(v_scale_ptr + slot * NUM_KV_HEADS * NUM_GROUPS + head * NUM_GROUPS
             + tl.arange(0, NUM_GROUPS), v_scale)


def store_kvcache_int8(
    key: torch.Tensor, value: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_scale: torch.Tensor, v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Wrapper: launch one program per (token, kv_head)."""
    N, num_kv_heads, head_dim = key.shape
    assert head_dim % NUM_GROUPS == 0
    store_kvcache_int8_kernel[(N, num_kv_heads)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, k_scale, v_scale,
        slot_mapping,
        HEAD_DIM=head_dim,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=head_dim // NUM_GROUPS,
        NUM_GROUPS=NUM_GROUPS,
    )


@triton.jit
def dequant_kvcache_kernel(
    cache_int8_ptr, scale_ptr, out_ptr,
    num_tokens, NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    """Dequantize INT8 cache to BF16. One program per (token, head).

    Scale layout: (token, head, group) — expanded along head_dim to multiply.
    """
    idx = tl.program_id(0)
    head = tl.program_id(1)
    if idx >= num_tokens:
        return
    col = tl.arange(0, HEAD_DIM)
    g = tl.arange(0, NUM_GROUPS)
    sc = tl.load(scale_ptr + idx * NUM_KV_HEADS * NUM_GROUPS + head * NUM_GROUPS + g)   # (G,)
    sc_full = tl.reshape(tl.broadcast_to(sc[:, None], (NUM_GROUPS, GROUP_SIZE)), (HEAD_DIM,))
    off = idx * NUM_KV_HEADS * HEAD_DIM + head * HEAD_DIM + col
    val_int8 = tl.load(cache_int8_ptr + off).to(tl.float32)
    tl.store(out_ptr + off, (val_int8 * sc_full).to(tl.bfloat16))


def dequant_kvcache(
    cache_int8: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize the full INT8 paged cache to BF16 (fallback path).

    cache_int8 shape: (num_blocks, block_size, num_kv_heads, head_dim).
    scale shape:      (num_blocks, block_size, num_kv_heads, NUM_GROUPS).
    Returns a BF16 tensor with the same 4D shape as cache_int8.
    """
    num_blocks, block_size, num_kv_heads, head_dim = cache_int8.shape
    out = torch.empty_like(cache_int8, dtype=torch.bfloat16)
    total_tokens = num_blocks * block_size
    cache_flat = cache_int8.reshape(total_tokens, num_kv_heads, head_dim)
    scale_flat = scale.reshape(total_tokens, num_kv_heads * NUM_GROUPS)
    out_flat = out.reshape(total_tokens, num_kv_heads, head_dim)
    dequant_kvcache_kernel[(total_tokens, num_kv_heads)](
        cache_flat, scale_flat, out_flat,
        total_tokens,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        GROUP_SIZE=head_dim // NUM_GROUPS,
        NUM_GROUPS=NUM_GROUPS,
    )
    return out
