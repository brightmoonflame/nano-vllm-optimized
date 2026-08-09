"""INT8 KV cache quantization — per-(token, head) symmetric Min-Max.

Store: BF16 K/V → INT8 cache + FP32 scale (one scale per token per head).
Decode: INT8 cache → BF16 buffer (passed to flash_attn_with_kvcache).
Prefill: unaffected (uses freshly computed K/V directly).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def store_kvcache_int8_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    k_scale_ptr, v_scale_ptr,
    slot_mapping_ptr,
    HEAD_DIM: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    """Quantize BF16 K/V to INT8 at store time. One program per (token, head)."""
    idx = tl.program_id(0)      # token index
    head = tl.program_id(1)     # kv head index
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    col = tl.arange(0, HEAD_DIM)
    # --- Key ---
    k_in = tl.load(key_ptr + idx * key_stride + head * HEAD_DIM + col).to(tl.float32)
    k_scale = tl.max(tl.abs(k_in)) / 127.0
    k_int8 = (k_in / k_scale).to(tl.int8)
    k_off = slot * NUM_KV_HEADS * HEAD_DIM + head * HEAD_DIM + col
    tl.store(k_cache_ptr + k_off, k_int8)
    tl.store(k_scale_ptr + slot * NUM_KV_HEADS + head, k_scale)

    # --- Value ---
    v_in = tl.load(value_ptr + idx * value_stride + head * HEAD_DIM + col).to(tl.float32)
    v_scale = tl.max(tl.abs(v_in)) / 127.0
    v_int8 = (v_in / v_scale).to(tl.int8)
    v_off = slot * NUM_KV_HEADS * HEAD_DIM + head * HEAD_DIM + col
    tl.store(v_cache_ptr + v_off, v_int8)
    tl.store(v_scale_ptr + slot * NUM_KV_HEADS + head, v_scale)


def store_kvcache_int8(
    key: torch.Tensor, value: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_scale: torch.Tensor, v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Wrapper: launch one program per (token, kv_head)."""
    N, num_kv_heads, head_dim = key.shape
    store_kvcache_int8_kernel[(N, num_kv_heads)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, k_scale, v_scale,
        slot_mapping,
        HEAD_DIM=head_dim,
        NUM_KV_HEADS=num_kv_heads,
    )


@triton.jit
def dequant_kvcache_kernel(
    cache_int8_ptr, scale_ptr, out_ptr,
    num_tokens, NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Dequantize INT8 cache to BF16. One program per (token, head)."""
    idx = tl.program_id(0)
    head = tl.program_id(1)
    if idx >= num_tokens:
        return
    col = tl.arange(0, HEAD_DIM)
    scale = tl.load(scale_ptr + idx * NUM_KV_HEADS + head)
    off = idx * NUM_KV_HEADS * HEAD_DIM + head * HEAD_DIM + col
    val_int8 = tl.load(cache_int8_ptr + off).to(tl.float32)
    val_bf16 = (val_int8 * scale).to(tl.bfloat16)
    tl.store(out_ptr + off, val_bf16)


def dequant_kvcache(
    cache_int8: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize the full INT8 paged cache to BF16.

    cache_int8 shape: (num_blocks, block_size, num_kv_heads, head_dim).
    scale shape:      (num_blocks, block_size, num_kv_heads).
    Returns a BF16 tensor with the same 4D shape as cache_int8.
    """
    num_blocks, block_size, num_kv_heads, head_dim = cache_int8.shape
    out = torch.empty_like(cache_int8, dtype=torch.bfloat16)
    total_tokens = num_blocks * block_size
    cache_flat = cache_int8.reshape(total_tokens, num_kv_heads, head_dim)
    scale_flat = scale.reshape(total_tokens, num_kv_heads)
    out_flat = out.reshape(total_tokens, num_kv_heads, head_dim)
    dequant_kvcache_kernel[(total_tokens, num_kv_heads)](
        cache_flat, scale_flat, out_flat,
        total_tokens,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
    )
    return out


def dequant_kvcache_to_buf(
    cache_int8: torch.Tensor, scale: torch.Tensor,
    out: torch.Tensor,
):
    """Dequantize INT8 cache into a pre-allocated BF16 buffer (no allocation)."""
    num_blocks, block_size, num_kv_heads, head_dim = cache_int8.shape
    total_tokens = num_blocks * block_size
    cache_flat = cache_int8.reshape(total_tokens, num_kv_heads, head_dim)
    scale_flat = scale.reshape(total_tokens, num_kv_heads)
    out_flat = out.reshape(total_tokens, num_kv_heads, head_dim)
    dequant_kvcache_kernel[(total_tokens, num_kv_heads)](
        cache_flat, scale_flat, out_flat,
        total_tokens,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
    )
