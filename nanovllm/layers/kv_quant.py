"""INT8 KV cache quantization.

Granularity is split by tensor because they have different sensitivity:
  * K — per-head static scale (shared across all tokens). Attention scores
    compare K across tokens (softmax over the sequence), so a per-token scale
    would distort the relative magnitudes. The static per-head scale is
    produced once by calibration (see `calibrate_k_scales`).
  * V — per-(token, head) dynamic scale. V only contributes to a weighted
    sum, so per-token scaling is safe and needs no calibration.

Store: BF16 K/V → INT8 cache + scales.
Decode: INT8 cache → BF16 buffer (fallback path, passed to flash_attn) or
        dequantized in-register by the fused Triton kernel (main path).
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
    """Quantize BF16 K/V to INT8 at store time. One program per (token, head).

    K uses the pre-calibrated per-head scale (a scalar read); V computes its
    per-(token, head) scale on the fly.
    """
    idx = tl.program_id(0)      # token index
    head = tl.program_id(1)     # kv head index
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    col = tl.arange(0, HEAD_DIM)
    # --- Key: static per-head scale ---
    k_in = tl.load(key_ptr + idx * key_stride + head * HEAD_DIM + col).to(tl.float32)
    k_scale = tl.load(k_scale_ptr + head)              # (num_kv_heads,) static
    k_int8 = (k_in / k_scale).to(tl.int8)
    tl.store(k_cache_ptr + slot * NUM_KV_HEADS * HEAD_DIM + head * HEAD_DIM + col, k_int8)

    # --- Value: dynamic per-(token, head) scale ---
    v_in = tl.load(value_ptr + idx * value_stride + head * HEAD_DIM + col).to(tl.float32)
    v_scale = tl.max(tl.abs(v_in)) / 127.0
    v_int8 = (v_in / v_scale).to(tl.int8)
    tl.store(v_cache_ptr + slot * NUM_KV_HEADS * HEAD_DIM + head * HEAD_DIM + col, v_int8)
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
    PER_HEAD: tl.constexpr,
):
    """Dequantize INT8 cache to BF16. One program per (token, head).

    PER_HEAD selects the scale layout: True → scale[head] (K), False →
    scale[token, head] (V).
    """
    idx = tl.program_id(0)
    head = tl.program_id(1)
    if idx >= num_tokens:
        return
    col = tl.arange(0, HEAD_DIM)
    if PER_HEAD:
        scale = tl.load(scale_ptr + head)
    else:
        scale = tl.load(scale_ptr + idx * NUM_KV_HEADS + head)
    off = idx * NUM_KV_HEADS * HEAD_DIM + head * HEAD_DIM + col
    val_int8 = tl.load(cache_int8_ptr + off).to(tl.float32)
    val_bf16 = (val_int8 * scale).to(tl.bfloat16)
    tl.store(out_ptr + off, val_bf16)


def dequant_kvcache(
    cache_int8: torch.Tensor, scale: torch.Tensor, per_head: bool = False,
) -> torch.Tensor:
    """Dequantize the full INT8 paged cache to BF16 (fallback path).

    cache_int8 shape: (num_blocks, block_size, num_kv_heads, head_dim).
    scale shape:      (num_kv_heads,) if per_head else (num_blocks, block_size, num_kv_heads).
    Returns a BF16 tensor with the same 4D shape as cache_int8.
    """
    num_blocks, block_size, num_kv_heads, head_dim = cache_int8.shape
    out = torch.empty_like(cache_int8, dtype=torch.bfloat16)
    total_tokens = num_blocks * block_size
    cache_flat = cache_int8.reshape(total_tokens, num_kv_heads, head_dim)
    scale_flat = scale if per_head else scale.reshape(total_tokens, num_kv_heads)
    out_flat = out.reshape(total_tokens, num_kv_heads, head_dim)
    dequant_kvcache_kernel[(total_tokens, num_kv_heads)](
        cache_flat, scale_flat, out_flat,
        total_tokens,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        PER_HEAD=per_head,
    )
    return out


def calibrate_k_scales(
    model: torch.nn.Module,
    num_layers: int,
    num_kv_heads: int,
    num_calib_tokens: int = 512,
) -> torch.Tensor:
    """Calibrate static per-head K scales.

    Runs one forward over `num_calib_tokens` random tokens and hooks each
    attention module's K input (the post-RoPE key, args[1] of Attention.forward),
    recording the per-head abs-max. Returns a (num_layers, num_kv_heads) scale
    tensor (max|K| / 127, clamped positive). This keeps K's scale shared
    across tokens, preserving the cross-token consistency that softmax needs.
    """
    from nanovllm.utils.context import set_context, reset_context

    device = next(model.parameters()).device
    seq_len = num_calib_tokens
    vocab_size = model.model.embed_tokens.num_embeddings \
        if hasattr(model.model.embed_tokens, "num_embeddings") else 128256

    # Per-layer running abs-max over K's (num_kv_heads,) head dimension.
    k_absmax = [torch.zeros(num_kv_heads, device=device) for _ in range(num_layers)]

    hooks = []
    layer_idx = 0

    def make_hook(slot: int):
        def pre_hook(module, args):
            k = args[1]                       # (tokens, num_kv_heads, head_dim)
            amax = k.float().abs().amax(dim=(0, 2))          # (num_kv_heads,)
            k_absmax[slot] = torch.maximum(k_absmax[slot], amax)
        return pre_hook

    # Hook every Attention module in forward order; their k_cache attribute is
    # the same marker ModelRunner uses to find attention layers.
    for module in model.modules():
        if hasattr(module, "k_cache"):
            hooks.append(module.register_forward_pre_hook(make_hook(layer_idx)))
            layer_idx += 1
    assert layer_idx == num_layers, f"expected {num_layers} attention layers, found {layer_idx}"

    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    # Prefill context so attention uses freshly-computed K/V (no KV-cache read).
    set_context(True, cu_seqlens, cu_seqlens, seq_len, seq_len, None, None, None)
    try:
        with torch.no_grad():
            model(input_ids, positions)
    finally:
        for h in hooks:
            h.remove()
        reset_context()

    scales = torch.stack(k_absmax).div(127.0).clamp_min(1e-6)   # (num_layers, num_kv_heads)
    return scales
