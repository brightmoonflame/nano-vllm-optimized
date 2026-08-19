import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.layers.kv_quant import store_kvcache_int8, dequant_kvcache
from nanovllm.layers.triton_attn import (
    triton_flash_attn_varlen,
    triton_paged_attention,
)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        sliding_window=None,
        kv_quant=False,
        use_triton_attn=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.kv_quant = kv_quant
        self.use_triton_attn = use_triton_attn
        self.mid_buffer = None      # shared flash-decoding buffer, assigned by ModelRunner
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if self.kv_quant:
                store_kvcache_int8(k, v, k_cache, v_cache, self.k_scale, self.v_scale, context.slot_mapping)
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # window_size: (left, right). sliding_window sets left bound; None means global.
        window_size = (self.sliding_window, 0) if self.sliding_window else (-1, -1)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # Triton FA2 covers dense prefill and paged prefill (prefix-cache
            # hit) for both BF16 and INT8 caches. Only sliding window still
            # falls back to flash_attn.
            use_triton = self.use_triton_attn and self.sliding_window is None
            if use_triton:
                if context.block_tables is None:
                    o = triton_flash_attn_varlen(q, k, v, context.cu_seqlens_q,
                                                 max_seqlen_q=context.max_seqlen_q, scale=self.scale)
                else:
                    o = triton_flash_attn_varlen(
                        q, k, v, context.cu_seqlens_q,
                        max_seqlen_q=context.max_seqlen_q, scale=self.scale,
                        cu_seqlens_k=context.cu_seqlens_k, block_tables=context.block_tables,
                        k_scale=self.k_scale if self.kv_quant else None,
                        v_scale=self.v_scale if self.kv_quant else None)
            else:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables,
                                           window_size=window_size)
        else:    # decode
            if self.kv_quant:
                if self.use_triton_attn and self.sliding_window is None:
                    # Stage 3: fused INT8 dequant — cache read once as int8,
                    # dequantized in-register, no whole-cache BF16 buffer.
                    o = triton_paged_attention(
                        q, k_cache, v_cache,
                        context.block_tables, context.context_lens, self.scale,
                        mid=self.mid_buffer, k_scale=self.k_scale, v_scale=self.v_scale)
                else:
                    # Default: whole-cache dequant + flash_attn (fallback baseline).
                    k_bf16 = dequant_kvcache(k_cache, self.k_scale)
                    v_bf16 = dequant_kvcache(v_cache, self.v_scale)
                    o = flash_attn_with_kvcache(q.unsqueeze(1), k_bf16, v_bf16,
                                                cache_seqlens=context.context_lens, block_table=context.block_tables,
                                                softmax_scale=self.scale, causal=True, window_size=window_size)
            elif self.use_triton_attn and self.sliding_window is None:
                # Triton paged attention (stage 2): single query over paged KV cache, BF16.
                o = triton_paged_attention(q, k_cache, v_cache,
                                           context.block_tables, context.context_lens, self.scale,
                                           mid=self.mid_buffer)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True, window_size=window_size)
        return o
