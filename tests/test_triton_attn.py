"""Stage 1 precision check: triton_flash_attn_varlen vs flash_attn_varlen_func.

Requires a CUDA GPU + flash-attn installed. Run directly:
    python -u tests/test_triton_attn.py

The `-u` flag makes every print flush immediately, so you can see exactly
which case is running (Triton compiles each kernel on first use, which can
take 1-3 minutes and prints nothing during that time).
"""
import torch


def _log(msg: str):
    print(msg, flush=True)


_log("importing flash_attn ...")
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

_log("importing triton kernel ...")
from nanovllm.layers.triton_attn import (
    triton_flash_attn_varlen,
    triton_paged_attention,
    triton_paged_attention_int8,
)
from nanovllm.layers.kv_quant import dequant_kvcache

_log(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    _log(f"device: {torch.cuda.get_device_name(0)}")


def _run_case(name, num_heads, num_kv_heads, head_dim, seqlens, dtype=torch.bfloat16):
    torch.manual_seed(0)
    device = "cuda"
    lens = torch.tensor(seqlens, dtype=torch.int32)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0).to(torch.int32)]).to(device)
    total_tokens = int(cu_seqlens[-1])
    max_seqlen = max(seqlens)
    scale = head_dim ** -0.5

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)

    _log(f"[{name}] running flash_attn_varlen_func (reference) ...")
    ref = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=scale, causal=True,
    )

    _log(f"[{name}] running triton_flash_attn_varlen (first call triggers JIT compile) ...")
    out = triton_flash_attn_varlen(q, k, v, cu_seqlens, max_seqlen=max_seqlen, scale=scale)

    max_abs_err = (out - ref).abs().max().item()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    _log(f"[{name}] PASSED  max_abs_err={max_abs_err:.4e}")


def test_mha_short():
    _run_case("mha_short(128)", num_heads=8, num_kv_heads=8, head_dim=128, seqlens=[128])


def test_mha_multi_seq_uneven():
    _run_case("mha_multi_seq_uneven(128,300,17)", num_heads=8, num_kv_heads=8, head_dim=128, seqlens=[128, 300, 17])


def test_gqa_2to1():
    _run_case("gqa_2to1(1024)", num_heads=8, num_kv_heads=4, head_dim=128, seqlens=[1024])


def test_gqa_4to1_long():
    _run_case("gqa_4to1_long(4096)", num_heads=16, num_kv_heads=4, head_dim=128, seqlens=[4096])


def _run_decode_case(name, num_heads, num_kv_heads, head_dim, ctx_lens,
                     block_size=256, dtype=torch.bfloat16):
    torch.manual_seed(0)
    device = "cuda"
    num_seqs = len(ctx_lens)
    max_seqlen = max(ctx_lens)
    scale = head_dim ** -0.5

    # Random physical-block assignment: logical block i of a seq maps to a
    # shuffled physical block id. Kernels that assume physical blocks are
    # contiguous (or in logical order) fail here.
    blocks_per_seq = [(l + block_size - 1) // block_size for l in ctx_lens]
    total_logical = sum(blocks_per_seq)
    pool = torch.randperm(total_logical).tolist()    # shuffled physical ids
    max_blocks = max(blocks_per_seq)
    rows, ptr = [], 0
    for nb in blocks_per_seq:
        rows.append(pool[ptr:ptr + nb] + [0] * (max_blocks - nb))   # pad unused entries
        ptr += nb
    block_tables = torch.tensor(rows, dtype=torch.int32, device=device)

    num_physical = total_logical
    k_cache = torch.randn(num_physical, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn(num_physical, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    q = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=device)
    context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device=device)

    _log(f"[{name}] running flash_attn_with_kvcache (reference) ...")
    ref = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context_lens, block_table=block_tables,
        softmax_scale=scale, causal=True,
    )
    _log(f"[{name}] running triton_paged_attention (first call triggers JIT compile) ...")
    out = triton_paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale)

    ref = ref.squeeze(1)   # flash_attn returns (S,1,H,D); ours is (S,H,D)
    max_abs_err = (out - ref).abs().max().item()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    _log(f"[{name}] PASSED  max_abs_err={max_abs_err:.4e}")


def test_decode_mha_partial_blocks():
    # 300 crosses two 256-token blocks (partial second block), MHA.
    _run_decode_case("decode_mha(300)", num_heads=8, num_kv_heads=8, head_dim=128, ctx_lens=[300])


def test_decode_multi_seq_uneven():
    # Mix of block boundaries: 128 (mid-block), 256 (exact), 511 (one short of 2 blocks).
    _run_decode_case("decode_multi(128,256,300,511)", num_heads=8, num_kv_heads=8, head_dim=128,
                     ctx_lens=[128, 256, 300, 511])


def test_decode_gqa_3to1():
    # Llama-3.2-3B config: 24 q heads / 8 kv heads.
    _run_decode_case("decode_gqa_3to1(1024)", num_heads=24, num_kv_heads=8, head_dim=128, ctx_lens=[1024])


def test_decode_gqa_4to1_long():
    _run_decode_case("decode_gqa_4to1(4096)", num_heads=16, num_kv_heads=4, head_dim=128, ctx_lens=[4096])


def _quantize_cache(bf16_cache: torch.Tensor):
    """Per-(token, head) symmetric min-max quantization, mirroring
    store_kvcache_int8_kernel: scale = max|.|/127, int8 = round(x/scale)."""
    sc = bf16_cache.float().abs().amax(dim=-1) / 127.0
    sc = sc.clamp(min=1e-6)
    i8 = torch.round(bf16_cache.float() / sc[..., None]).clamp(-127, 127).to(torch.int8)
    return i8, sc


def _run_decode_int8_case(name, num_heads, num_kv_heads, head_dim, ctx_lens,
                          block_size=256, dtype=torch.bfloat16):
    torch.manual_seed(0)
    device = "cuda"
    num_seqs = len(ctx_lens)
    scale = head_dim ** -0.5

    # Shuffled physical blocks + quantized cache (same recipe as BF16 decode cases).
    blocks_per_seq = [(l + block_size - 1) // block_size for l in ctx_lens]
    total_logical = sum(blocks_per_seq)
    pool = torch.randperm(total_logical).tolist()
    max_blocks = max(blocks_per_seq)
    rows, ptr = [], 0
    for nb in blocks_per_seq:
        rows.append(pool[ptr:ptr + nb] + [0] * (max_blocks - nb))
        ptr += nb
    block_tables = torch.tensor(rows, dtype=torch.int32, device=device)

    k_i8, k_sc = _quantize_cache(torch.randn(total_logical, block_size, num_kv_heads, head_dim, dtype=dtype, device=device))
    v_i8, v_sc = _quantize_cache(torch.randn(total_logical, block_size, num_kv_heads, head_dim, dtype=dtype, device=device))
    q = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=device)
    context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device=device)

    # Reference: the exact default kv_quant=True path in attention.py —
    # whole-cache dequant to BF16, then flash_attn. The fused kernel shares
    # this numeric path (int8 × scale), so outputs should match closely.
    _log(f"[{name}] running dequant + flash_attn_with_kvcache (reference) ...")
    k_bf16 = dequant_kvcache(k_i8, k_sc)
    v_bf16 = dequant_kvcache(v_i8, v_sc)
    ref = flash_attn_with_kvcache(
        q.unsqueeze(1), k_bf16, v_bf16,
        cache_seqlens=context_lens, block_table=block_tables,
        softmax_scale=scale, causal=True,
    ).squeeze(1)

    _log(f"[{name}] running triton_paged_attention_int8 (first call triggers JIT compile) ...")
    out = triton_paged_attention_int8(q, k_i8, v_i8, k_sc, v_sc, block_tables, context_lens, scale)

    max_abs_err = (out - ref).abs().max().item()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    _log(f"[{name}] PASSED  max_abs_err={max_abs_err:.4e}")


def test_decode_int8_partial_blocks():
    _run_decode_int8_case("decode_int8_mha(300)", num_heads=8, num_kv_heads=8, head_dim=128, ctx_lens=[300])


def test_decode_int8_multi_seq_uneven():
    _run_decode_int8_case("decode_int8_multi(128,256,511)", num_heads=8, num_kv_heads=8, head_dim=128,
                          ctx_lens=[128, 256, 511])


def test_decode_int8_gqa_3to1():
    _run_decode_int8_case("decode_int8_gqa_3to1(2048)", num_heads=24, num_kv_heads=8, head_dim=128, ctx_lens=[2048])


def test_decode_int8_gqa_4to1_long():
    _run_decode_int8_case("decode_int8_gqa_4to1(4096)", num_heads=16, num_kv_heads=4, head_dim=128, ctx_lens=[4096])


def _run_splitk_consistency(name, num_heads, num_kv_heads, head_dim, ctx_len,
                            block_size=256, splits: int = 64):
    """Flash-decoding correctness: multi-split output must equal single-split.

    The KV partition + reduce is mathematically equivalent to one sequential
    pass, so outputs must match to float-rounding regardless of split count.
    Empty splits (split range beyond ctx_len) contribute zero — this also
    exercises that path since splits=64 > ceil(ctx_len/BLOCK_N) tiles.
    """
    torch.manual_seed(0)
    device = "cuda"
    scale = head_dim ** -0.5
    num_seqs = 1

    block_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
    k_cache = torch.randn(1, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(1, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    q = torch.randn(num_seqs, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    context_lens = torch.tensor([ctx_len], dtype=torch.int32, device=device)

    single = triton_paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale, num_splits=1)
    multi = triton_paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale, num_splits=splits)

    max_abs_err = (multi - single).abs().max().item()
    torch.testing.assert_close(multi, single, atol=1e-2, rtol=1e-2)
    _log(f"[{name}] PASSED  splits=1 vs {splits}  max_abs_err={max_abs_err:.4e}")


def test_decode_splitk_consistency():
    # 300 tokens / BLOCK_N=64 = 5 tiles vs 64 splits → 59 empty splits exercised.
    _run_splitk_consistency("splitk_bf16(300,64splits)", num_heads=24, num_kv_heads=8, head_dim=128, ctx_len=300)


def test_decode_splitk_consistency_int8():
    torch.manual_seed(0)
    device = "cuda"
    num_heads, num_kv_heads, head_dim, ctx_len = 24, 8, 128, 300
    scale = head_dim ** -0.5

    block_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
    k_bf16 = torch.randn(1, 256, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(1, 256, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_i8, k_sc = _quantize_cache(k_bf16)
    v_i8, v_sc = _quantize_cache(v_bf16)
    q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    context_lens = torch.tensor([ctx_len], dtype=torch.int32, device=device)

    single = triton_paged_attention_int8(q, k_i8, v_i8, k_sc, v_sc, block_tables, context_lens, scale, num_splits=1)
    multi = triton_paged_attention_int8(q, k_i8, v_i8, k_sc, v_sc, block_tables, context_lens, scale, num_splits=64)

    diff = (multi - single).abs()
    nan_mask = torch.isnan(diff)
    if nan_mask.any():
        bad_heads = torch.nonzero(nan_mask[0].any(dim=-1)).flatten().tolist()
        _log(f"[splitk_int8] NaN in heads {bad_heads} / {num_heads}")
    max_abs_err = diff.max().item()
    torch.testing.assert_close(multi, single, atol=1e-2, rtol=1e-2)
    _log(f"[splitk_int8(300,64splits)] PASSED  splits=1 vs 64  max_abs_err={max_abs_err:.4e}")


if __name__ == "__main__":
    test_mha_short()
    test_mha_multi_seq_uneven()
    test_gqa_2to1()
    test_gqa_4to1_long()
    test_decode_mha_partial_blocks()
    test_decode_multi_seq_uneven()
    test_decode_gqa_3to1()
    test_decode_gqa_4to1_long()
    test_decode_int8_partial_blocks()
    test_decode_int8_multi_seq_uneven()
    test_decode_int8_gqa_3to1()
    test_decode_int8_gqa_4to1_long()
    test_decode_splitk_consistency()
    test_decode_splitk_consistency_int8()
    print("All precision tests passed (prefill FA2 + decode paged + INT8 fused + flash-decoding).", flush=True)
