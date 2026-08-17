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
from nanovllm.layers.triton_attn import triton_flash_attn_varlen, triton_paged_attention

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


if __name__ == "__main__":
    test_mha_short()
    test_mha_multi_seq_uneven()
    test_gqa_2to1()
    test_gqa_4to1_long()
    test_decode_mha_partial_blocks()
    test_decode_multi_seq_uneven()
    test_decode_gqa_3to1()
    test_decode_gqa_4to1_long()
    print("All precision tests passed (prefill FA2 + decode paged attention).", flush=True)
