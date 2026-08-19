"""Correctness tests for per-channel INT8 W8A16 GEMM.

Run on an NVIDIA GPU:
    pytest -q tests/test_w8a16.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from nanovllm.layers.w8a16_gemm import (
    quantize_per_channel,
    triton_w8a16_gemm,
    w8a16_linear_reference,
)
from nanovllm.layers import linear as linear_layers


def test_quantize_per_channel_layout_and_error_bound():
    torch.manual_seed(0)
    weight = torch.randn(37, 65, dtype=torch.float32)
    qweight, scale = quantize_per_channel(weight)

    assert qweight.shape == (65, 37)
    assert qweight.dtype == torch.int8
    assert scale.shape == (37,)
    assert scale.dtype == torch.float32
    restored = qweight.t().float() * scale[:, None]
    # Round-to-nearest symmetric INT8 has <= half-scale error per element.
    assert torch.all((restored - weight).abs() <= scale[:, None] * 0.50001)


def _mock_tp(monkeypatch, rank: int, world_size: int):
    monkeypatch.setattr(linear_layers.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(linear_layers.dist, "get_world_size", lambda: world_size)


def _dequantize_module_weight(module):
    return module.weight.t().float() * module.weight_scale[:, None]


def _assert_quantized_close(module, expected):
    restored = _dequantize_module_weight(module)
    # Each value has at most half of its row's quantization step of error.
    assert torch.all((restored - expected).abs() <= module.weight_scale[:, None] * 0.50001)


def test_w8a16_loaders_quantize_after_tp_sharding(monkeypatch):
    """Column, merged-column, and row loaders preserve their old TP slices."""
    _mock_tp(monkeypatch, rank=1, world_size=2)

    column = linear_layers.ColumnParallelLinear(4, 8)
    column.enable_w8a16()
    full_column = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    column.weight_loader(column.weight, full_column)
    _assert_quantized_close(column, full_column[4:8])

    merged = linear_layers.MergedColumnParallelLinear(4, [4, 4])
    merged.enable_w8a16()
    gate = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    up = gate + 100
    merged.weight_loader(merged.weight, gate, 0)
    merged.weight_loader(merged.weight, up, 1)
    expected_merged = torch.cat((gate[2:4], up[2:4]), dim=0)
    _assert_quantized_close(merged, expected_merged)

    qkv = linear_layers.QKVParallelLinear(
        hidden_size=4, head_size=2, total_num_heads=4, total_num_kv_heads=2,
    )
    qkv.enable_w8a16()
    q = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    k = q[:4] + 100
    v = q[:4] + 200
    qkv.weight_loader(qkv.weight, q, "q")
    qkv.weight_loader(qkv.weight, k, "k")
    qkv.weight_loader(qkv.weight, v, "v")
    expected_qkv = torch.cat((q[4:8], k[2:4], v[2:4]), dim=0)
    _assert_quantized_close(qkv, expected_qkv)

    row = linear_layers.RowParallelLinear(8, 4)
    row.enable_w8a16()
    full_row = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    row.weight_loader(row.weight, full_row)
    _assert_quantized_close(row, full_row[:, 4:8])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,n,k", [(1, 128, 96), (17, 256, 128), (65, 512, 256)])
def test_triton_w8a16_matches_fused_reference(dtype, m, n, k):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype)
    qweight, scale = quantize_per_channel(weight)

    ref = w8a16_linear_reference(x, qweight, scale, bias)
    out = triton_w8a16_gemm(x, qweight, scale, bias)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
