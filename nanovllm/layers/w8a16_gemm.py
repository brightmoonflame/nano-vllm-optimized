"""INT8 W8A16 linear layers with per-output-channel PTQ.

The checkpoint weights are quantized once while loading:

    W[N, K] -> Q[K, N] (int8) + S[N] (fp32)

``Q`` is stored transposed because the GEMM consumes it as its right-hand
operand.  Per-output-channel scales can be applied after the dot product:

    X @ (W.T) = (X @ Q) * S

This avoids allocating a dequantized BF16 weight tensor during inference.
"""

import torch
import triton
import triton.language as tl


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a logical Linear weight ``[N, K]`` symmetrically to INT8.

    The returned qweight is laid out as ``[K, N]`` for GEMM.  Scales stay
    FP32: their O(N) footprint is negligible next to O(NK) INT8 weights and
    retaining their precision makes the PTQ error attributable only to INT8.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D Linear weight, got shape {tuple(weight.shape)}")
    w = weight.float()
    scale = w.abs().amax(dim=1).div_(127.0).clamp_(min=1e-8)
    qweight = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return qweight.t().contiguous(), scale


def w8a16_linear_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP32 reference for the same fused operation as the Triton kernel.

    Do not materialize ``qweight * scale`` in BF16/FP16 here: that introduces
    one extra rounding step per weight and is therefore not numerically the
    same operation as applying the FP32 scale after the dot product.
    """
    input_shape = x.shape
    x_2d = x.reshape(-1, input_shape[-1])
    out = x_2d.float() @ qweight.float()
    out.mul_(scale.float())
    if bias is not None:
        out.add_(bias.float())
    return out.to(dtype=x.dtype).reshape(*input_shape[:-1], qweight.shape[1])


@triton.jit
def _w8a16_gemm_kernel(
    x_ptr, qweight_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qk, stride_qn,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute ``out = (x @ qweight) * scale + bias``.

    qweight is INT8 [K, N].  Scale is per output channel, so applying it
    after the FP32 accumulation is exactly equivalent to dequantizing every
    weight before the dot product.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_block * BLOCK_K + offs_k
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0,
        )
        q = tl.load(
            qweight_ptr + k[:, None] * stride_qk + offs_n[None, :] * stride_qn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0,
        )
        if INPUT_IS_BF16:
            q = q.to(tl.bfloat16)
        else:
            q = q.to(tl.float16)
        # ``acc`` is FP32, matching the project's existing Triton attention
        # accumulation style and remaining compatible with Triton 3.0.
        acc += tl.dot(a, q)

    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc *= scale[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_w8a16_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """W8A16 Linear forward for BF16/FP16 activations and INT8 weights."""
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"W8A16 expects BF16 or FP16 activations, got {x.dtype}")
    if qweight.dtype != torch.int8 or scale.dtype != torch.float32:
        raise TypeError("W8A16 expects INT8 qweight and FP32 per-channel scale")
    if qweight.ndim != 2 or scale.ndim != 1:
        raise ValueError("expected qweight[K, N] and scale[N]")

    input_shape = x.shape
    k = input_shape[-1]
    if qweight.shape[0] != k or qweight.shape[1] != scale.numel():
        raise ValueError(
            f"shape mismatch: x[-1]={k}, qweight={tuple(qweight.shape)}, scale={tuple(scale.shape)}"
        )
    x_2d = x.reshape(-1, k).contiguous()
    m, n = x_2d.shape[0], qweight.shape[1]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    # Decode is usually small-M and bandwidth bound; prefill benefits from a
    # wider M tile.  Keeping this deterministic (rather than autotuning on the
    # first execution) makes it safe to pre-warm before CUDA graph capture.
    block_m = 16 if m <= 16 else 32 if m <= 128 else 64
    block_n = 64 if n <= 1024 else 128
    block_k = 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    # Triton's runtime arguments must be concrete pointers even when the
    # compile-time HAS_BIAS branch is false. qweight is a harmless dummy.
    bias_ptr = bias if bias is not None else qweight
    _w8a16_gemm_kernel[grid](
        x_2d, qweight, scale, bias_ptr, out,
        m, n, k,
        x_2d.stride(0), x_2d.stride(1),
        qweight.stride(0), qweight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        INPUT_IS_BF16=x.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4 if block_n <= 64 else 8,
        num_stages=3,
    )
    return out.reshape(*input_shape[:-1], n)
