import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.w8a16_gemm import quantize_per_channel, triton_w8a16_gemm


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # These are local dimensions after tensor parallel sharding.  Do not
        # derive them from ``weight.shape``: W8A16 stores its weight transposed.
        self.input_size = input_size
        self.output_size = output_size
        self.weight_quant = None
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def enable_w8a16(self):
        """Switch this module to load-time per-channel INT8 quantization.

        The parameter deliberately keeps the checkpoint-visible name ``weight``
        so ``load_model`` keeps its simple name-based dispatch.  Its storage
        changes from logical [N, K] BF16 to GEMM-friendly [K, N] INT8.
        """
        if self.weight_quant is not None:
            raise RuntimeError(f"Linear layer already configured for {self.weight_quant}")
        self.weight_quant = "int8_w8a16"
        self.weight = nn.Parameter(
            torch.empty(self.input_size, self.output_size, dtype=torch.int8),
            requires_grad=False,
        )
        self.weight.weight_loader = self.weight_loader
        self.register_buffer("weight_scale", torch.empty(self.output_size, dtype=torch.float32))

    def _store_loaded(self, param: nn.Parameter, loaded_weight: torch.Tensor, output_offset: int = 0):
        """Copy a locally sharded checkpoint tensor into weight/bias storage."""
        if param.ndim == 1:
            # Biases are never quantized. Packed QKV/gate-up tensors use a
            # slice of the same 1-D bias parameter.
            param.data.narrow(0, output_offset, loaded_weight.numel()).copy_(loaded_weight)
            return
        if self.weight_quant is None:
            param.data.narrow(0, output_offset, loaded_weight.size(0)).copy_(loaded_weight)
            return

        qweight, scale = quantize_per_channel(loaded_weight)
        self.weight.data.narrow(1, output_offset, qweight.size(1)).copy_(qweight)
        self.weight_scale.narrow(0, output_offset, scale.numel()).copy_(scale)

    def _forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if self.weight_quant is None:
            return F.linear(x, self.weight, bias)
        return triton_w8a16_gemm(x, self.weight, self.weight_scale, bias)


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self._store_loaded(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # Output (N) is sharded for ColumnParallel.  Use logical dimensions,
        # because W8A16 stores its parameter as [K, N].
        start_idx = self.tp_rank * self.output_size
        loaded_weight = loaded_weight.narrow(0, start_idx, self.output_size)
        self._store_loaded(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        self._store_loaded(param, loaded_weight, shard_offset)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        self._store_loaded(param, loaded_weight, shard_offset)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if param.ndim == 1:
            self._store_loaded(param, loaded_weight)
            return
        # Input (K) is sharded for RowParallel, while output remains local.
        start_idx = self.tp_rank * self.input_size
        loaded_weight = loaded_weight.narrow(1, start_idx, self.input_size)
        self._store_loaded(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._forward(x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


def enable_w8a16(model: nn.Module):
    """Configure every Transformer LinearBase in ``model`` for W8A16.

    Embeddings and ParallelLMHead intentionally are not LinearBase subclasses,
    so they retain BF16 weights in the first implementation.
    """
    for module in model.modules():
        if isinstance(module, LinearBase):
            module.enable_w8a16()
