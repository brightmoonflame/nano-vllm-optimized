import torch
from torch import nn
import torch.distributed as dist
from transformers import Gemma3Config

from nanovllm.layers.activation import GeluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Gemma3RMSNorm(nn.Module):
    """Gemma 3 uses (1 + weight) scaling; weights are zero-initialized (identity at start)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    @torch.compile(dynamic=True)
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            x = x + residual
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(1.0 + self.weight)
        return x


class Gemma3Attention(nn.Module):

    def __init__(
        self,
        config: Gemma3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, "head_dim", hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Gemma 3 uses query_pre_attn_scalar for scaling (not head_dim ** -0.5).
        qk_scale = getattr(config, "query_pre_attn_scalar", self.head_dim ** 2) ** -0.5

        # Sliding window: 5 local + 1 global interleaved.
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            is_sliding = layer_types[layer_idx] == "sliding_attention"
        else:
            pattern = getattr(config, "_sliding_window_pattern", 6)
            is_sliding = (layer_idx + 1) % pattern != 0
        sliding_window = config.sliding_window if is_sliding else None

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 1000000),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            qk_scale,
            self.num_kv_heads,
            sliding_window=sliding_window,
        )
        # QK-Norm (same as Qwen3, but Gemma 3 always has it regardless of bias).
        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Gemma3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        if hidden_activation == "gelu_pytorch_tanh":
            self.act_fn = GeluAndMul()
        else:
            assert hidden_activation == "silu"
            from nanovllm.layers.activation import SiluAndMul
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Gemma3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp = Gemma3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=getattr(config, "hidden_activation", "gelu_pytorch_tanh"),
        )
        # Gemma 3 has 4 norms: pre/post for both attention and MLP.
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gemma 3 uses local residuals: each block saves its own input, adds it back after post-norm.
        # This differs from Qwen3/Llama which use a rolling residual passed across norm calls.
        # The incoming residual (from the previous layer's MLP output) is already fully resolved,
        # so we ignore it and manage residuals locally.

        # Attention block: norm → attn → post-norm → add residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block: norm → mlp → post-norm → add residual
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, None


class Gemma3Model(nn.Module):

    def __init__(
        self,
        config: Gemma3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # Gemma3DecoderLayer returns residual=None (local residuals already resolved).
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Gemma3Config,
    ) -> None:
        super().__init__()
        self.model = Gemma3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
