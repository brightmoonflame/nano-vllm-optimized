import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaConfig

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import get_rope


class Eagle3Attention(nn.Module):
    """Single attention layer whose QKV projections consume
    cat(input_layernorm(embed), hidden_norm(hidden)) = 2 * hidden_size."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        input_size = 2 * config.hidden_size
        self.q_proj = nn.Linear(input_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(input_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(input_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000),
        )

    def forward(
        self,
        positions: torch.Tensor,           # [B, T] absolute positions
        hidden_states: torch.Tensor,       # [B, T, 2H]
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None,  # [B, L, n_kv, D], right-padded
        cache_seqlens: torch.Tensor,       # [B] valid past lengths (<= L)
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions.flatten(), q.view(B * T, self.num_heads, self.head_dim),
                               k.view(B * T, self.num_kv_heads, self.head_dim))
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)

        L = 0
        if past_key_values is not None:
            past_k, past_v = past_key_values
            L = past_k.size(1)
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        new_past = (k, v)
        S = k.size(1)

        n_rep = self.num_heads // self.num_kv_heads
        k_exp = k.repeat_interleave(n_rep, dim=2).transpose(1, 2)   # [B, n_heads, S, D]
        v_exp = v.repeat_interleave(n_rep, dim=2).transpose(1, 2)

        # Past columns [0, L): valid where j < cache_seqlens[b] (right-padded).
        # New columns [L, L+T): causal within the chunk (j - L <= t).
        j = torch.arange(S, device=hidden_states.device)
        t = torch.arange(T, device=hidden_states.device)
        allowed = (j[None, None, :] < cache_seqlens[:, None, None]) | (
            (j[None, None, :] >= L) & (j[None, None, :] < L + t[None, :, None] + 1))
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k_exp, v_exp, attn_mask=allowed[:, None])
        o = o.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(o), new_past


class Eagle3MLP(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DecoderLayer(nn.Module):
    """The EAGLE3 input layer: norms embed/hidden separately, concats them (2H),
    then runs a standard attention+MLP block with deferred residuals."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.self_attn = Eagle3Attention(config)
        self.mlp = Eagle3MLP(config)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,              # [B, T, H]
        hidden_states: torch.Tensor,       # [B, T, H] (fc output or own previous aux)
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None,
        cache_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        embeds = self.input_layernorm(embeds)
        x = torch.cat([embeds, hidden_states], dim=-1)
        attn_out, new_past = self.self_attn(positions, x, past_key_values, cache_seqlens)
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, new_past


class Eagle3DraftModel(nn.Module):
    """EAGLE3 draft head for a Llama target model.

    Matches the thoughtworks/Llama-3.2-3B-Instruct-Eagle3 checkpoint layout:
    fc (3H->H), midlayer.*, norm, lm_head (draft vocab), d2t (id diffs).
    embed_tokens is NOT in the checkpoint — share the target model's
    (assign ``draft.embed_tokens = target.model.embed_tokens``).
    Note: with TP > 1 the shared embedding's forward contains collectives,
    so this rank-0-only draft currently requires TP == 1.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.draft_vocab_size = config.draft_vocab_size
        self.fc = nn.Linear(3 * config.hidden_size, config.hidden_size, bias=False)
        self.midlayer = Eagle3DecoderLayer(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        self.embed_tokens = None    # shared from the target model after init
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.int32))

    def get_hot_token_id(self) -> torch.Tensor:
        # d2t stores (target_id - draft_id) differences.
        return self.d2t.long() + torch.arange(self.draft_vocab_size, dtype=torch.long, device=self.d2t.device)

    def forward(
        self,
        input_ids: torch.Tensor,           # [B, T] target-vocab token ids
        positions: torch.Tensor,           # [B, T] absolute positions
        hidden_states: torch.Tensor,       # [B, T, 3H] first use of a round / [B, T, H] later steps
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Returns (normed_hidden [B,T,H], aux_hidden [B,T,H], new_past).
        normed_hidden feeds compute_logits; aux_hidden (pre-norm residual) is
        the hidden input for the next draft step."""
        B = input_ids.size(0)
        if cache_seqlens is None:
            assert past_key_values is None
            cache_seqlens = torch.zeros(B, dtype=torch.int32, device=input_ids.device)
        embeds = self.embed_tokens(input_ids)
        if hidden_states.size(-1) != embeds.size(-1):
            hidden_states = self.fc(hidden_states)
        hidden_states, residual, new_past = self.midlayer(
            positions, embeds, hidden_states, past_key_values, cache_seqlens)
        normed, aux = self.norm(hidden_states, residual)
        return normed, aux, new_past

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
