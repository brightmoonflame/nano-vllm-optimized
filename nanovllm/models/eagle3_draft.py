import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaConfig

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import get_rope


class Eagle3Attention(nn.Module):
    """Single attention layer whose QKV projections consume
    cat(input_layernorm(embed), hidden_norm(hidden)) = 2 * hidden_size.

    Paged attention: identical structure to `models/llama.py`'s
    `LlamaAttention` — a `layers.attention.Attention` submodule reads/writes
    KV through the global `context` (see `utils/context.py`), so this class
    never touches past_key_values/cache_seqlens explicitly. The draft's KV
    cache tensor (assigned onto `self.attn.k_cache/v_cache` by
    `ModelRunner.allocate_kv_cache`) is a *separate* tensor from the
    target's, but indexed by the *same* block ids (`seq.block_table`) — see
    SPEC_DECODE_PLAN.md 4a.
    """

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
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,      # [N] absolute positions (flat, N = total tokens in batch)
        hidden_states: torch.Tensor,  # [N, 2H]
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


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
        embeds: torch.Tensor,              # [N, H]
        hidden_states: torch.Tensor,       # [N, H] (fc output or own previous aux)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        embeds = self.input_layernorm(embeds)
        x = torch.cat([embeds, hidden_states], dim=-1)
        attn_out = self.self_attn(positions, x)
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Eagle3DraftModel(nn.Module):
    """EAGLE3 draft head for a Llama target model.

    Matches the thoughtworks/Llama-3.2-3B-Instruct-Eagle3 checkpoint layout:
    fc (3H->H), midlayer.*, norm, lm_head (draft vocab), d2t (id diffs).
    embed_tokens is NOT in the checkpoint — share the target model's
    (assign ``draft.embed_tokens = target.model.embed_tokens``).
    Note: with TP > 1 the shared embedding's forward contains collectives,
    so this rank-0-only draft currently requires TP == 1.

    All batching/paging is expressed via the flat [N, ...] convention (N =
    total tokens across the batch) plus the global `context` — same style
    as `models/llama.py` — rather than an explicit [B, T] + past_key_values
    interface. This lets the caller (`spec_decode/proposer.py`) run a
    genuine varlen batched forward for `extend()` and a paged single-token
    decode step for each `propose()` self-chain step.
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
        input_ids: torch.Tensor,           # [N] target-vocab token ids
        positions: torch.Tensor,           # [N] absolute positions
        hidden_states: torch.Tensor,       # [N, 3H] first use of a round / [N, H] later steps
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (normed_hidden [N,H], aux_hidden [N,H]).
        normed_hidden feeds compute_logits; aux_hidden (pre-norm residual) is
        the hidden input for the next draft step."""
        embeds = self.embed_tokens(input_ids)
        if hidden_states.size(-1) != embeds.size(-1):
            hidden_states = self.fc(hidden_states)
        hidden_states, residual = self.midlayer(positions, embeds, hidden_states)
        normed, aux = self.norm(hidden_states, residual)
        return normed, aux

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
