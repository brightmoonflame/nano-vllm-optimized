"""EAGLE3 draft-head proposer for speculative decoding.

Replaces the old independent-HF-draft-LM proposer with `Eagle3DraftModel`
(nanovllm/models/eagle3_draft.py) — a 1-layer head that shares the target
model's embedding and consumes the target's aux hidden states instead of
re-running a full draft LM over the whole context every round.

State model
-----------
Per-sequence draft state (`_draft_ctx_len` / `_aux` / `_draft0`, keyed by
`seq.seq_id`) is committed **only** by `extend`, using the target model's
true aux hidden states. `propose` is read-only: it self-chains K steps
from the committed state and never commits its own speculative KV to
`_draft_ctx_len`/`_aux`/`_draft0`.

Why propose() never commits its own speculative KV: the K-1 chain steps
use the draft's own self-predicted hidden states, which are a guess. Once
the target verifies the round, it has computed the *true* hidden states
for every accepted position — so the next round's `extend` rebuilds
committed state from that true data instead of trusting the draft's own
guess.

Paged KV cache (SPEC_DECODE_PLAN.md 4a)
----------------------------------------
The draft's KV no longer lives in a per-seq dense python dict. It's
stored in a dedicated `draft_kv_cache` tensor (assigned onto
`self.draft.midlayer.self_attn.attn.k_cache/v_cache` by
`ModelRunner.allocate_kv_cache`) that is addressed by the *same* block
ids as the target's cache (`seq.block_table`) — see
`nanovllm/utils/paging.py`. This is safe because:

  - `BlockManager` only ever hands out/tracks block ids; it has no idea
    what physically lives in a block, so target and draft can each keep
    their own tensor at the same block id with no conflict.
  - Speculative writes during `propose()`'s self-chain are never "read
    back" incorrectly: the next round's `extend` call always starts from
    `positions[0] == self._draft_ctx_len[seq_id]` (the last *committed*
    row) and overwrites forward from there, and `propose()`'s
    `context_lens` never exceeds the true committed length + chain step
    count, so a rejected tail's stale KV is simply never read.
  - Preemption invalidates both caches identically: `BlockManager.
    deallocate` frees the same block ids that back both tensors, so once
    those ids are reused by a different sequence, the old draft KV in
    those slots is naturally overwritten before it could ever be read.

`extend()` batches every seq's new tokens into one varlen forward
(mirrors `ModelRunner.prepare_prefill`'s "continuing prefix cache"
shape). `propose()`'s K-1 self-chain steps are batched single-token
paged decode steps (mirrors `ModelRunner.prepare_decode`), with
slot_mapping recomputed from the padded block-table tensor on-GPU (no
per-step host sync).

Round lifecycle (owned by the caller, e.g. `ModelRunner.run_spec` — P5):
  1. `extend(...)`  — catch up on tokens confirmed since the last round
     (full prompt on a sequence's first round, else accepted drafts +
     bonus/recovered token from the previous round's verify pass).
  2. `propose(...)` — get K new draft tokens (target-vocab ids).
  3. target verifies the K+1-token chunk; rejection sampling decides the
     accepted tokens for the *next* round's `extend` call.
"""

import torch
from transformers import AutoConfig

from nanovllm.engine.sequence import Sequence
from nanovllm.models.eagle3_draft import Eagle3DraftModel
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_eagle3_weights
from nanovllm.utils.paging import compute_slot_mapping, pad_block_tables


def _aux_layer_ids(num_hidden_layers: int) -> list[int]:
    """SpecForge/EAGLE3 training convention: low/mid/high layer outputs
    (e.g. {1, 13, 24} for Llama-3.2-3B's 28 layers). Must match whatever
    the draft checkpoint was actually trained against."""
    return sorted({1, num_hidden_layers // 2 - 1, num_hidden_layers - 4})


class Proposer:
    """EAGLE3 draft-head proposer (requires target TP == 1, see
    `Eagle3DraftModel`'s docstring)."""

    def __init__(
        self,
        draft_model_path: str,
        target_model,
        block_size: int,
        num_spec_tokens: int = 5,
        aux_layer_ids: list[int] | None = None,
    ):
        self.num_spec_tokens = num_spec_tokens
        self.block_size = block_size
        device = next(target_model.parameters()).device
        dtype = next(target_model.parameters()).dtype

        draft_config = AutoConfig.from_pretrained(draft_model_path)
        self.draft = Eagle3DraftModel(draft_config).to(device=device, dtype=dtype).eval()
        load_eagle3_weights(self.draft, draft_model_path)
        # Must come after load_eagle3_weights: sharing first would make
        # embed_tokens.weight show up in named_parameters() before the
        # completeness check runs, mislabeling it as a missing weight.
        self.draft.embed_tokens = target_model.model.embed_tokens
        self.hot_token_id = self.draft.get_hot_token_id()

        # Which target layer outputs to feed `extend` — the caller (P5)
        # passes this to the target model's forward as `aux_layer_ids`.
        # Caller-supplied ids take priority (must match how the draft
        # checkpoint was actually trained); fall back to the SpecForge/
        # EAGLE3 low/mid/high heuristic when not specified.
        self.aux_layer_ids = (
            sorted(aux_layer_ids) if aux_layer_ids is not None
            else _aux_layer_ids(len(target_model.model.layers))
        )

        # Committed per-seq state, written only by `extend`. The draft KV
        # itself lives in `draft.midlayer.self_attn.attn.{k,v}_cache`
        # (paged, block ids shared with the target) — only its *length*
        # needs bookkeeping here.
        self._draft_ctx_len: dict[int, int] = {}
        self._aux: dict[int, torch.Tensor] = {}
        self._draft0: dict[int, torch.Tensor] = {}

    def drop(self, seq_id: int) -> None:
        """Free a finished sequence's committed draft state.

        Does not touch the KV cache tensor itself — those physical slots
        are reclaimed (and will be overwritten) once `BlockManager`
        reassigns the freed block ids to a different sequence.
        """
        self._draft_ctx_len.pop(seq_id, None)
        self._aux.pop(seq_id, None)
        self._draft0.pop(seq_id, None)

    @torch.inference_mode()
    def extend(
        self,
        seqs: list[Sequence],
        token_ids: list[list[int]],
        positions: list[list[int]],
        target_aux_hidden: list[torch.Tensor],
    ) -> None:
        """Catch the draft up on newly-confirmed real tokens using the
        target model's aux hidden states at those positions. No sampling
        — only updates committed state.

        One batched varlen forward for the whole call (mirrors
        `ModelRunner.prepare_prefill`'s "continuing prefix cache" shape):
        each seq contributes a contiguous chunk of new rows continuing
        from its own committed draft KV length; KV for the chunk is
        written into (and, for seqs with existing committed context,
        read back from) the paged draft cache via `seq.block_table`.

        Args:
            token_ids[i]: length T_i, the tokens newly confirmed for seq i
                (full prompt on the first call for a sequence, else
                accepted drafts + bonus/recovered token).
            positions[i] / target_aux_hidden[i]: EAGLE shifted-token
                pairing (vLLM eagle.py: input_ids shifted by one,
                positions/hidden_states not) — positions[i][j] is the
                position whose target hidden *predicted* token_ids[i][j]
                (= token position - 1), and target_aux_hidden[i][j] is the
                target's aux hidden state at that position.
        """
        device = target_aux_hidden[0].device
        input_ids_flat: list[int] = []
        positions_flat: list[int] = []
        hidden_rows = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping: list[int] = []
        row_ends: list[int] = []

        for seq, ids, pos, aux in zip(seqs, token_ids, positions, target_aux_hidden):
            sid = seq.seq_id
            T = len(ids)
            start = pos[0]
            end = pos[-1] + 1
            assert start == self._draft_ctx_len.get(sid, 0), (
                f"seq {sid}: extend() must continue contiguously from the "
                f"committed draft context (expected start={self._draft_ctx_len.get(sid, 0)}, got {start})"
            )
            input_ids_flat.extend(ids)
            positions_flat.extend(pos)
            hidden_rows.append(aux)
            cu_seqlens_q.append(cu_seqlens_q[-1] + T)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(T, max_seqlen_q)
            max_seqlen_k = max(end, max_seqlen_k)
            slot_mapping.extend(compute_slot_mapping(seq.block_table, start, end, self.block_size))
            row_ends.append(cu_seqlens_q[-1] - 1)

        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # at least one seq continues from committed context
            block_tables = pad_block_tables([seq.block_table for seq in seqs], device=device)

        input_ids = torch.as_tensor(input_ids_flat, dtype=torch.long, device=device)
        positions_t = torch.as_tensor(positions_flat, dtype=torch.long, device=device)
        hidden = torch.cat(hidden_rows, dim=0)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, device=device)

        set_context(True, cu_seqlens_q_t, cu_seqlens_k_t, max_seqlen_q, max_seqlen_k, slot_mapping_t, None, block_tables)
        normed, aux_out = self.draft(input_ids, positions_t, hidden)
        reset_context()

        row_ends_t = torch.tensor(row_ends, dtype=torch.long, device=device)
        draft_id = self.draft.compute_logits(normed[row_ends_t]).argmax(dim=-1)
        hot_ids = self.hot_token_id[draft_id]
        for i, seq in enumerate(seqs):
            sid = seq.seq_id
            self._draft_ctx_len[sid] = positions[i][-1] + 1
            self._aux[sid] = aux_out[row_ends[i]:row_ends[i] + 1]
            self._draft0[sid] = hot_ids[i:i + 1]

    @torch.inference_mode()
    def propose(self, seqs: list[Sequence], num_spec_tokens: int | None = None) -> list[list[int]]:
        """Self-chain K draft tokens per seq from committed state.

        Read-only w.r.t. committed state (`_draft_ctx_len`/`_aux`/
        `_draft0` are never mutated — see module docstring for why), but
        each chain step *does* write speculative KV into the paged draft
        cache at the positions it advances through; that write is safe
        per the module docstring's "paged KV cache" section — it is never
        read back incorrectly because the next round's `extend` and this
        round's own verify pass both bound their reads to the true
        committed/accepted length.

        Requires `extend` to have already been called at least once for
        every seq in `seqs`; committed KV length is always len(seq) - 1
        at that point (the last token has no hidden state to pair with
        yet — see `extend`).

        Each self-chain step is a batched single-token paged decode step
        (mirrors `ModelRunner.prepare_decode`): slot_mapping is derived
        from the padded block-table tensor + current positions entirely
        on-GPU (gather), so there's no host sync inside the K-1 loop.

        Returns:
            [B, K] target-vocab token ids, one row per seq.
        """
        K = num_spec_tokens or self.num_spec_tokens
        device = self.hot_token_id.device

        positions = torch.tensor([self._draft_ctx_len[seq.seq_id] for seq in seqs], dtype=torch.long, device=device)
        token = torch.cat([self._draft0[seq.seq_id] for seq in seqs])                 # [B]
        aux = torch.cat([self._aux[seq.seq_id] for seq in seqs], dim=0)               # [B, H]
        block_tables = pad_block_tables([seq.block_table for seq in seqs], device=device)

        draft_steps = [token]
        for _ in range(K - 1):
            context_lens = (positions + 1).to(torch.int32)
            block_idx = (positions // self.block_size).unsqueeze(1)
            slot_mapping = (
                block_tables.gather(1, block_idx).squeeze(1).to(torch.int64) * self.block_size
                + (positions % self.block_size)
            ).to(torch.int32)

            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
            normed, aux_out = self.draft(token, positions, aux)
            reset_context()

            draft_id = self.draft.compute_logits(normed).argmax(dim=-1)
            token = self.hot_token_id[draft_id]
            aux = aux_out
            positions = positions + 1
            draft_steps.append(token)

        return torch.stack(draft_steps, dim=1).tolist()
