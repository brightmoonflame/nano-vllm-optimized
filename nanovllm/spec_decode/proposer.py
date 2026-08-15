"""EAGLE3 draft-head proposer for speculative decoding.

Replaces the old independent-HF-draft-LM proposer with `Eagle3DraftModel`
(nanovllm/models/eagle3_draft.py) — a 1-layer head that shares the target
model's embedding and consumes the target's aux hidden states instead of
re-running a full draft LM over the whole context every round.

State model
-----------
Per-sequence draft state (`_kv` / `_aux` / `_draft0`, keyed by `seq.seq_id`)
is committed **only** by `extend`, using the target model's true aux
hidden states. `propose` is read-only: it self-chains K steps from the
committed state in a local (discarded) KV copy and never writes back.

Why propose() never commits its own speculative KV: the K-1 chain steps
use the draft's own self-predicted hidden states, which are a guess. Once
the target verifies the round, it has computed the *true* hidden states
for every accepted position — so the next round's `extend` rebuilds
committed state from that true data instead of trusting the draft's own
guess. This also removes the need for a separate "truncate rejected
tail" step: the whole round's speculative KV is simply never committed.

Round lifecycle (owned by the caller, e.g. `ModelRunner.run_spec` — P5):
  1. `extend(...)`  — catch up on tokens confirmed since the last round
     (full prompt on a sequence's first round, else accepted drafts +
     bonus/recovered token from the previous round's verify pass).
  2. `propose(...)` — get K new draft tokens (target-vocab ids).
  3. target verifies the K+1-token chunk; rejection sampling decides the
     accepted tokens for the *next* round's `extend` call.

NOTE: this proposer's public interface (constructor takes `target_model`;
`propose`/`extend` signatures) is intentionally incompatible with the
current `model_runner.py` call sites — those are updated in the next
phase (P5) once `run_spec` is wired to pass target aux hidden states.
"""

import torch
from transformers import AutoConfig

from nanovllm.engine.sequence import Sequence
from nanovllm.models.eagle3_draft import Eagle3DraftModel
from nanovllm.utils.loader import load_eagle3_weights


def _aux_layer_ids(num_hidden_layers: int) -> list[int]:
    """SpecForge/EAGLE3 training convention: low/mid/high layer outputs
    (e.g. {1, 13, 24} for Llama-3.2-3B's 28 layers). Must match whatever
    the draft checkpoint was actually trained against."""
    return sorted({1, num_hidden_layers // 2 - 1, num_hidden_layers - 4})


class Proposer:
    """EAGLE3 draft-head proposer (requires target TP == 1, see
    `Eagle3DraftModel`'s docstring)."""

    def __init__(self, draft_model_path: str, target_model, num_spec_tokens: int = 5):
        self.num_spec_tokens = num_spec_tokens
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
        self.aux_layer_ids = _aux_layer_ids(len(target_model.model.layers))

        # Committed per-seq state, written only by `extend`.
        self._kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._aux: dict[int, torch.Tensor] = {}
        self._draft0: dict[int, torch.Tensor] = {}

    def drop(self, seq_id: int) -> None:
        """Free a finished sequence's committed draft state."""
        self._kv.pop(seq_id, None)
        self._aux.pop(seq_id, None)
        self._draft0.pop(seq_id, None)

    @torch.inference_mode()
    def extend(
        self,
        seq_ids: list[int],
        token_ids: list[list[int]],
        positions: list[list[int]],
        target_aux_hidden: list[torch.Tensor],
    ) -> None:
        """Catch the draft up on newly-confirmed real tokens using the
        target model's aux hidden states at those positions. No sampling
        — only updates committed state.

        Per-seq token counts vary (acceptance count differs every round),
        so this loops per seq rather than batching.

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
        for sid, ids, pos, aux in zip(seq_ids, token_ids, positions, target_aux_hidden):
            self._extend_one(sid, ids, pos, aux)

    def _extend_one(
        self,
        seq_id: int,
        token_ids: list[int],
        positions: list[int],
        target_aux_hidden: torch.Tensor,
    ) -> None:
        device = target_aux_hidden.device
        T = len(token_ids)
        input_ids = torch.as_tensor(token_ids, dtype=torch.long, device=device).view(1, T)
        pos = torch.as_tensor(positions, dtype=torch.long, device=device).view(1, T)
        hidden = target_aux_hidden.view(1, T, -1)

        past = self._kv.get(seq_id)
        cache_seqlens = None
        if past is not None:
            k, v = past
            past = (k.unsqueeze(0), v.unsqueeze(0))
            cache_seqlens = torch.tensor([k.size(0)], dtype=torch.int32, device=device)

        normed, aux, (new_k, new_v) = self.draft(input_ids, pos, hidden, past, cache_seqlens)
        self._kv[seq_id] = (new_k.squeeze(0), new_v.squeeze(0))
        self._aux[seq_id] = aux[:, -1, :]
        # The last processed position's output predicts the very next
        # token — i.e. this round's first draft token, for free.
        draft_id = self.draft.compute_logits(normed[:, -1, :]).argmax(dim=-1)
        self._draft0[seq_id] = self.hot_token_id[draft_id]

    @torch.inference_mode()
    def propose(self, seqs: list[Sequence], num_spec_tokens: int | None = None) -> list[list[int]]:
        """Self-chain K draft tokens per seq from committed state.

        Read-only: never mutates `_kv`/`_aux`/`_draft0` (see module
        docstring for why). Requires `extend` to have already been called
        at least once for every seq in `seqs`; committed KV length is
        always len(seq) - 1 at that point (the last token has no hidden
        state to pair with yet — see `extend`).

        Returns:
            [B, K] target-vocab token ids, one row per seq.
        """
        K = num_spec_tokens or self.num_spec_tokens
        seq_ids = [seq.seq_id for seq in seqs]
        device = self.hot_token_id.device
        B = len(seq_ids)

        k_list = [self._kv[sid][0] for sid in seq_ids]
        v_list = [self._kv[sid][1] for sid in seq_ids]
        lens = [k.size(0) for k in k_list]
        L = max(lens)
        n_kv, D = k_list[0].shape[1:]
        k_batch = torch.zeros(B, L, n_kv, D, dtype=k_list[0].dtype, device=device)
        v_batch = torch.zeros(B, L, n_kv, D, dtype=v_list[0].dtype, device=device)
        for i, (k, v) in enumerate(zip(k_list, v_list)):
            k_batch[i, :k.size(0)] = k
            v_batch[i, :v.size(0)] = v
        cache_seqlens = torch.tensor(lens, dtype=torch.int32, device=device)
        past = (k_batch, v_batch)

        token = torch.cat([self._draft0[sid] for sid in seq_ids])                  # [B]
        aux = torch.cat([self._aux[sid] for sid in seq_ids], dim=0).unsqueeze(1)    # [B, 1, H]
        # draft0's row position = committed KV length (`lens`): EAGLE's
        # shifted-token convention puts token t_{p+1} at row position p,
        # so the next token goes at the row right after the last committed
        # one. `lens` == len(seq) - 1 for every seq here (extend pairs the
        # sampled/accepted tokens with the hidden states that predicted
        # them), so this matches prepare_spec_decode's window placement.
        positions = torch.tensor(lens, dtype=torch.long, device=device)

        draft_steps = [token]
        for _ in range(K - 1):
            normed, aux_out, past = self.draft(
                token.view(B, 1), positions.view(B, 1), aux, past, cache_seqlens)
            cache_seqlens = cache_seqlens + 1
            positions = positions + 1
            draft_id = self.draft.compute_logits(normed[:, -1, :]).argmax(dim=-1)
            token = self.hot_token_id[draft_id]
            aux = aux_out[:, -1:, :]
            draft_steps.append(token)

        return torch.stack(draft_steps, dim=1).tolist()
