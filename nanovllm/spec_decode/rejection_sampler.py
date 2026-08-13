"""Rejection sampler for speculative decoding.

Implements the accept/reject logic and residual sampling that guarantees
the output distribution is identical to sampling directly from the target model.
"""

import torch

from nanovllm.spec_decode.metadata import SpecDecodeMetadata


class RejectionSampler:
    """Accept/reject draft tokens using target model probabilities.

    Supports both greedy (argmax comparison) and probabilistic (ratio-based)
    modes. Greedy mode is fully deterministic — every output token equals
    target argmax, so output is identical to non-speculative greedy decoding.
    """

    def __init__(self):
        pass

    @torch.inference_mode()
    def __call__(
        self,
        spec_decode_metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
    ) -> list[list[int]]:
        """Perform rejection sampling on draft tokens.

        Args:
            spec_decode_metadata: Flattened indices for batch verification.
            draft_probs: [total_drafts, V] draft model probabilities.
                None for greedy mode (argmax comparison only).
            target_logits: [total_drafts, V] target model logits at draft
                verification positions.
            bonus_logits: [num_reqs, V] logits at bonus positions.

        Returns:
            List of accepted token IDs per request. Each list contains
            accepted draft tokens + 1 recovered/bonus token.
            Length = num_accepted + 1 (min 1, max num_drafts + 1).
        """
        num_reqs = len(spec_decode_metadata.num_draft_tokens)
        draft_token_ids = spec_decode_metadata.draft_token_ids
        bonus_token_ids = bonus_logits.argmax(dim=-1)

        if draft_probs is None:
            return self._greedy(spec_decode_metadata, target_logits, bonus_token_ids)
        return self._probabilistic(spec_decode_metadata, draft_probs, target_logits, bonus_token_ids)

    def _greedy(
        self,
        metadata: SpecDecodeMetadata,
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
    ) -> list[list[int]]:
        """Greedy rejection: every output token equals target argmax.

        Only the accepted count per seq varies. No softmax needed since
        argmax(logits) == argmax(softmax(logits)). All GPU computation is
        done upfront, then a single batch of .tolist() calls transfers
        results to CPU — no per-token .item() syncs.
        """
        target_argmax = target_logits.argmax(dim=-1)
        matches = metadata.draft_token_ids == target_argmax

        # Batch GPU→CPU transfer (3 syncs vs B*K before).
        target_argmax_list = target_argmax.tolist()
        bonus_list = bonus_token_ids.tolist()
        matches_list = matches.tolist()

        output_token_ids = []
        draft_start = 0
        for req_idx in range(len(metadata.num_draft_tokens)):
            num_drafts = metadata.num_draft_tokens[req_idx]
            if num_drafts == 0:
                output_token_ids.append([bonus_list[req_idx]])
                continue

            # Find first mismatch position (0-indexed within this seq's drafts).
            accepted_count = num_drafts
            for i in range(num_drafts):
                if not matches_list[draft_start + i]:
                    accepted_count = i
                    break

            if accepted_count == num_drafts:
                # All drafts accepted → drafts + bonus.
                output_token_ids.append(
                    target_argmax_list[draft_start:draft_start + num_drafts]
                    + [bonus_list[req_idx]]
                )
            else:
                # Rejected at accepted_count → accepted drafts + recovered.
                output_token_ids.append(
                    target_argmax_list[draft_start:draft_start + accepted_count + 1]
                )
            draft_start += num_drafts
        return output_token_ids

    def _probabilistic(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
    ) -> list[list[int]]:
        """Probabilistic rejection: ratio-based accept, residual sampling on reject.

        Guarantees the output distribution equals direct target model sampling.
        """
        target_probs = torch.softmax(target_logits, dim=-1)
        device = target_logits.device

        output_token_ids = []
        draft_start = 0
        for req_idx in range(len(metadata.num_draft_tokens)):
            num_drafts = metadata.num_draft_tokens[req_idx]
            if num_drafts == 0:
                output_token_ids.append([bonus_token_ids[req_idx].item()])
                continue

            accepted = []
            for i in range(num_drafts):
                idx = draft_start + i
                draft_token = metadata.draft_token_ids[idx].item()
                draft_prob = draft_probs[idx][draft_token].item()
                target_prob = target_probs[idx][draft_token].item()
                accept_prob = min(1.0, target_prob / draft_prob) if draft_prob > 0 else 0.0

                if torch.rand(1, device=device).item() < accept_prob:
                    accepted.append(draft_token)
                else:
                    residual = torch.clamp(target_probs[idx] - draft_probs[idx], min=0)
                    recovered = self._sample_from_residual(residual, device)
                    accepted.append(recovered)
                    break
            else:
                # All drafts accepted → bonus.
                accepted.append(bonus_token_ids[req_idx].item())

            output_token_ids.append(accepted)
            draft_start += num_drafts
        return output_token_ids

    @staticmethod
    def _sample_from_residual(residual: torch.Tensor, device: torch.device) -> int:
        """Sample from residual distribution max(p - q, 0).

        Used in probabilistic mode when draft probabilities are available.
        """
        total = residual.sum()
        if total > 0:
            probs = residual / total
        else:
            probs = torch.ones_like(residual, device=device) / len(residual)
        return torch.multinomial(probs, 1).item()
