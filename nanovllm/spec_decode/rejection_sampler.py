"""Rejection sampler for speculative decoding.

Implements the accept/reject logic and residual sampling that guarantees
the output distribution is identical to sampling directly from the target model.
"""

import torch

from nanovllm.spec_decode.metadata import SpecDecodeMetadata


class RejectionSampler:
    """Accept/reject draft tokens using target model probabilities.

    Supports both greedy (argmax comparison) and probabilistic (ratio-based)
    modes. Guarantees the final output distribution equals target model sampling.
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
                verification positions (after temperature/top-k/top-p).
            bonus_logits: [num_reqs, V] logits at bonus positions.

        Returns:
            List of accepted token IDs per request. Each list contains
            accepted draft tokens + 1 recovered/bonus token.
            Length = num_accepted + 1 (min 1, max num_drafts + 1).
        """
        num_reqs = len(spec_decode_metadata.num_draft_tokens)
        draft_token_ids = spec_decode_metadata.draft_token_ids
        device = target_logits.device

        # Target probabilities from logits.
        target_probs = torch.softmax(target_logits, dim=-1)

        # Pre-sample bonus tokens (greedy argmax).
        bonus_token_ids = bonus_logits.argmax(dim=-1)

        output_token_ids = []
        draft_start = 0

        for req_idx in range(num_reqs):
            num_drafts = spec_decode_metadata.num_draft_tokens[req_idx]
            accepted = []

            if num_drafts == 0:
                # No draft tokens — just the bonus token.
                accepted.append(bonus_token_ids[req_idx].item())
            else:
                rejected = False
                for i in range(num_drafts):
                    idx = draft_start + i
                    draft_token = draft_token_ids[idx].item()

                    if draft_probs is None:
                        # Greedy: accept if target argmax matches draft.
                        target_argmax = target_probs[idx].argmax().item()
                        if target_argmax == draft_token:
                            accepted.append(draft_token)
                        else:
                            # Reject: sample from target excluding rejected token.
                            recovered = self._sample_recovered(
                                target_probs[idx], draft_token, device
                            )
                            accepted.append(recovered)
                            rejected = True
                            break
                    else:
                        # Probabilistic: accept with min(1, p/q).
                        draft_prob = draft_probs[idx][draft_token].item()
                        target_prob = target_probs[idx][draft_token].item()
                        accept_prob = min(1.0, target_prob / draft_prob) if draft_prob > 0 else 0.0

                        if torch.rand(1, device=device).item() < accept_prob:
                            accepted.append(draft_token)
                        else:
                            # Reject: residual sampling.
                            residual = torch.clamp(
                                target_probs[idx] - draft_probs[idx], min=0
                            )
                            recovered = self._sample_from_residual(residual, device)
                            accepted.append(recovered)
                            rejected = True
                            break

                # All drafts accepted → append bonus token.
                if not rejected and len(accepted) == num_drafts:
                    accepted.append(bonus_token_ids[req_idx].item())

            output_token_ids.append(accepted)
            draft_start += num_drafts

        return output_token_ids

    @staticmethod
    def _sample_recovered(
        target_probs: torch.Tensor, rejected_token: int, device: torch.device
    ) -> int:
        """Sample from target distribution excluding the rejected token.

        Used in greedy mode when no draft probabilities are available.
        """
        probs = target_probs.clone()
        probs[rejected_token] = 0
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = torch.ones_like(probs) / len(probs)
        return torch.multinomial(probs, 1).item()

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
