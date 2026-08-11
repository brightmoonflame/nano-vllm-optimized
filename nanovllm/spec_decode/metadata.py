"""Speculative decoding metadata — flattened indices for batch verification.

Converts per-request draft token counts into GPU-ready flattened indices
so the target model can verify all requests' draft tokens in one forward pass.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    """Metadata for speculative decoding verification and sampling."""

    # Flattened draft token IDs across all requests.
    draft_token_ids: torch.Tensor

    # Number of draft tokens per request.
    num_draft_tokens: list[int]

    # Prefix sum of num_draft_tokens.
    cu_num_draft_tokens: torch.Tensor

    # Prefix sum of (num_draft_tokens + 1) — includes bonus position.
    cu_num_sampled_tokens: torch.Tensor

    # Row indices in flattened logits for verifying draft tokens.
    target_logits_indices: torch.Tensor

    # Row indices in flattened logits for bonus token sampling.
    bonus_logits_indices: torch.Tensor

    # Physical positions in target model hidden states for computing logits.
    logits_indices: torch.Tensor


def _get_cumsum_and_arange(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative sum and per-segment arange offsets.

    Example: counts = [3, 0, 2]
    cumsum = [3, 3, 5]
    arange = [0, 1, 2, 0, 1]
    """
    cumsum = np.cumsum(counts, dtype=np.int32)
    total = int(cumsum[-1]) if len(cumsum) > 0 else 0
    if total == 0:
        return cumsum, np.empty(0, dtype=np.int32)
    arange = np.arange(total, dtype=np.int32)
    segment_starts = np.repeat(cumsum - counts, counts)
    arange -= segment_starts
    return cumsum, arange


def make_spec_decode_metadata(
    num_draft_tokens: np.ndarray,
    cu_num_scheduled_tokens: np.ndarray,
) -> SpecDecodeMetadata:
    """Build SpecDecodeMetadata from per-request draft counts.

    Args:
        num_draft_tokens: [num_reqs] draft token count per request.
        cu_num_scheduled_tokens: [num_reqs] cumulative scheduled token end
            positions in the flattened hidden states.

    Returns:
        SpecDecodeMetadata with flattened indices.

    Example:
        cu_num_scheduled_tokens = [4, 104, 107, 207, 209]
        num_draft_tokens        = [3,   0,   2,   0,   1]

        num_sampled_tokens = [4, 1, 3, 1, 2]
        logits_indices     = [0,1,2,3, 103, 104,105,106, 206, 207,208]
        target_logits_indices = [0,1,2, 5,6, 9]
        bonus_logits_indices  = [3, 4, 7, 8, 10]
    """
    # Each request needs draft_count + 1 logits positions (1 bonus).
    num_sampled_tokens = num_draft_tokens + 1
    cu_num_sampled_tokens, arange_sampled = _get_cumsum_and_arange(num_sampled_tokens)

    # logits_indices: physical positions in hidden states.
    # Each request takes its last num_sampled_tokens positions.
    logits_indices = np.repeat(
        cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
    ).astype(np.int64)
    logits_indices += arange_sampled.astype(np.int64)

    # bonus_logits_indices: last position of each request's segment.
    bonus_logits_indices = cu_num_sampled_tokens - 1

    # target_logits_indices: positions for verifying draft tokens.
    cu_num_draft_tokens, arange_draft = _get_cumsum_and_arange(num_draft_tokens)
    target_logits_indices = np.repeat(
        cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
    ).astype(np.int64)
    target_logits_indices += arange_draft.astype(np.int64)

    return SpecDecodeMetadata(
        draft_token_ids=torch.empty(0, dtype=torch.int64),  # filled by caller
        num_draft_tokens=num_draft_tokens.tolist(),
        cu_num_draft_tokens=torch.from_numpy(cu_num_draft_tokens),
        cu_num_sampled_tokens=torch.from_numpy(cu_num_sampled_tokens),
        target_logits_indices=torch.from_numpy(target_logits_indices),
        bonus_logits_indices=torch.from_numpy(bonus_logits_indices),
        logits_indices=torch.from_numpy(logits_indices),
    )
