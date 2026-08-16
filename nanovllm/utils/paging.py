"""Shared paged-KV-cache helpers.

Both the target model runner (`engine/model_runner.py`) and the EAGLE3
draft proposer (`spec_decode/proposer.py`) need to turn a per-sequence
`block_table` + an absolute-position range into physical cache slot ids,
and to pad a batch of variable-length block tables into a rectangular
tensor. Factored out here so the draft side can share the exact slot
addressing scheme used by the target side (required for block-id sharing
to be correct — see SPEC_DECODE_PLAN.md 4a).
"""

import torch


def compute_slot_mapping(block_table: list[int], start: int, end: int, block_size: int) -> list[int]:
    """Physical cache slot ids for the half-open absolute position range
    [start, end), given a sequence's block table.

    slot(p) = block_table[p // block_size] * block_size + p % block_size
    — the same addressing scheme `model_runner.py`'s prepare_prefill /
    prepare_spec_decode / prepare_chunked use for the target model, so a
    draft cache sharing the same block ids stays consistent with it.
    """
    slots = []
    start_block = start // block_size
    end_block = (end + block_size - 1) // block_size
    for i in range(start_block, end_block):
        slot_start = block_table[i] * block_size
        if i == start_block:
            slot_start += start % block_size
        if i != end_block - 1:
            slot_end = block_table[i] * block_size + block_size
        else:
            slot_end = block_table[i] * block_size + end - i * block_size
        slots.extend(range(slot_start, slot_end))
    return slots


def pad_block_tables(block_tables: list[list[int]], device=None) -> torch.Tensor:
    """Pad variable-length block tables with -1 and stack into [B, max_len]."""
    max_len = max(len(bt) for bt in block_tables)
    padded = [bt + [-1] * (max_len - len(bt)) for bt in block_tables]
    return torch.tensor(padded, dtype=torch.int32, device=device)
