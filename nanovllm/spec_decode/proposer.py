"""Draft model proposer for speculative decoding.

Generates K candidate tokens using an independent draft model.
Interface is compatible with EAGLE3 — hidden_states parameter is reserved.
"""

import torch
from transformers import AutoModelForCausalLM


class Proposer:
    """Draft model proposer.

    Loads a small draft model and autoregressively generates K candidate
    tokens for each request. All requests are batched into a single forward
    pass per step (left-padded prefill + K-1 batched decode steps).
    """

    def __init__(self, draft_model_path: str, num_spec_tokens: int = 5):
        self.num_spec_tokens = num_spec_tokens
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        self.draft_model.eval()

    @torch.inference_mode()
    def propose(
        self,
        target_token_ids: list[list[int]],
        target_hidden_states: torch.Tensor | None = None,
        num_spec_tokens: int | None = None,
    ) -> list[list[int]]:
        """Generate K draft tokens per request, batched across all requests.

        Args:
            target_token_ids: Token ID sequences per request (current context).
            target_hidden_states: Reserved for EAGLE3 (unused in classic mode).
            num_spec_tokens: Number of draft tokens to generate. Defaults to
                self.num_spec_tokens.

        Returns:
            List of draft token ID lists, one per request. Each inner list
            has length K.
        """
        K = num_spec_tokens or self.num_spec_tokens
        B = len(target_token_ids)
        device = next(self.draft_model.parameters()).device

        # Left-pad to uniform length for batched prefill.
        max_len = max(len(ids) for ids in target_token_ids)
        pad_id = getattr(self.draft_model.config, "pad_token_id", 0) or 0
        input_ids = torch.tensor(
            [[pad_id] * (max_len - len(ids)) + ids for ids in target_token_ids],
            dtype=torch.long, device=device,
        )
        attention_mask = torch.tensor(
            [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in target_token_ids],
            dtype=torch.long, device=device,
        )
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        # Batched prefill: one forward for all seqs.
        outputs = self.draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # Collect K draft tokens as [B] tensors, one per step.
        draft_steps = [outputs.logits[:, -1, :].argmax(dim=-1)]

        for _ in range(K - 1):
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(B, 1, dtype=torch.long, device=device),
            ], dim=1)
            outputs = self.draft_model(
                input_ids=draft_steps[-1].unsqueeze(-1),
                attention_mask=attention_mask,
                position_ids=attention_mask.sum(dim=1, keepdim=True) - 1,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            draft_steps.append(outputs.logits[:, -1, :].argmax(dim=-1))

        # Single GPU→CPU sync: [K steps, B] → [B, K].
        return torch.stack(draft_steps, dim=1).tolist()
