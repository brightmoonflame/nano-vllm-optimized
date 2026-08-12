"""Draft model proposer for speculative decoding.

Generates K candidate tokens using an independent draft model.
Interface is compatible with EAGLE3 — hidden_states parameter is reserved.
"""

import torch
from transformers import AutoModelForCausalLM


class Proposer:
    """Draft model proposer.

    Loads a small draft model and autoregressively generates K candidate
    tokens for each request. Uses KV cache for efficient incremental decoding.
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
        """Generate K draft tokens per request.

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
        all_draft_ids = []

        for token_ids in target_token_ids:
            input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
            past_key_values = None
            draft_ids = []

            for _ in range(K):
                outputs = self.draft_model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                # Greedy sampling: argmax of last position logits.
                next_token = outputs.logits[:, -1, :].argmax(dim=-1).item()
                draft_ids.append(next_token)
                past_key_values = outputs.past_key_values
                input_ids = torch.tensor([[next_token]], dtype=torch.long, device="cuda")

            all_draft_ids.append(draft_ids)

        return all_draft_ids
