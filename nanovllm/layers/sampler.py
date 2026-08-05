import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile(dynamic=True)
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
    ):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        if top_ks is None and top_ps is None:
            # Fast path: full-vocab softmax + exponential race (original behavior).
            probs = torch.softmax(logits, dim=-1)
            sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
            return sample_tokens

        # Filtering path
        if top_ks is not None:
            # Keep only the top-k logits per row; mask the rest to -inf.
            # k stays a tensor to avoid a GPU→CPU sync (graph break) inside @torch.compile.
            k = top_ks.max()
            vals, idxs = logits.topk(k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, idxs, vals)
            logits = mask

        if top_ps is not None:
            # Nucleus sampling: keep the smallest set of tokens whose
            # cumulative probability reaches top_p (at least one token).
            sorted_probs, sorted_indices = torch.softmax(logits, dim=-1).sort(dim=-1, descending=True)
            cum = sorted_probs.cumsum(dim=-1)
            keep = cum <= top_ps.unsqueeze(1)
            keep[..., 0] = True  # guarantee at least one candidate
            sorted_probs = sorted_probs * keep  # zero out filtered tokens
            probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)  # renormalize
            sample_in_sorted = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
            # Map sorted positions back to original vocab indices
            sample_tokens = sorted_indices.gather(1, sample_in_sorted.unsqueeze(1)).squeeze(1)
            return sample_tokens

        # top_k only, no top_p: softmax over full vocab (filtered positions are -inf → 0)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
