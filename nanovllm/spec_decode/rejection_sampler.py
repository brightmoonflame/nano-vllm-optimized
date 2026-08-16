"""Rejection sampler for speculative decoding.

Implements the accept/reject logic and residual sampling that guarantees
the output distribution is identical to sampling directly from the target model.
"""

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.spec_decode.metadata import SpecDecodeMetadata


class RejectionSampler:
    """Accept/reject draft tokens using target model probabilities.

    Supports both greedy (argmax comparison) and probabilistic (ratio-based)
    acceptance, chosen per request by its temperature, so one batch may mix
    greedy and sampling requests. Greedy requests are fully deterministic —
    every output token equals target argmax, identical to non-speculative
    greedy decoding. Sampling requests follow Leviathan et al. (2023):
    draft token x ~ q is accepted with probability min(1, p(x)/q(x)); on
    rejection a replacement is drawn from norm(max(0, p - q)), and when all
    drafts are accepted the bonus token is drawn from p. p is the target
    distribution after the request's temperature/top-k/top-p (mirroring the
    engine's non-speculative Sampler), so spec-decode output matches direct
    target sampling in distribution.
    """

    def __init__(self, sampler: Sampler):
        # The engine's Sampler, reused for bonus tokens so spec and
        # non-spec runs share identical temperature/top-k/top-p semantics.
        # (vLLM's signature is RejectionSampler(sampler, spec_config,
        # device); the latter two are omitted here — tensors carry their
        # own device and there is no spec-config object to consult.)
        self.sampler = sampler

    @torch.inference_mode()
    def __call__(
        self,
        spec_decode_metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        temperatures: list[float],
        top_ks: list[int],
        top_ps: list[float],
    ) -> list[list[int]]:
        """Perform rejection sampling on draft tokens.

        Args:
            spec_decode_metadata: Flattened indices for batch verification.
            draft_probs: [total_drafts, V] draft probabilities q in
                target-vocab space (zero outside the draft's hot set).
                None only when every request in the batch is greedy.
            target_logits: [total_drafts, V] target model logits at draft
                verification positions.
            bonus_logits: [num_reqs, V] logits at bonus positions.
            temperatures / top_ks / top_ps: per-request sampling params.

        Returns:
            List of accepted token IDs per request. Each list contains
            accepted draft tokens + 1 recovered/bonus token.
            Length = num_accepted + 1 (min 1, max num_drafts + 1).
        """
        # Bonus token: argmax for greedy requests; sampled from the
        # request's own filtered distribution otherwise (via the same
        # Sampler the non-speculative path uses).
        bonus_token_ids = bonus_logits.argmax(dim=-1)
        sampling_reqs = [i for i, t in enumerate(temperatures) if t > 0]
        if sampling_reqs:
            device = bonus_logits.device
            idx = torch.tensor(sampling_reqs, dtype=torch.long, device=device)
            temps = torch.tensor([temperatures[i] for i in sampling_reqs], dtype=torch.float32, device=device)
            ks = [top_ks[i] for i in sampling_reqs]
            ps = [top_ps[i] for i in sampling_reqs]
            top_ks_t = torch.tensor(ks, dtype=torch.int32, device=device) if any(k != -1 for k in ks) else None
            top_ps_t = torch.tensor(ps, dtype=torch.float32, device=device) if any(p != 1.0 for p in ps) else None
            bonus_token_ids[idx] = self.sampler(bonus_logits[idx], temps, top_ks_t, top_ps_t)

        if draft_probs is None:
            return self._greedy(spec_decode_metadata, target_logits, bonus_token_ids)
        return self._probabilistic(
            spec_decode_metadata, draft_probs, target_logits, bonus_token_ids,
            temperatures, top_ks, top_ps,
        )

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
        temperatures: list[float],
        top_ks: list[int],
        top_ps: list[float],
    ) -> list[list[int]]:
        """Ratio-based acceptance with residual sampling on rejection.

        Greedy requests inside the batch keep argmax acceptance (their
        draft_probs rows are ignored). All GPU work is batched upfront —
        one filtered softmax over the target logits, one gather for
        p(x)/q(x), one uniform draw, one multinomial over every rejection
        row's residual — followed by a single batch of .tolist() syncs.
        """
        device = target_logits.device

        # Expand per-request sampling params to per-draft-row.
        row_temps = [t if t > 0 else 1.0 for t, n in zip(temperatures, metadata.num_draft_tokens) for _ in range(n)]
        row_ks = [k for k, n in zip(top_ks, metadata.num_draft_tokens) for _ in range(n)]
        row_ps = [p for p, n in zip(top_ps, metadata.num_draft_tokens) for _ in range(n)]
        temps_t = torch.tensor(row_temps, dtype=torch.float32, device=device)
        top_ks_t = torch.tensor(row_ks, dtype=torch.int32, device=device) if any(k != -1 for k in row_ks) else None
        top_ps_t = torch.tensor(row_ps, dtype=torch.float32, device=device) if any(p != 1.0 for p in row_ps) else None
        target_probs = self._filtered_probs(target_logits, temps_t, top_ks_t, top_ps_t)

        draft_tokens = metadata.draft_token_ids
        target_argmax = target_logits.argmax(dim=-1)
        p_x = target_probs.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)
        q_x = draft_probs.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)
        # q(x) > 0 for every proposed token by construction; the guard only
        # covers float underflow, where rejection is the safe fallback.
        accept_prob = torch.where(q_x > 0, (p_x / q_x).clamp(max=1.0), torch.zeros_like(q_x))
        accept_mask = torch.rand_like(accept_prob) < accept_prob

        # Single batch of GPU→CPU syncs.
        argmax_list = target_argmax.tolist()
        match_list = (draft_tokens == target_argmax).tolist()
        accept_list = accept_mask.tolist()
        draft_list = draft_tokens.tolist()
        bonus_list = bonus_token_ids.tolist()

        output_token_ids = []
        residual_slots = []    # (output idx, flattened draft row) pending a residual draw
        draft_start = 0
        for req_idx in range(len(metadata.num_draft_tokens)):
            num_drafts = metadata.num_draft_tokens[req_idx]
            if num_drafts == 0:
                output_token_ids.append([bonus_list[req_idx]])
                continue

            if temperatures[req_idx] <= 0:
                # Greedy request: same argmax acceptance as _greedy.
                accepted_count = num_drafts
                for i in range(num_drafts):
                    if not match_list[draft_start + i]:
                        accepted_count = i
                        break
                if accepted_count == num_drafts:
                    output_token_ids.append(
                        argmax_list[draft_start:draft_start + num_drafts]
                        + [bonus_list[req_idx]]
                    )
                else:
                    output_token_ids.append(
                        argmax_list[draft_start:draft_start + accepted_count + 1]
                    )
            else:
                # Find first rejection position.
                rejected = num_drafts
                for i in range(num_drafts):
                    if not accept_list[draft_start + i]:
                        rejected = i
                        break
                accepted = draft_list[draft_start:draft_start + rejected]
                if rejected == num_drafts:
                    # All drafts accepted → bonus.
                    accepted.append(bonus_list[req_idx])
                else:
                    residual_slots.append((len(output_token_ids), draft_start + rejected))
                output_token_ids.append(accepted)
            draft_start += num_drafts

        if residual_slots:
            rows = torch.tensor([row for _, row in residual_slots], dtype=torch.long, device=device)
            residual = (target_probs[rows] - draft_probs[rows]).clamp_min_(0)
            # A rejected row always has positive residual mass (q(x) > p(x)
            # at the rejected token); the uniform fallback only guards
            # float dust.
            total = residual.sum(dim=-1, keepdim=True)
            residual = torch.where(
                total > 0,
                residual / total.clamp_min(1e-30),
                torch.full_like(residual, 1.0 / residual.size(-1)),
            )
            recovered = torch.multinomial(residual, 1).squeeze(1).tolist()
            for (out_idx, _), token in zip(residual_slots, recovered):
                output_token_ids[out_idx].append(token)
        return output_token_ids

    @staticmethod
    def _filtered_probs(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None,
        top_ps: torch.Tensor | None,
    ) -> torch.Tensor:
        """Softmax after temperature scaling and top-k/top-p filtering,
        mirroring `layers/sampler.Sampler` step for step so the acceptance
        distribution equals the engine's non-speculative sampling
        distribution."""
        logits = logits.float() / temperatures.unsqueeze(1)
        if top_ks is not None:
            # Keep only the top-k logits per row; mask the rest to -inf.
            k = int(top_ks.max())
            vals, idxs = logits.topk(k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, idxs, vals)
            logits = mask

        probs = torch.softmax(logits, dim=-1)
        if top_ps is not None:
            # Same nucleus rule as Sampler: keep while cumulative
            # probability stays within top_p (at least one token), then
            # renormalize — applied in sorted space and scattered back.
            sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
            keep = sorted_probs.cumsum(dim=-1) <= top_ps.unsqueeze(1)
            keep[..., 0] = True
            sorted_probs = sorted_probs * keep
            probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs
