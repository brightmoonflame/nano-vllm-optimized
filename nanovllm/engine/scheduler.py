from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.prefill_chunk_size = config.prefill_chunk_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.speculative_config = config.speculative_config
        self.num_spec_tokens = self.speculative_config.get("num_spec_tokens", 5) if self.speculative_config else 0
        # Seq ids preempted since the last step, drained by llm_engine.step()
        # to drop their draft-side state (spec decode only).
        self.preempted_seq_ids: list[int] = []

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # Spec decode: force a full-recompute allocation for this request so
                # the target forward produces aux hidden states for every position —
                # see BlockManager.can_allocate's docstring for why a prefix-cache
                # hit would otherwise leave a gap in the draft proposer's KV.
                num_cached_blocks = self.block_manager.can_allocate(seq, enable_prefix_cache=self.speculative_config is None)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        num_tokens = self.num_spec_tokens + 1 if self.speculative_config else 1
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq, num_tokens):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = num_tokens
                seq.is_prefill = False
                self.block_manager.may_append(seq, num_tokens)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        if self.speculative_config is not None:
            # Re-prefill re-extends the draft from scratch, so the stale
            # committed draft KV/aux/draft0_logits must be dropped —
            # otherwise the next extend() would append on top of it.
            self.preempted_seq_ids.append(seq.seq_id)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id in self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def postprocess_spec(self, seqs: list[Sequence], token_ids_list: list[list[int]]):
        """Consume spec-decode output: each seq gets 1..K+1 accepted tokens.

        Accepted = accepted drafts + 1 bonus/recovered token. Only the
        accepted drafts' KV is committed (len(accepted) - 1 rows): the
        final token's KV was never correctly written — its verify row
        held a rejected draft (or didn't exist, when all K drafts were
        accepted). It becomes next round's last_token and its KV is
        (re)written by the next verify window's row 0.
        """
        for seq, accepted_tokens in zip(seqs, token_ids_list):
            finished = False
            for token_id in accepted_tokens:
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id in self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    finished = True
                    break
            if finished:
                seq.num_scheduled_tokens = 0
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            else:
                seq.num_scheduled_tokens = len(accepted_tokens) - 1
                self.block_manager.hash_blocks(seq)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0

    # ------------------------------------------------------------------
    # Chunked prefill: decode and prefill chunks are scheduled in the same step.
    # Enabled by config.enable_chunked_prefill; the original schedule()/postprocess()
    # above remain unchanged when the flag is off.
    # ------------------------------------------------------------------

    def schedule_chunked(self) -> tuple[list[Sequence], dict[int, int]]:
        scheduled_seqs = []
        seqlen_this_time: dict[int, int] = {}
        token_budget = self.max_num_batched_tokens
        preempted = False

        # 1. Decode pass: schedule tokens for each running seq (non-blocking).
        #    With spec decode, blocks are allocated for K+1 positions up front so
        #    a later pure-decode step can verify drafts; a mixed batch still runs
        #    single-token decode (spec not supported there).
        num_decode_tokens = self.num_spec_tokens + 1 if self.speculative_config else 1
        decode_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs and token_budget > 0:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq, num_decode_tokens):
                if self.running:
                    self.preempt(self.running.pop())
                    preempted = True
                else:
                    self.preempt(seq)
                    preempted = True
                    break
            else:
                seq.num_scheduled_tokens = num_decode_tokens
                seq.is_prefill = False
                self.block_manager.may_append(seq, num_decode_tokens)
                decode_seqs.append(seq)
                scheduled_seqs.append(seq)
                seqlen_this_time[seq.seq_id] = 1
                token_budget -= 1
        self.running.extendleft(reversed(decode_seqs))

        # 2. Prefill pass: only when no preemption happened this step (avoids thrashing).
        #    Uses remaining token_budget, capped by prefill_chunk_size per chunk.
        #    Pop each seq first so we can schedule chunks from multiple requests in one step.
        if not preempted:
            partial_prefill_seqs = []
            while self.waiting and token_budget > 0 and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.waiting.popleft()
                if not seq.block_table:
                    # Same prefix-cache opt-out as schedule() above.
                    num_cached_blocks = self.block_manager.can_allocate(seq, enable_prefix_cache=self.speculative_config is None)
                    if num_cached_blocks == -1:
                        self.waiting.appendleft(seq)
                        break
                    self.block_manager.allocate(seq, num_cached_blocks)
                    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                else:
                    num_tokens = seq.num_tokens - seq.num_cached_tokens

                # Cap by: remaining budget, chunk size limit, and max_model_len boundary.
                num_new = min(num_tokens, token_budget, self.prefill_chunk_size)
                num_new = min(num_new, self.max_model_len - 1 - seq.num_cached_tokens)
                if num_new <= 0:
                    self.waiting.appendleft(seq)
                    break

                seq.num_scheduled_tokens = num_new
                seq.is_prefill = True
                token_budget -= num_new
                seqlen_this_time[seq.seq_id] = num_new

                if seq.num_cached_tokens + num_new == seq.num_tokens:
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                else:
                    partial_prefill_seqs.append(seq)
                scheduled_seqs.append(seq)

            # Put unfinished prefill seqs back to the front of waiting.
            for seq in reversed(partial_prefill_seqs):
                self.waiting.appendleft(seq)

        # Mixed batch: run_chunked processes decode seqs as single-token, so
        # num_scheduled_tokens must reflect what actually runs this step.
        if self.speculative_config is not None and any(seq.is_prefill for seq in scheduled_seqs):
            for seq in decode_seqs:
                seq.num_scheduled_tokens = 1

        assert scheduled_seqs
        return scheduled_seqs, seqlen_this_time

    def postprocess_chunked(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            was_prefill = seq.is_prefill
            seq.num_scheduled_tokens = 0
            # Only finished-prefill and decode seqs produce tokens.
            if was_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id in self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
