import pickle
import numpy as np
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

# warmup, CUDA-graph capture batch sizes and prefill shapes each need their own
# compiled variant; the default limit (8) causes silent eager fallback.
torch._dynamo.config.cache_size_limit = 64

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.llama import LlamaForCausalLM
from nanovllm.models.gemma3 import Gemma3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.triton_attn import mid_buffer_size
from nanovllm.layers.kv_quant import NUM_GROUPS
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.spec_decode.metadata import make_spec_decode_metadata
from nanovllm.spec_decode.proposer import Proposer
from nanovllm.spec_decode.rejection_sampler import RejectionSampler

# Maps hf_config.model_type to the corresponding model class.
# New architectures are added here — model code lives in nanovllm/models/.
model_dict = {
    "qwen3": Qwen3ForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "llama": LlamaForCausalLM,
    "gemma3": Gemma3ForCausalLM,
    "gemma3_text": Gemma3ForCausalLM,
}


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = model_dict[hf_config.model_type](hf_config)
        load_model(self.model, config.model)
        # Must happen before warmup_model(): warmup runs a real prefill forward,
        # so the switch needs to be in place for it to exercise the Triton path.
        for module in self.model.modules():
            if hasattr(module, "k_cache"):
                module.use_triton_attn = config.use_triton_attn
        self.sampler = Sampler()
        # Draft model is a standalone HF model (not TP-sharded), so only rank 0
        # proposes + rejection-samples; other ranks only run the target forward.
        self.speculative_config = config.speculative_config
        self.warmup_model()
        # Must be constructed before allocate_kv_cache(): the draft model's
        # attention module needs to exist so allocate_kv_cache() can assign
        # its paged KV cache tensor onto it (same pass as the target's).
        if self.speculative_config is not None and rank == 0:
            self.proposer = Proposer(
                self.speculative_config["model"],
                target_model=self.model,
                block_size=self.block_size,
                num_spec_tokens=self.speculative_config.get("num_spec_tokens", 5),
                aux_layer_ids=self.speculative_config.get("aux_layer_ids"),
            )
            self.rejection_sampler = RejectionSampler(self.sampler)
        self.allocate_kv_cache()
        # KV quant's default path uses dynamic whole-cache dequant at decode
        # time — incompatible with CUDA graph. With the fused Triton INT8
        # kernel there is no dynamic op, so graphs are safe to capture.
        if config.use_triton_attn:
            self._alloc_mid_buffer()
        if not self.enforce_eager and not (config.kv_quant and not config.use_triton_attn):
            self.capture_cudagraph()
            # Prefill graphs never pass aux_layer_ids, so they can't produce the
            # aux hidden states extend() needs — keep prompt prefill on the eager
            # path whenever spec decode is on rather than teaching the graph
            # about a second output tensor for a niche combination.
            if config.enable_prefill_cudagraph and config.speculative_config is None:
                self.capture_prefill_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def drop_proposer_state(self, seq_ids: list[int]) -> None:
        """Free a finished/preempted sequence's draft-side state.

        Only rank 0 owns the proposer; spec-off runs never call this.
        """
        proposer = getattr(self, "proposer", None)
        if proposer is None:
            return
        for sid in seq_ids:
            proposer.drop(sid)

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if hasattr(self, 'graphs'):
            del self.graphs, self.graph_pool
        if hasattr(self, 'prefill_graphs'):
            del self.prefill_graphs, self.prefill_graph_vars, self.prefill_graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # Bytes per block: INT8 uses 1 byte per element, BF16 uses 2.
        elem_size = 1 if config.kv_quant else hf_config.dtype.itemsize
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * elem_size
        if config.kv_quant:
            # Per-(token, head, group) FP32 scales (both K and V): group-wise
            # along head_dim isolates outliers without calibration.
            scale_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * NUM_GROUPS * 4
            block_bytes += scale_bytes

        # Draft (EAGLE3) KV cache: 1 layer, always full precision (kv_quant
        # is a target-only optimization here), same num_kvcache_blocks /
        # block_size as the target so block ids stay directly shareable.
        proposer = getattr(self, "proposer", None)
        draft_num_kv_heads = draft_head_dim = None
        if proposer is not None:
            draft_config = proposer.draft.config
            draft_num_kv_heads = getattr(draft_config, "num_key_value_heads", draft_config.num_attention_heads)
            draft_head_dim = getattr(draft_config, "head_dim", draft_config.hidden_size // draft_config.num_attention_heads)
            block_bytes += 2 * self.block_size * draft_num_kv_heads * draft_head_dim * hf_config.dtype.itemsize

        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        cache_dtype = torch.int8 if config.kv_quant else hf_config.dtype
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, dtype=cache_dtype)
        if config.kv_quant:
            # Group-wise scales: (layers, blocks, block_size, kv_heads, NUM_GROUPS).
            self.kv_scales = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, NUM_GROUPS, dtype=torch.float32)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                if config.kv_quant:
                    module.k_scale = self.kv_scales[0, layer_id]
                    module.v_scale = self.kv_scales[1, layer_id]
                    module.kv_quant = True
                layer_id += 1

        if proposer is not None:
            self.draft_kv_cache = torch.empty(
                2, config.num_kvcache_blocks, self.block_size, draft_num_kv_heads, draft_head_dim, dtype=hf_config.dtype
            )
            attn = proposer.draft.midlayer.self_attn.attn
            attn.k_cache = self.draft_kv_cache[0]
            attn.v_cache = self.draft_kv_cache[1]

        kv_mb = self.kv_cache.numel() * self.kv_cache.element_size() / 1e6
        if config.kv_quant:
            kv_mb += self.kv_scales.numel() * self.kv_scales.element_size() / 1e6
        if proposer is not None:
            kv_mb += self.draft_kv_cache.numel() * self.draft_kv_cache.element_size() / 1e6
        per_block_mb = kv_mb / config.num_kvcache_blocks
        dtype_str = 'INT8' if config.kv_quant else 'BF16'
        print(f"KV cache: {config.num_kvcache_blocks} blocks × {per_block_mb:.1f} MB/block = {kv_mb:.1f} MB ({dtype_str})")

    def _alloc_mid_buffer(self):
        """Pre-allocate one shared flash-decoding mid buffer for all target
        attention layers. Layers execute sequentially within a forward, so a
        single buffer sized for the largest (batch, splits) combination
        suffices. This removes per-step torch.empty in the split path and
        keeps those kernels CUDA-graph capturable."""
        mods = [m for m in self.model.modules() if getattr(m, "use_triton_attn", False)]
        max_bs = min(self.config.max_num_seqs, 512)
        need = max((mid_buffer_size(m.num_heads, m.head_dim, max_bs) for m in mods), default=0)
        if need:
            buf = torch.empty(need, dtype=torch.float32, device="cuda")
            for m in mods:
                m.mid_buffer = buf

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_spec_decode(self, seqs: list[Sequence]):
        """Build a flattened batch for speculative verification.

        Each seq contributes [last_token, draft_0, ..., draft_{K-1}]
        continuing from its cached context — structurally identical to a
        prefill chunk, so the same varlen/paged-attention kernel verifies
        all requests' drafts in one forward pass (causal within the chunk,
        so draft_i's hidden state depends on draft_0..i-1 as if accepted).

        NOTE: assumes seq.block_table already has enough blocks allocated
        for the K+1 new positions — scheduler-side multi-token block
        allocation is a follow-up, not yet implemented.
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        num_draft_tokens = []
        for seq in seqs:
            chunk = [seq.last_token] + seq.spec_token_ids
            start = len(seq) - 1
            end = start + len(chunk)
            input_ids.extend(chunk)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + len(chunk))
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(len(chunk), max_seqlen_q)
            max_seqlen_k = max(end, max_seqlen_k)
            num_draft_tokens.append(len(seq.spec_token_ids))

            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        block_tables = self.prepare_block_tables(seqs)
        spec_metadata = make_spec_decode_metadata(
            np.array(num_draft_tokens, dtype=np.int32),
            np.array(cu_seqlens_q[1:], dtype=np.int32),
        )
        spec_metadata.draft_token_ids = torch.tensor(
            [t for seq in seqs for t in seq.spec_token_ids], dtype=torch.int64
        ).cuda(non_blocking=True)
        spec_metadata.logits_indices = spec_metadata.logits_indices.cuda(non_blocking=True)
        spec_metadata.target_logits_indices = spec_metadata.target_logits_indices.cuda(non_blocking=True)
        spec_metadata.bonus_logits_indices = spec_metadata.bonus_logits_indices.cuda(non_blocking=True)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # Always pass block_tables: K/V for the verified chunk is read from paged cache.
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions, spec_metadata

    def _spec_aux_layer_ids(self) -> list[int] | None:
        """aux_layer_ids to request from the target forward when spec decode
        is on. Only rank 0 owns the proposer / consumes the aux states.
        getattr guard: the proposer is created after warmup_model(), so
        warmup's prefill forward sees None and skips aux capture."""
        proposer = getattr(self, "proposer", None)
        return proposer.aux_layer_ids if proposer is not None else None

    def _target_forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Target forward, requesting aux hidden states when spec is on.
        Only LlamaForCausalLM.forward accepts the third arg — pass it
        conditionally so other archs (qwen/gemma) keep working."""
        aux_ids = self._spec_aux_layer_ids()
        if aux_ids is not None:
            return self.model(input_ids, positions, aux_ids)
        return self.model(input_ids, positions)

    def _extend_verify_aux(self, seqs: list[Sequence], accepted: list[list[int]]) -> None:
        """Catch the draft up on this verify round's accepted tokens,
        immediately (no cross-round stash) — using the target's true aux
        hidden states for the tokens it just confirmed. Called right
        after rejection sampling, before the next round's `propose`.

        Verify window = [last_token, draft_0, ..., draft_{K-1}] @ positions
        [start .. start+K]. EAGLE shifted-token pairing (vLLM eagle.py:
        input_ids shifted by one, positions/hidden_states not): token a_i
        @ seq position start+1+i pairs with the hidden that *predicted*
        it, f_{start+i} — exactly window row i. The bonus token (all K
        accepted, @ start+K+1) pairs with f_{start+K}, the last window
        row, so no row is ever missing and no padding is needed.
        """
        aux = self.model.model._aux_hidden_states
        if aux is None:
            return
        offset = 0
        ext_tokens, ext_positions, ext_aux = [], [], []
        for seq, out in zip(seqs, accepted):
            start = len(seq) - 1
            ext_tokens.append(out)
            ext_positions.append(list(range(start, start + len(out))))
            ext_aux.append(aux[offset:offset + len(out)])
            offset += 1 + len(seq.spec_token_ids)
        self.proposer.extend(seqs, ext_tokens, ext_positions, ext_aux)

    @torch.inference_mode()
    def run_spec(self, seqs: list[Sequence]) -> list[list[int]] | None:
        """Draft-propose-then-verify speculative decoding step.

        1. Draft model autoregressively proposes K candidate tokens per seq
           (argmax for greedy seqs, sampled from the draft distribution
           otherwise).
        2. Target model verifies all candidates for all seqs in one forward pass.
        3. RejectionSampler accepts/rejects per seq — greedy seqs by argmax
           comparison, sampling seqs by the speculative-sampling ratio test
           with residual replacement — so output matches non-speculative
           decoding (greedy: token-for-token; sampling: in distribution).

        Returns one accepted-token-id list per seq (length 1..K+1), or None
        on non-zero ranks (sampling only happens on rank 0).

        Precondition: `extend` has already been called for every seq in
        `seqs` — either by the prefill path (`_extend_prefill_aux`, first
        round) or by the previous round's verify pass
        (`_extend_verify_aux`, at the end of this same method).
        """
        # Only rank 0 owns the draft model. Broadcast the proposals so every
        # rank builds the identical verification batch — the TP forward is
        # collective, so all ranks must see the same shapes and token ids.
        # draft_probs stays rank-0-local: only rank 0 rejection-samples.
        if self.rank == 0:
            draft_lists, draft_probs = self.proposer.propose(seqs)
        else:
            draft_lists, draft_probs = None, None
        if self.world_size > 1:
            obj = [draft_lists]
            dist.broadcast_object_list(obj, src=0)
            draft_lists = obj[0]
        for seq, draft_ids in zip(seqs, draft_lists):
            seq.spec_token_ids = draft_ids

        input_ids, positions, spec_metadata = self.prepare_spec_decode(seqs)
        hidden_states = self._target_forward(input_ids, positions)
        if self.rank != 0:
            return None

        # Reset prefill flag so ParallelLMHead doesn't extract only last-token
        # logits — we need logits at all K+1 positions for rejection sampling.
        set_context(False)
        logits = self.model.compute_logits(hidden_states[spec_metadata.logits_indices])
        target_logits = logits[spec_metadata.target_logits_indices]
        bonus_logits = logits[spec_metadata.bonus_logits_indices]
        accepted = self.rejection_sampler(
            spec_metadata, draft_probs, target_logits, bonus_logits,
            [seq.temperature for seq in seqs],
            [seq.top_k for seq in seqs],
            [seq.top_p for seq in seqs],
        )
        self._extend_verify_aux(seqs, accepted)
        return accepted

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ks = [seq.top_k for seq in seqs]
        top_ps = [seq.top_p for seq in seqs]
        # Only build tensors when filtering is actually requested, so the default path stays untouched.
        top_ks_tensor = None if all(k == -1 for k in top_ks) else torch.tensor(top_ks, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        top_ps_tensor = None if all(p == 1.0 for p in top_ps) else torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures, top_ks_tensor, top_ps_tensor

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill:
            if self._can_replay_prefill_graph(input_ids):
                return self.run_prefill_cudagraph(input_ids, positions)
            return self.model.compute_logits(self._target_forward(input_ids, positions))

        if self.enforce_eager or input_ids.size(0) > 512 or self.config.kv_quant:
            return self.model.compute_logits(self.model(input_ids, positions))

        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_teacher_forced(self, seqs: list[Sequence], is_prefill: bool) -> torch.Tensor:
        """Teacher-forced forward used only by the accuracy eval path.

        Unlike `run`: performs NO sampling/argmax and returns the raw next-token
        logits. For prefill it returns per-position logits for the whole
        (packed) prompt — (total_tokens, vocab) — via the `full_logits` context
        flag, so the caller can score the continuation of every position. For
        decode it returns the single forced token's logits (num_seqs, vocab).

        Callers drive scheduler state themselves (this method does not touch
        Sequence bookkeeping beyond what prepare_* reads).
        """
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        context = get_context()
        context.full_logits = True
        logits = self.model.compute_logits(self._target_forward(input_ids, positions))
        reset_context()
        return logits

    def _can_replay_prefill_graph(self, input_ids: torch.Tensor) -> bool:
        """Prefill graphs safely support one non-prefix sequence per replay.

        A multi-sequence varlen batch has a dynamic cu_seqlens shape and needs a
        separate graph for every (sequence-count, token-count) pair. It falls
        back to the existing eager path to preserve correctness.
        """
        context = get_context()
        return (
            hasattr(self, "prefill_graphs")
            and context.block_tables is None
            and context.cu_seqlens_q.numel() == 2
            and input_ids.numel() <= self.prefill_graph_tokens[-1]
        )

    def run_prefill_cudagraph(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Replay a token-bucketed single-sequence prefill graph.

        Padding is appended after the real prompt, so causal attention leaves
        the hidden state of the prompt's final token unchanged. Padding slots
        are -1, preventing writes to the KV cache.
        """
        actual_tokens = input_ids.numel()
        bucket = next(size for size in self.prefill_graph_tokens if size >= actual_tokens)
        graph_vars = self.prefill_graph_vars[bucket]
        graph_vars["input_ids"][:actual_tokens] = input_ids
        graph_vars["positions"][:actual_tokens] = positions
        if actual_tokens < bucket:
            graph_vars["input_ids"][actual_tokens:].zero_()
            graph_vars["positions"][actual_tokens:].zero_()
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:actual_tokens] = get_context().slot_mapping
        self.prefill_graphs[bucket].replay()

        # The captured context has cu_seqlens=[0, bucket], but only the real
        # prompt's last hidden state should feed the LM head.
        set_context(False)
        return self.model.compute_logits(graph_vars["outputs"][actual_tokens - 1:actual_tokens])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | list[list[int]]:
        # Speculative decode is only used for pure-decode steps; each entry in
        # token_ids becomes a list of accepted tokens instead of a single int —
        # Scheduler.postprocess needs a matching update to consume it (follow-up).
        if not is_prefill and self.speculative_config is not None:
            token_ids = self.run_spec(seqs)
        else:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            logits = self.run_model(input_ids, positions, is_prefill)
            if self.rank == 0:
                # temperature=0 → greedy argmax (bypasses stochastic sampler).
                # Note: for prefill, ParallelLMHead already extracts only the
                # last-position logits per seq, so logits is [B, vocab_size].
                if all(seq.temperature <= 0 for seq in seqs):
                    token_ids = logits.argmax(dim=-1).tolist()
                else:
                    sampling_args = self.prepare_sample(seqs)
                    token_ids = self.sampler(logits, *sampling_args).tolist()
                if is_prefill and self.speculative_config is not None:
                    # Extend after sampling: the sampled token closes the
                    # shifted-token pairing for the final chunk (see below).
                    self._extend_prefill_aux(seqs, token_ids)
            else:
                token_ids = None
        reset_context()
        return token_ids

    def _extend_prefill_aux(
        self,
        seqs: list[Sequence],
        sampled: list[int],
        seqlen: dict[int, int] | None = None,
    ) -> None:
        """Catch the draft up on this prefill step's newly-produced target
        aux hidden states immediately (no cross-round stash) — one
        `proposer.extend()` call per prefill step. For a multi-chunk
        prefill, each chunk gets its own `extend()` call as soon as its
        aux hidden states are available; `extend`'s own
        `assert start == _draft_ctx_len` enforces that successive chunks
        (and finished-prefill's first decode round) stay contiguous, so no
        manual cross-call accumulation is needed here.

        EAGLE shifted-token pairing (vLLM eagle.py: input_ids shifted by
        one, positions/hidden_states not): the draft row at position p
        pairs token t_{p+1} with the target hidden f_p that predicted it.
        So a chunk covering positions [start, start+n) contributes rows
        (tokens[start+1 .. start+n], positions [start .. start+n-1], aux
        rows offset..offset+n-1); the final chunk's closing token
        t_{start+n} is this step's sampled token (passed in via
        `sampled`, which is why this runs after sampling).

        A chunk starting at position 0 (fresh prefill, or re-prefill after
        preemption reset num_cached_tokens) naturally satisfies `extend`'s
        continuity assert because preemption drops the seq's stale
        committed state (`drop_proposer_state` → `Proposer.drop`) before
        the seq is ever re-scheduled.

        For mixed prefill+decode batches (prepare_chunked), pass `seqlen`
        and skip non-prefill seqs — their length-1 step bypasses spec (see
        Scheduler.schedule_chunked's "spec not supported there" comment),
        so they simply aren't extended this round; their draft state picks
        up again next time this seq gets a prefill/verify round.
        """
        aux = self.model.model._aux_hidden_states
        if aux is None:
            return
        offset = 0
        ext_seqs, ext_tokens, ext_positions, ext_aux = [], [], [], []
        for seq, next_token in zip(seqs, sampled):
            n = seq.num_scheduled_tokens if seqlen is None else seqlen[seq.seq_id]
            if seqlen is not None and not seq.is_prefill:
                offset += n
                continue
            start = seq.num_cached_tokens
            # Shifted by one: the row at position p holds token t_{p+1}.
            tokens = seq.token_ids[start + 1:start + n + 1]
            if len(tokens) < n:    # final chunk: t_{start+n} is the sampled token
                tokens = tokens + [next_token]
            ext_seqs.append(seq)
            ext_tokens.append(tokens)
            ext_positions.append(list(range(start, start + n)))
            ext_aux.append(aux[offset:offset + n])
            offset += n
        if ext_seqs:
            self.proposer.extend(ext_seqs, ext_tokens, ext_positions, ext_aux)

    def prepare_chunked(self, seqs: list[Sequence], seqlen_this_time: dict[int, int]):
        """Build a single varlen batch mixing prefill chunks and decode tokens.

        Decode tokens are treated as length-1 prefill sequences. All K/V is
        read from paged cache via block_table, so flash_attn_varlen_func
        handles both types uniformly.
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        for seq in seqs:
            seqlen_q = seqlen_this_time[seq.seq_id]
            if seq.is_prefill:
                start = seq.num_cached_tokens
                end = start + seqlen_q
                seqlen_k = end
                input_ids.extend(seq[start:end])
                positions.extend(range(start, end))
                # slot mapping: traverse blocks covering [start, end)
                start_block = start // self.block_size
                end_block = (end + self.block_size - 1) // self.block_size
                for i in range(start_block, end_block):
                    slot_start = seq.block_table[i] * self.block_size
                    if i == start_block:
                        slot_start += start % self.block_size
                    if i != end_block - 1:
                        slot_end = seq.block_table[i] * self.block_size + self.block_size
                    else:
                        slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                    slot_mapping.extend(range(slot_start, slot_end))
            else:
                # Decode: 1 new token (last_token), K spans full history (includes last_token).
                seqlen_k = len(seq)
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

        block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # Always pass block_tables so attention reads K/V from paged cache.
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def run_chunked(self, seqs: list[Sequence], seqlen_this_time: dict[int, int]) -> list[int]:
        input_ids, positions = self.prepare_chunked(seqs, seqlen_this_time)
        logits = self.run_model(input_ids, positions, is_prefill=True)
        if self.rank == 0:
            if all(seq.temperature <= 0 for seq in seqs):
                token_ids = logits.argmax(dim=-1).tolist()
            else:
                sampling_args = self.prepare_sample(seqs)
                token_ids = self.sampler(logits, *sampling_args).tolist()
            if self.speculative_config is not None:
                # After sampling, same reason as run()'s prefill branch.
                self._extend_prefill_aux(seqs, token_ids, seqlen_this_time)
        else:
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_prefill_cudagraph(self):
        """Capture one-sequence prefill graphs grouped by total-token buckets."""
        config = self.config
        hf_config = config.hf_config
        max_tokens = min(config.max_num_batched_tokens, config.max_model_len)
        base_buckets = [256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
        self.prefill_graph_tokens = sorted({size for size in base_buckets if size < max_tokens} | {max_tokens})
        self.prefill_graphs = {}
        self.prefill_graph_vars = {}
        self.prefill_graph_pool = None

        # Capture largest first so smaller graphs can reuse the same pool.
        for bucket in reversed(self.prefill_graph_tokens):
            input_ids = torch.zeros(bucket, dtype=torch.int64)
            positions = torch.zeros(bucket, dtype=torch.int64)
            slot_mapping = torch.full((bucket,), -1, dtype=torch.int32)
            cu_seqlens = torch.tensor([0, bucket], dtype=torch.int32)
            outputs = torch.zeros(bucket, hf_config.hidden_size)
            graph = torch.cuda.CUDAGraph()

            # Captured prefill is a single causal sequence without a prefix cache.
            set_context(True, cu_seqlens, cu_seqlens, bucket, bucket, slot_mapping)
            outputs.copy_(self.model(input_ids, positions))  # warmup
            with torch.cuda.graph(graph, self.prefill_graph_pool):
                outputs.copy_(self.model(input_ids, positions))
            if self.prefill_graph_pool is None:
                self.prefill_graph_pool = graph.pool()
            self.prefill_graphs[bucket] = graph
            self.prefill_graph_vars[bucket] = dict(
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                outputs=outputs,
            )
            torch.cuda.synchronize()
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
