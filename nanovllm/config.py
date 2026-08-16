import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False
    prefill_chunk_size: int = 1024
    enable_prefill_cudagraph: bool = False
    kv_quant: bool = False
    speculative_config: dict | None = None
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.speculative_config is not None:
            # The EAGLE3 draft only lives on rank 0 (shares the target's
            # embedding, which would need TP collectives otherwise) — see
            # Eagle3DraftModel's docstring. Also: allocate_kv_cache() sizes
            # num_kvcache_blocks off rank 0's memory budget (which additionally
            # carries the draft's paged KV cache), so a TP>1 setup would give
            # different ranks different block counts for the *same* block ids.
            assert self.tensor_parallel_size == 1, "speculative decoding currently requires tensor_parallel_size == 1"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
