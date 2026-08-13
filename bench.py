import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 64
    max_input_len = 1024
    max_ouput_len = 1024

    target_path = os.path.expanduser("/root/model/Qwen3-1.7B/")
    draft_path = os.path.expanduser("/root/model/Qwen3-0.6B/")

    # Toggle chunked prefill: set to True to interleave prefill chunks with decode.
    enable_chunked_prefill = False
    # Toggle INT8 KV cache quantization (~48% memory reduction).
    kv_quant = False
    # Toggle token-bucketed CUDA Graph for single-sequence prefill.
    enable_prefill_cudagraph = False

    # Set draft_path=None to disable speculative decoding.
    speculative_config = {"model": draft_path, "num_spec_tokens": 5} if draft_path else None

    llm = LLM(target_path, enforce_eager=False, max_model_len=4096,
              enable_chunked_prefill=enable_chunked_prefill,
              enable_prefill_cudagraph=enable_prefill_cudagraph,
              kv_quant=kv_quant,
              gpu_memory_utilization=0.8,
              speculative_config=speculative_config)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # temperature=0 (greedy): spec decode is greedy-only, so this ensures
    # spec and non-spec produce identical output for correctness verification.
    sampling_params = [SamplingParams(temperature=0, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    spec_tag = f"spec(K={speculative_config['num_spec_tokens']})" if speculative_config else "no-spec"
    print(f"[{spec_tag}] Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
