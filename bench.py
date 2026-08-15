import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 16
    max_input_len = 512
    max_ouput_len = 512

    # EAGLE3 spec decode requires a Llama target (only LlamaModel.forward has
    # aux_layer_ids support) and a matching EAGLE3 draft checkpoint. The
    # canonical combo is meta-llama/Llama-3.2-3B-Instruct (target) +
    # thoughtworks/Llama-3.2-3B-Instruct-Eagle3 (draft).
    target_path = os.path.expanduser("/root/model/Llama-3.2-3B-Instruct/")
    draft_path = os.path.expanduser("/root/model/Llama-3.2-3B-Instruct-Eagle3/")

    # Toggle chunked prefill: set to True to interleave prefill chunks with decode.
    enable_chunked_prefill = False
    # Toggle INT8 KV cache quantization (~48% memory reduction).
    kv_quant = False
    # Toggle token-bucketed CUDA Graph for single-sequence prefill.
    enable_prefill_cudagraph = False

    # Set draft_path=None to disable speculative decoding.
    speculative_config = {"model": draft_path, "num_spec_tokens": 5} if draft_path else None

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # temperature=0 (greedy): spec decode is greedy-only, so this ensures
    # spec and non-spec produce identical output for correctness verification.
    sampling_params = [SamplingParams(temperature=0, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    # --- Correctness check: spec vs non-spec output must match under greedy ---
    # (EAGLE3 rejection sampling preserves the target distribution; greedy mode
    # reduces that to argmax, so spec and non-spec outputs are token-for-token
    # identical if the implementation is correct.)
    check_prompts = [prompt_token_ids[0]]
    check_sp = [SamplingParams(temperature=0, ignore_eos=True, max_tokens=32)]

    llm_no_spec = LLM(target_path, enforce_eager=True, max_model_len=4096,
                      enable_chunked_prefill=enable_chunked_prefill,
                      enable_prefill_cudagraph=enable_prefill_cudagraph,
                      kv_quant=kv_quant,
                      gpu_memory_utilization=0.7,
                      speculative_config=None)
    ref_out = llm_no_spec.generate(check_prompts, check_sp, use_tqdm=False)[0]["token_ids"]
    llm_no_spec.exit()
    del llm_no_spec
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    llm = LLM(target_path, enforce_eager=True, max_model_len=4096,
              enable_chunked_prefill=enable_chunked_prefill,
              enable_prefill_cudagraph=enable_prefill_cudagraph,
              kv_quant=kv_quant,
              gpu_memory_utilization=0.7,
              speculative_config=speculative_config)
    spec_out = llm.generate(check_prompts, check_sp, use_tqdm=False)[0]["token_ids"]

    assert ref_out == spec_out, (
        f"spec/non-spec mismatch!\n"
        f"  non-spec: {ref_out[:20]}\n"
        f"  spec    : {spec_out[:20]}"
    )
    print(f"[correctness] spec == non-spec ✓ ({len(spec_out)} tokens)")
    # ----------------------------------------------------------------------

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    spec_time = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    spec_throughput = total_tokens / spec_time
    print(f"[spec(K={speculative_config['num_spec_tokens']})] Total: {total_tokens}tok, Time: {spec_time:.2f}s, Throughput: {spec_throughput:.2f}tok/s")
    llm.exit()
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    # --- Throughput comparison: same batch, spec off ---
    llm_no_spec = LLM(target_path, enforce_eager=True, max_model_len=4096,
                      enable_chunked_prefill=enable_chunked_prefill,
                      enable_prefill_cudagraph=enable_prefill_cudagraph,
                      kv_quant=kv_quant,
                      gpu_memory_utilization=0.7,
                      speculative_config=None)
    llm_no_spec.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm_no_spec.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    no_spec_time = time.time() - t
    no_spec_throughput = total_tokens / no_spec_time
    print(f"[no-spec            ] Total: {total_tokens}tok, Time: {no_spec_time:.2f}s, Throughput: {no_spec_throughput:.2f}tok/s")
    print(f"[speedup] {spec_throughput / no_spec_throughput:.2f}x")
    llm_no_spec.exit()


if __name__ == "__main__":
    main()
