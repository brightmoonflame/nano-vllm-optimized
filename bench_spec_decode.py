"""Benchmark speculative decoding (EAGLE3) across scenarios.

Measures, for the canonical Llama-3.2-3B-Instruct target + Eagle3 draft
combo, five things:

  1. Correctness  — greedy (temperature=0) spec output equals non-spec output
                    token-for-token (rejection sampling preserves the target
                    distribution).
  2. Throughput   — spec vs non-spec decode throughput (tok/s) and speedup.
  3. Acceptance   — mean accepted draft tokens per spec round.
  4. K sweep      — speedup & acceptance vs num_spec_tokens (diminishing
                    returns as K grows).
  5. Batch sweep  — speedup vs batch size (spec decode is memory-bound, so
                    the gain is largest at small batch).

Default checkpoint paths point at the canonical pair; override with
--target-model / --draft-model. Both spec and non-spec run with
enforce_eager=True: nano-vllm's spec path is eager-only, so eager-vs-eager
isolates the draft mechanism from CUDA-graph launch-overhead savings.

Examples:
  python bench_spec_decode.py                     # correctness + single-point
  python bench_spec_decode.py --k-sweep           # + K sweep at batch=1
  python bench_spec_decode.py --batch-sweep       # + batch sweep at K=5
  python bench_spec_decode.py --k-sweep --batch-sweep
"""

import argparse
import gc
import os
import random
import time

import torch

from nanovllm import LLM, SamplingParams


def build_llm(target_model, draft_model, num_spec_tokens, gpu_memory_utilization):
    speculative_config = (
        {"model": draft_model, "num_spec_tokens": num_spec_tokens}
        if draft_model else None
    )
    return LLM(
        target_model,
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=gpu_memory_utilization,
        speculative_config=speculative_config,
    )


def free_llm(llm):
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def run_offline(llm, num_seqs, max_input_len, max_output_len, temperature, seed):
    """Queue `num_seqs` requests and drive step() to completion.

    Returns dict with decode-phase wall time, tokens, step count, throughput
    (tok/s) and mean accepted drafts per round (tokens/round - 1). Driving
    step() directly (rather than generate()) lets us count decode rounds,
    which is how acceptance rate is derived without touching engine internals.
    """
    rng = random.Random(seed)
    prompts = [
        [rng.randrange(10000) for _ in range(rng.randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=temperature, ignore_eos=True, max_tokens=max_output_len)
        for _ in range(num_seqs)
    ]

    # Warmup: triggers lazy init / torch.compile before the timed loop.
    llm.generate([[0] * 16], [SamplingParams(temperature=temperature, ignore_eos=True, max_tokens=1)], use_tqdm=False)

    for prompt, sp in zip(prompts, sampling_params):
        llm.add_request(prompt, sp)

    num_decode_steps = 0
    total_decode_tokens = 0
    t0 = time.perf_counter()
    while not llm.is_finished():
        _, num_tokens = llm.step()
        if num_tokens < 0:            # decode step (num_tokens = -new completion tokens)
            num_decode_steps += 1
            total_decode_tokens += -num_tokens
    elapsed = time.perf_counter() - t0

    throughput = total_decode_tokens / elapsed if elapsed > 0 else 0.0
    accept = (total_decode_tokens / num_decode_steps - 1.0) if num_decode_steps else 0.0
    return {
        "elapsed": elapsed,
        "tokens": total_decode_tokens,
        "steps": num_decode_steps,
        "throughput": throughput,
        "accept": accept,
    }


def check_correctness(target_model, draft_model, num_spec_tokens, max_input_len,
                      gpu_memory_utilization, seed):
    rng = random.Random(seed)
    prompt = [rng.randrange(10000) for _ in range(rng.randint(100, max_input_len))]
    sp = SamplingParams(temperature=0, ignore_eos=True, max_tokens=32)

    llm = build_llm(target_model, None, num_spec_tokens, gpu_memory_utilization)
    ref = llm.generate([prompt], [sp], use_tqdm=False)[0]["token_ids"]
    free_llm(llm)

    llm = build_llm(target_model, draft_model, num_spec_tokens, gpu_memory_utilization)
    out = llm.generate([prompt], [sp], use_tqdm=False)[0]["token_ids"]
    free_llm(llm)
    return ref == out, ref, out


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark EAGLE3 speculative decoding across scenarios.")
    p.add_argument("--target-model", default=os.path.expanduser("/root/model/Llama-3.2-3B-Instruct/"))
    p.add_argument("--draft-model", default=os.path.expanduser("/root/model/Llama-3.2-3B-Instruct-Eagle3/"))
    p.add_argument("--num-spec-tokens", type=int, default=5, help="Default K for single-point / batch sweep.")
    p.add_argument("--num-seqs", type=int, default=1, help="Batch size for single-point / K sweep.")
    p.add_argument("--max-input-len", type=int, default=512)
    p.add_argument("--max-output-len", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--correctness", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--k-sweep", action="store_true")
    p.add_argument("--k-list", default="1,3,5,7")
    p.add_argument("--batch-sweep", action="store_true")
    p.add_argument("--batch-list", default="1,2,4,8")
    return p.parse_args()


def main():
    args = parse_args()
    k_list = [int(x) for x in args.k_list.split(",")]
    batch_list = [int(x) for x in args.batch_list.split(",")]

    print("=" * 60)
    print("Speculative Decoding Benchmark (EAGLE3)")
    print(f"  target: {args.target_model}")
    print(f"  draft : {args.draft_model}")
    print(f"  gpu   : {torch.cuda.get_device_name(0)}")
    print(f"  input/output len: {args.max_input_len}/{args.max_output_len}, temperature={args.temperature}")
    print("=" * 60)

    if args.correctness:
        ok, ref, out = check_correctness(
            args.target_model, args.draft_model, args.num_spec_tokens,
            args.max_input_len, args.gpu_memory_utilization, args.seed,
        )
        print(f"\n[1] Correctness (greedy, temperature=0)")
        if ok:
            print(f"    spec == non-spec ✓ ({len(out)} tokens)")
        else:
            print(f"    MISMATCH! non-spec: {ref[:20]}")
            print(f"              spec    : {out[:20]}")

    # ---- Single-point: spec vs non-spec at --num-seqs, K = --num-spec-tokens ----
    print(f"\n[2] Single-point (batch={args.num_seqs}, K={args.num_spec_tokens})")
    llm = build_llm(args.target_model, None, args.num_spec_tokens, args.gpu_memory_utilization)
    base = run_offline(llm, args.num_seqs, args.max_input_len, args.max_output_len,
                       args.temperature, args.seed + args.num_seqs)
    free_llm(llm)

    llm = build_llm(args.target_model, args.draft_model, args.num_spec_tokens, args.gpu_memory_utilization)
    spec = run_offline(llm, args.num_seqs, args.max_input_len, args.max_output_len,
                       args.temperature, args.seed + args.num_seqs)
    free_llm(llm)

    print(f"    no-spec : {base['throughput']:8.1f} tok/s")
    print(f"    spec    : {spec['throughput']:8.1f} tok/s")
    print(f"    speedup : {spec['throughput'] / base['throughput']:5.2f}x")
    print(f"    accept  : {spec['accept']:5.2f} mean accepted drafts/round")

    # ---- K sweep ----
    if args.k_sweep:
        print(f"\n[3] K sweep (batch={args.num_seqs})")
        print(f"    {'K':>3}  {'speedup':>8}  {'accept':>7}  {'tok/s':>9}")
        for K in k_list:
            llm = build_llm(args.target_model, args.draft_model, K, args.gpu_memory_utilization)
            m = run_offline(llm, args.num_seqs, args.max_input_len, args.max_output_len,
                            args.temperature, args.seed + args.num_seqs)
            free_llm(llm)
            print(f"    {K:>3}  {m['throughput'] / base['throughput']:>7.2f}x  "
                  f"{m['accept']:>7.2f}  {m['throughput']:>9.1f}")

    # ---- Batch sweep ----
    if args.batch_sweep:
        print(f"\n[4] Batch sweep (K={args.num_spec_tokens})")
        print(f"    {'batch':>5}  {'no-spec tok/s':>14}  {'spec tok/s':>11}  {'speedup':>8}  {'accept':>7}")
        llm = build_llm(args.target_model, None, args.num_spec_tokens, args.gpu_memory_utilization)
        base_by_bs = {}
        for bs in batch_list:
            base_by_bs[bs] = run_offline(llm, bs, args.max_input_len, args.max_output_len,
                                         args.temperature, args.seed + bs)
        free_llm(llm)

        llm = build_llm(args.target_model, args.draft_model, args.num_spec_tokens, args.gpu_memory_utilization)
        for bs in batch_list:
            m = run_offline(llm, bs, args.max_input_len, args.max_output_len,
                            args.temperature, args.seed + bs)
            print(f"    {bs:>5}  {base_by_bs[bs]['throughput']:>14.1f}  "
                  f"{m['throughput']:>11.1f}  {m['throughput'] / base_by_bs[bs]['throughput']:>7.2f}x  "
                  f"{m['accept']:>7.2f}")
        free_llm(llm)

    print("\nDone.")


if __name__ == "__main__":
    main()
