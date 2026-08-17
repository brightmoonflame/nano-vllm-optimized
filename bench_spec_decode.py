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


# A short corpus of real English prose. Prompts are drawn from this text's
# token encoding rather than uniform random tokens: random token prompts
# carry no semantics, so the draft's accept rate swings wildly with the
# prompt's incidental structure, making speedup unreproducible. Real prose
# keeps accept rate — and therefore speedup — stable and meaningful.
REAL_TEXT = (
    "Machine learning is a branch of artificial intelligence that enables systems to learn from data "
    "and improve their performance on a task without being explicitly programmed for every case. "
    "A model is trained by adjusting its parameters so that its predictions match observed outcomes, "
    "using an objective function that measures the difference between the two. "
    "Deep learning builds on this idea by stacking many layers of simple computations, letting the "
    "network discover hierarchical representations of its input. Convolutional networks excel at "
    "image data, while recurrent and attention-based architectures dominate natural language. "
    "The transformer, introduced in 2017, replaced recurrence with self-attention, allowing every "
    "position in a sequence to attend to every other position in parallel. This made it practical to "
    "train much larger models on much more data, and it became the foundation of modern language models. "
    "A large language model is a transformer trained to predict the next token in a text sequence. "
    "During training it reads enormous corpora of books, articles, code, and conversation, learning "
    "patterns of grammar, reasoning, and world knowledge in the process. At inference time it generates "
    "text one token at a time, each step conditioned on all the tokens produced so far. "
    "The quality of generated text depends on the training data, the model size, and the decoding strategy. "
    "Greedy decoding always picks the single most likely next token, which is deterministic but can "
    "produce repetitive text. Sampling introduces randomness by drawing from the model's probability "
    "distribution, and temperature controls how sharply that distribution is peaked. "
    "Top-k and top-p filtering discard the long tail of unlikely tokens to keep output coherent. "
    "Language models are used for translation, summarization, question answering, code completion, and "
    "conversation. They also raise important questions about bias, factual accuracy, and responsible use, "
    "because a model reflects patterns present in its training data rather than verified truth. "
    "Careful evaluation and human oversight remain essential when these systems are deployed in practice. "
    "Speculative decoding is a technique that accelerates generation by using a small draft model to "
    "propose several candidate tokens at once, which a larger target model then verifies in a single "
    "forward pass. When the draft guesses correctly, several tokens are accepted in one step instead of "
    "one, reducing the number of sequential passes through the target model and thereby lowering latency. "
    "The acceptance rate, meaning how many of the draft's proposals the target agrees with, is the key "
    "factor determining how much speedup is achieved, and it depends strongly on how well the draft "
    "matches the target's predictions on the input at hand."
)


def make_real_prompt(llm: LLM, length: int, offset_seed: int) -> list[int]:
    """Return a `length`-token prompt drawn from REAL_TEXT.

    Encodes the corpus once, starts at a seed-determined offset, and wraps
    around (repeating the text) if `length` exceeds the remaining tokens.
    """
    rng = random.Random(offset_seed)
    base = llm.tokenizer.encode(REAL_TEXT)
    start = rng.randrange(len(base))
    tokens = base[start:] + base[:start]
    while len(tokens) < length:
        tokens.extend(base)
    return tokens[:length]


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
    (tok/s) and mean accepted drafts per sequence per round
    (tokens/round/batch - 1). Driving step() directly (rather than
    generate()) lets us count decode rounds, which is how acceptance rate is
    derived without touching engine internals.
    """
    rng = random.Random(seed)
    prompts = [
        make_real_prompt(llm, rng.randint(100, max_input_len), seed + i)
        for i in range(num_seqs)
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
    # Each decode round emits (accepted drafts + 1) tokens per seq, so the
    # per-seq mean accepted count is tokens/round/batch - 1.
    accept = (total_decode_tokens / num_decode_steps / num_seqs - 1.0) if num_decode_steps else 0.0
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
    sp = SamplingParams(temperature=0, ignore_eos=True, max_tokens=32)

    llm = build_llm(target_model, None, num_spec_tokens, gpu_memory_utilization)
    prompt = make_real_prompt(llm, rng.randint(100, max_input_len), seed)
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
