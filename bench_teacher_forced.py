"""INT8 vs BF16 accuracy, measured teacher-forced (cascade-free).

The free-running greedy benchmark (`bench_accuracy.py`) is a *pessimistic*
metric: greedy decoding is autoregressive, so a single early argmax flip
changes the context for every later step, and the two sequences then keep
generating different (but individually valid) continuations that get scored
as mismatches. That cascade effect makes a near-lossless quantizer look bad.

This script removes the cascade by evaluating both models **teacher-forced on
the same reference continuation**: at every position the model is fed the
reference prefix and only has to predict the *next* token, so the contexts
never diverge. We then report:

  1. next-token agreement — how often INT8's argmax equals BF16's argmax
     (both forced along the same reference trajectory). This is the cascade-free
     analogue of token-match; a good INT8 should sit close to 100%.
  2. continuation perplexity — the mean next-token log-prob of the reference
     under each model (ΔPPL). This is the literature-standard number (KVQuant,
     vLLM) that free-running match rates cannot approximate.

Reference continuations are the BF16 model's own greedy outputs, so BF16 is
(on) its own greedy trajectory and its agreement is ~100% by construction —
making INT8's agreement a direct "does quantization change the answer" read.

Run:
    python bench_teacher_forced.py --model /root/model/Llama-3.2-3B-Instruct
"""
import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus

# Same prompt set as bench_accuracy.py so results are directly comparable.
PROMPTS = [
    "Explain the difference between a thread and a process.",
    "Write a Python function to reverse a linked list.",
    "What is the capital of France?",
    "Summarize the theory of relativity in two sentences.",
    "List three benefits of renewable energy.",
    "What is the Fibonacci sequence? Give the first 10 numbers.",
    "Explain how a hash table works.",
    "Write a haiku about winter.",
    "What is machine learning?",
    "Describe the water cycle.",
    "Name three sorting algorithms and their complexities.",
    "What causes the seasons on Earth?",
    "Explain the concept of recursion with an example.",
    "What is the difference between HTTP and HTTPS?",
    "Give a one-paragraph introduction to quantum computing.",
    "What is photosynthesis and why is it important?",
]


def free_run(model: str, prompts: list[str], max_tokens: int) -> list[list[int]]:
    """BF16 greedy generation — produces the reference continuations."""
    llm = LLM(model, kv_quant=False, use_triton_attn=True, enforce_eager=True)
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return [o["token_ids"] for o in outs]


def teacher_forced_eval(model: str, kv_quant: bool, prompts: list[str], continuations: list[list[int]]):
    """Teacher-force one model along each reference continuation.

    Returns (preds, sum_logprob, count):
      preds        — per-prompt list of argmax next-token ids, one per
                     continuation position (position 0 from prefill's last
                     logits, the rest from forced decode steps).
      sum_logprob  — sum of the reference token's next-token log-prob.
      count        — number of continuation positions scored.

    Cache bookkeeping (`num_cached_tokens`, `append_token`) is advanced by hand
    instead of `Scheduler.postprocess`, because postprocess would append the
    *sampled* token — teacher forcing needs the *reference* token appended.
    """
    llm = LLM(model, kv_quant=kv_quant, use_triton_attn=True, enforce_eager=True)
    engine, runner, sched = llm, llm.model_runner, llm.scheduler

    all_preds, total_lp, total_n = [], 0.0, 0
    for prompt, cont in zip(prompts, continuations):
        prompt_ids = engine.tokenizer.encode(prompt)
        seq = Sequence(prompt_ids)
        sched.add(seq)

        # --- Prefill the prompt; full per-position logits; position -1 predicts cont[0].
        seqs, _ = sched.schedule()
        logits = runner.call("run_teacher_forced", seqs, True).float()   # (P, V)
        logp = F.log_softmax(logits, dim=-1)
        preds = [int(logits[-1].argmax())]
        total_lp += float(logp[-1, cont[0]])
        total_n += 1
        # Advance cache bookkeeping without appending a sampled token.
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        # Prime the first forced decode input with the reference token.
        seq.append_token(cont[0])

        # --- Teacher-forced decode over the rest of the continuation.
        for i in range(1, len(cont)):
            seqs, _ = sched.schedule()
            logits = runner.call("run_teacher_forced", seqs, False).float()  # (1, V)
            logp = F.log_softmax(logits[0], dim=-1)
            preds.append(int(logits[0].argmax()))
            total_lp += float(logp[cont[i]])
            total_n += 1
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            seq.append_token(cont[i])   # teacher forcing: feed the reference, not the argmax
        all_preds.append(preds)

        # Free this seq's blocks and drop it from `running`, so the next
        # prompt's schedule()/decode batch isn't polluted by leftover seqs.
        seq.status = SequenceStatus.FINISHED
        sched.block_manager.deallocate(seq)
        sched.running.remove(seq)

    llm.exit()
    # `engine`/`runner`/`sched` hold references to the ModelRunner (and its
    # multi-GB KV cache). They must be released BEFORE gc.collect()/empty_cache,
    # or the KV cache stays allocated and the NEXT LLM instance's
    # allocate_kv_cache() sees no free VRAM (num_kvcache_blocks <= 0).
    del llm, engine, runner, sched
    gc.collect()
    torch.cuda.empty_cache()
    return all_preds, total_lp, total_n


def main():
    p = argparse.ArgumentParser(description="Teacher-forced INT8 vs BF16 accuracy.")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()
    model = str(Path(args.model).expanduser().resolve())

    print(">> free-run (BF16 greedy) to build reference continuations ...")
    refs = free_run(model, PROMPTS, args.max_tokens)
    print(">> teacher-forced eval: BF16 ...")
    bf16_preds, bf16_lp, n = teacher_forced_eval(model, False, PROMPTS, refs)
    print(">> teacher-forced eval: INT8 ...")
    int8_preds, int8_lp, _ = teacher_forced_eval(model, True, PROMPTS, refs)

    agree = sum(x == y for a, b in zip(bf16_preds, int8_preds) for x, y in zip(a, b))
    # Sanity: BF16 teacher-forced along its own greedy output should reproduce
    # the reference argmax at ~100%. A low number here means the driver is buggy,
    # not that quantization is bad.
    self_check = sum(p == r for a, b in zip(bf16_preds, refs) for p, r in zip(a, b))
    bf16_ppl = math.exp(-bf16_lp / n)
    int8_ppl = math.exp(-int8_lp / n)

    print("=" * 60)
    print(f"device: {torch.cuda.get_device_name(0)}   prompts={len(PROMPTS)}  cont tokens={n}")
    print("=" * 60)
    print(f"[sanity] BF16 vs its own reference:  {self_check}/{n} ({100 * self_check / n:.2f}%)")
    print(f"teacher-forced agreement INT8 vs BF16: {agree}/{n} ({100 * agree / n:.2f}%)")
    print(f"continuation PPL   BF16: {bf16_ppl:.3f}   INT8: {int8_ppl:.3f}   Δ: {int8_ppl - bf16_ppl:+.3f}")


if __name__ == "__main__":
    main()
