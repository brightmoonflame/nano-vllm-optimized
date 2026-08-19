"""Quantization vs BF16 accuracy, measured teacher-forced (cascade-free).

The free-running greedy benchmark (`bench_accuracy.py`) is a *pessimistic*
metric: greedy decoding is autoregressive, so a single early argmax flip
changes the context for every later step, and the two sequences then keep
generating different (but individually valid) continuations that get scored
as mismatches. That cascade effect makes a near-lossless quantizer look bad.

This script removes the cascade by evaluating both models **teacher-forced on
the same reference continuation**: at every position the model is fed the
reference prefix and only has to predict the *next* token, so the contexts
never diverge. We then report:

  1. next-token agreement — how often the quantized model's argmax equals
     BF16's argmax (both forced along the same reference trajectory). This is
     the cascade-free analogue of token-match.
  2. continuation perplexity — the mean next-token log-prob of the reference
     under each model (ΔPPL). This is the literature-standard number (KVQuant,
     vLLM) that free-running match rates cannot approximate.
  3. BF16 top-k logits MAE/RMSE — a compact, cascade-free measure of the
     numerical change in the candidates BF16 considered most likely. Storing
     only top-k avoids retaining multi-GB full-vocabulary logits on the CPU.

Reference continuations are the BF16 model's own greedy outputs, so BF16 is
(on) its own greedy trajectory and its agreement is ~100% by construction —
making INT8's agreement a direct "does quantization change the answer" read.

Run:
    # Existing KV INT8 evaluation (default)
    python bench_teacher_forced.py --model /root/model/Llama-3.2-3B-Instruct

    # Online INT8 W8A16 Linear evaluation
    python bench_teacher_forced.py --model /root/model/Llama-3.2-3B-Instruct \
        --quantization int8_w8a16
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


def teacher_forced_eval(
    model: str,
    kv_quant: bool,
    weight_quant: str | None,
    prompts: list[str],
    continuations: list[list[int]],
    logits_topk: int,
    logit_reference: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
):
    """Teacher-force one model along each reference continuation.

    Returns (preds, sum_logprob, count, logit_reference, error_stats):
      preds        — per-prompt list of argmax next-token ids, one per
                     continuation position (position 0 from prefill's last
                     logits, the rest from forced decode steps).
      sum_logprob  — sum of the reference token's next-token log-prob.
      count        — number of continuation positions scored.
      logit_reference — BF16 top-k (ids, values) for every scored position;
                        returned only when ``logit_reference`` is None.
      error_stats  — (absolute-error sum, squared-error sum, value count) on
                     the BF16 top-k logits, or None for the BF16 pass.

    Cache bookkeeping (`num_cached_tokens`, `append_token`) is advanced by hand
    instead of `Scheduler.postprocess`, because postprocess would append the
    *sampled* token — teacher forcing needs the *reference* token appended.
    """
    llm = LLM(
        model,
        kv_quant=kv_quant,
        weight_quant=weight_quant,
        use_triton_attn=True,
        enforce_eager=True,
    )
    engine, runner, sched = llm, llm.model_runner, llm.scheduler

    all_preds, total_lp, total_n = [], 0.0, 0
    captured_logits = [] if logit_reference is None else None
    abs_error_sum = sq_error_sum = 0.0
    error_count = 0
    logit_position = 0

    def score(logit_row: torch.Tensor, target_id: int) -> tuple[int, float]:
        """Score one forced next-token position and compare/capture top-k."""
        nonlocal abs_error_sum, sq_error_sum, error_count, logit_position
        if logit_reference is None:
            values, ids = torch.topk(logit_row, logits_topk)
            captured_logits.append((ids.cpu(), values.float().cpu()))
        else:
            ids, values = logit_reference[logit_position]
            probe = logit_row.index_select(0, ids.to(logit_row.device)).float().cpu()
            delta = probe - values
            abs_error_sum += float(delta.abs().sum())
            sq_error_sum += float(delta.square().sum())
            error_count += delta.numel()
        logit_position += 1
        logp = F.log_softmax(logit_row, dim=-1)
        return int(logit_row.argmax()), float(logp[target_id])

    for prompt, cont in zip(prompts, continuations):
        prompt_ids = engine.tokenizer.encode(prompt)
        seq = Sequence(prompt_ids)
        sched.add(seq)

        # --- Prefill the prompt; full per-position logits; position -1 predicts cont[0].
        seqs, _ = sched.schedule()
        logits = runner.call("run_teacher_forced", seqs, True).float()   # (P, V)
        pred, lp = score(logits[-1], cont[0])
        preds = [pred]
        total_lp += lp
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
            pred, lp = score(logits[0], cont[i])
            preds.append(pred)
            total_lp += lp
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
    error_stats = None if logit_reference is None else (abs_error_sum, sq_error_sum, error_count)
    return all_preds, total_lp, total_n, captured_logits, error_stats


def main():
    p = argparse.ArgumentParser(description="Teacher-forced quantization vs BF16 accuracy.")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument(
        "--quantization",
        choices=("kv_int8", "int8_w8a16"),
        default="kv_int8",
        help="quantized path to compare with BF16 (default: kv_int8, preserving old behavior)",
    )
    p.add_argument("--logits-topk", type=int, default=32)
    args = p.parse_args()
    model = str(Path(args.model).expanduser().resolve())

    print(">> free-run (BF16 greedy) to build reference continuations ...")
    refs = free_run(model, PROMPTS, args.max_tokens)
    assert args.logits_topk > 0
    print(">> teacher-forced eval: BF16 ...")
    bf16_preds, bf16_lp, n, logit_reference, _ = teacher_forced_eval(
        model, False, None, PROMPTS, refs, args.logits_topk,
    )
    kv_quant = args.quantization == "kv_int8"
    weight_quant = "int8_w8a16" if args.quantization == "int8_w8a16" else None
    print(f">> teacher-forced eval: {args.quantization} ...")
    int8_preds, int8_lp, _, _, error_stats = teacher_forced_eval(
        model, kv_quant, weight_quant, PROMPTS, refs, args.logits_topk, logit_reference,
    )

    agree = sum(x == y for a, b in zip(bf16_preds, int8_preds) for x, y in zip(a, b))
    # Sanity: BF16 teacher-forced along its own greedy output should reproduce
    # the reference argmax at ~100%. A low number here means the driver is buggy,
    # not that quantization is bad.
    self_check = sum(p == r for a, b in zip(bf16_preds, refs) for p, r in zip(a, b))
    bf16_ppl = math.exp(-bf16_lp / n)
    int8_ppl = math.exp(-int8_lp / n)
    abs_error_sum, sq_error_sum, error_count = error_stats
    logit_mae = abs_error_sum / error_count
    logit_rmse = math.sqrt(sq_error_sum / error_count)

    print("=" * 60)
    print(f"device: {torch.cuda.get_device_name(0)}   prompts={len(PROMPTS)}  cont tokens={n}")
    print("=" * 60)
    print(f"[sanity] BF16 vs its own reference:  {self_check}/{n} ({100 * self_check / n:.2f}%)")
    print(f"teacher-forced agreement {args.quantization} vs BF16: {agree}/{n} ({100 * agree / n:.2f}%)")
    print(f"continuation PPL   BF16: {bf16_ppl:.3f}   {args.quantization}: {int8_ppl:.3f} "
          f"Δ: {int8_ppl - bf16_ppl:+.3f} ({(int8_ppl / bf16_ppl - 1) * 100:+.3f}%)")
    print(f"BF16 top-{args.logits_topk} logits  MAE: {logit_mae:.6f}   RMSE: {logit_rmse:.6f}")


if __name__ == "__main__":
    main()
