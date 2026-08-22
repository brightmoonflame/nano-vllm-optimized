"""Teacher-forced INT8 KV-cache quality benchmark.

The default mode follows BF16 greedy continuations, which is a fast,
cascade-free *fidelity* check: both engines receive exactly the same history,
and every continuation token is evaluated through the paged Decode path.

Pass --corpus-file for the quality metric that should be reported externally:
real corpus tokens are split into fixed prefix/continuation windows and scored
teacher-forced. This reports standard NLL/PPL degradation, while still reading
the quantized KV cache on every continuation Decode step.

Both modes also report BF16-top-K logits MAE/RMSE (a raw numerical diagnostic),
top-K log-prob MAE/RMSE, and a top-K-plus-other-bucket KL divergence. Log-prob
error and KL are probability-aware; raw logits error alone is not a quality
metric because an additive logit shift leaves softmax unchanged.

Run:
  # Fast BF16-reference fidelity check
  python bench_teacher_forced.py --model /root/model/Llama-3.2-3B-Instruct

  # Standard corpus PPL (for example, WikiText-2 test text)
  python bench_teacher_forced.py --model /root/model/Llama-3.2-3B-Instruct \
      --corpus-file /path/to/wiki.test.raw --prompt-tokens 256 --eval-tokens 256
"""
import argparse
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus

# Same prompt set as bench_accuracy.py so the default fidelity result remains
# directly comparable to the free-running greedy benchmark.
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


@dataclass
class LogitTracker:
    """Compact BF16/INT8 distribution comparison without full-vocab storage.

    BF16 stores its top-K logits/ids plus logsumexp. INT8 gathers the same ids
    and computes a K-token + other-mass KL. This is exact for retained
    categories and uses O(num_positions * K) CPU memory.
    """

    topk: int
    ids: list[torch.Tensor]
    values: list[torch.Tensor]
    logsumexp: list[float]
    position: int = 0
    raw_abs_sum: float = 0.0
    raw_sq_sum: float = 0.0
    logprob_abs_sum: float = 0.0
    logprob_sq_sum: float = 0.0
    element_count: int = 0
    kl_sum: float = 0.0

    @classmethod
    def create(cls, topk: int) -> Optional["LogitTracker"]:
        return cls(topk, [], [], []) if topk > 0 else None

    def add_reference(self, logits: torch.Tensor) -> None:
        values, ids = torch.topk(logits, self.topk)
        self.ids.append(ids.cpu())
        self.values.append(values.cpu())
        self.logsumexp.append(float(torch.logsumexp(logits, dim=-1)))

    def add_quantized(self, logits: torch.Tensor) -> None:
        assert self.position < len(self.ids), "BF16/INT8 teacher-forced positions differ"
        ids = self.ids[self.position].to(logits.device)
        bf_values = self.values[self.position].to(logits.device)
        bf_lse = self.logsumexp[self.position]
        int_values = logits.gather(0, ids)
        int_lse = torch.logsumexp(logits, dim=-1)

        raw_diff = int_values - bf_values
        bf_logprob = bf_values - bf_lse
        int_logprob = int_values - int_lse
        logprob_diff = int_logprob - bf_logprob
        self.raw_abs_sum += float(raw_diff.abs().sum())
        self.raw_sq_sum += float(raw_diff.square().sum())
        self.logprob_abs_sum += float(logprob_diff.abs().sum())
        self.logprob_sq_sum += float(logprob_diff.square().sum())
        self.element_count += self.topk

        # Aggregate all non-top-K tokens into one category. This is not a full
        # vocabulary KL, but is probability-aware and exact for BF16's top-K.
        bf_p = bf_logprob.exp()
        int_p = int_logprob.exp()
        bf_other = (1.0 - bf_p.sum()).clamp_min(1e-12)
        int_other = (1.0 - int_p.sum()).clamp_min(1e-12)
        kl = torch.sum(bf_p * (bf_logprob - int_logprob))
        kl += bf_other * (bf_other.log() - int_other.log())
        self.kl_sum += float(kl)
        self.position += 1

    def report(self) -> dict:
        assert self.position == len(self.ids), "INT8 did not score every BF16 position"
        count = max(self.element_count, 1)
        positions = max(self.position, 1)
        return {
            "raw_mae": self.raw_abs_sum / count,
            "raw_rmse": math.sqrt(self.raw_sq_sum / count),
            "logprob_mae": self.logprob_abs_sum / count,
            "logprob_rmse": math.sqrt(self.logprob_sq_sum / count),
            "topk_other_kl": self.kl_sum / positions,
        }


def free_run(model: str, prompts: list[str], max_tokens: int,
             use_triton_attn: bool) -> list[list[int]]:
    """BF16 greedy generation — produces reference continuations."""
    llm = LLM(model, kv_quant=False, use_triton_attn=use_triton_attn, enforce_eager=True)
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return [o["token_ids"] for o in outs]


def corpus_windows(model: str, corpus_file: Path, prompt_tokens: int,
                   eval_tokens: int, max_samples: int) -> tuple[list[list[int]], list[list[int]]]:
    """Split real corpus tokens into independent prefix/continuation windows."""
    text = corpus_file.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(model)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    window = prompt_tokens + eval_tokens
    prompts, continuations = [], []
    for start in range(0, len(token_ids) - window + 1, window):
        prompts.append(token_ids[start:start + prompt_tokens])
        continuations.append(token_ids[start + prompt_tokens:start + window])
        if len(prompts) == max_samples:
            break
    if not prompts:
        raise ValueError(
            f"corpus needs at least {window} tokens; got {len(token_ids)} from {corpus_file}")
    return prompts, continuations


def teacher_forced_eval(model: str, kv_quant: bool, prompts: List[Union[str, List[int]]],
                         continuations: List[List[int]], use_triton_attn: bool,
                         tracker: Optional[LogitTracker] = None,
                         collect_reference_logits: bool = False):
    """Score continuations with cached Prefill followed by token-by-token Decode.

    Scheduler.postprocess appends a sampled token; teacher forcing instead
    appends the known target. Thus every continuation token after the first
    actually reads the BF16 or INT8 paged KV cache.
    """
    llm = LLM(model, kv_quant=kv_quant, use_triton_attn=use_triton_attn, enforce_eager=True)
    engine, runner, sched = llm, llm.model_runner, llm.scheduler

    all_preds, total_lp, total_n = [], 0.0, 0
    for prompt, cont in zip(prompts, continuations):
        if not cont:
            continue
        prompt_ids = prompt if isinstance(prompt, list) else engine.tokenizer.encode(prompt)
        seq = Sequence(prompt_ids)
        sched.add(seq)

        def score(logits_1d: torch.Tensor, target: int) -> int:
            nonlocal total_lp, total_n
            logp = F.log_softmax(logits_1d, dim=-1)
            pred = int(logits_1d.argmax())
            total_lp += float(logp[target])
            total_n += 1
            if tracker is not None:
                if collect_reference_logits:
                    tracker.add_reference(logits_1d)
                else:
                    tracker.add_quantized(logits_1d)
            return pred

        # Prefill writes prompt K/V; final logits predict cont[0].
        seqs, _ = sched.schedule()
        logits = runner.call("run_teacher_forced", seqs, True).float()
        preds = [score(logits[-1], cont[0])]
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        seq.append_token(cont[0])

        # Every subsequent target uses the Decode cache-read path.
        for target in cont[1:]:
            seqs, _ = sched.schedule()
            logits = runner.call("run_teacher_forced", seqs, False).float()
            preds.append(score(logits[0], target))
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            seq.append_token(target)
        all_preds.append(preds)

        seq.status = SequenceStatus.FINISHED
        sched.block_manager.deallocate(seq)
        sched.running.remove(seq)

    llm.exit()
    # ModelRunner owns a multi-GB KV cache. Drop every alias before releasing
    # CUDA memory so the second BF16/INT8 pass can allocate its own cache.
    del llm, engine, runner, sched
    gc.collect()
    torch.cuda.empty_cache()
    return all_preds, total_lp, total_n


def main() -> None:
    p = argparse.ArgumentParser(description="Teacher-forced INT8 KV-cache fidelity and corpus-PPL benchmark.")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=128,
                   help="BF16 greedy continuation length in default fidelity mode")
    p.add_argument("--corpus-file", type=Path,
                   help="UTF-8 real-text corpus; enables standard corpus NLL/PPL mode")
    p.add_argument("--prompt-tokens", type=int, default=256,
                   help="prefill prefix length per corpus window")
    p.add_argument("--eval-tokens", type=int, default=256,
                   help="teacher-forced continuation tokens per corpus window")
    p.add_argument("--max-samples", type=int, default=8,
                   help="maximum independent corpus windows")
    p.add_argument("--logits-topk", type=int, default=32,
                   help="BF16 top-K logits used for MAE/RMSE and top-K+other KL; 0 disables")
    p.add_argument("--use-triton-attn", action=argparse.BooleanOptionalAction, default=True,
                   help="use Triton attention (default: true); pass --no-use-triton-attn for flash-attn fallback")
    args = p.parse_args()
    if args.max_tokens <= 0 or args.prompt_tokens <= 0 or args.eval_tokens <= 0 or args.max_samples <= 0:
        p.error("token and sample counts must be positive")
    if args.logits_topk < 0:
        p.error("--logits-topk must be non-negative")
    model = str(Path(args.model).expanduser().resolve())

    corpus_mode = args.corpus_file is not None
    if corpus_mode:
        corpus_file = args.corpus_file.expanduser().resolve()
        if not corpus_file.is_file():
            p.error(f"corpus file does not exist: {corpus_file}")
        prompts, refs = corpus_windows(model, corpus_file, args.prompt_tokens, args.eval_tokens, args.max_samples)
        mode_label = f"real corpus: {corpus_file.name}  windows={len(prompts)} x {args.prompt_tokens}+{args.eval_tokens} tokens"
    else:
        print(">> free-run (BF16 greedy) to build reference continuations ...")
        prompts = PROMPTS
        refs = free_run(model, prompts, args.max_tokens, args.use_triton_attn)
        mode_label = f"BF16-generated reference continuations  prompts={len(prompts)}"

    tracker = LogitTracker.create(args.logits_topk)
    print(">> teacher-forced eval: BF16 ...")
    bf16_preds, bf16_lp, n = teacher_forced_eval(
        model, False, prompts, refs, args.use_triton_attn,
        tracker=tracker, collect_reference_logits=True)
    print(">> teacher-forced eval: INT8 ...")
    int8_preds, int8_lp, _ = teacher_forced_eval(
        model, True, prompts, refs, args.use_triton_attn,
        tracker=tracker, collect_reference_logits=False)

    agree = sum(x == y for a, b in zip(bf16_preds, int8_preds) for x, y in zip(a, b))
    bf16_nll = -bf16_lp / n
    int8_nll = -int8_lp / n
    bf16_ppl = math.exp(bf16_nll)
    int8_ppl = math.exp(int8_nll)

    print("=" * 72)
    backend = "Triton" if args.use_triton_attn else "flash-attn fallback"
    print(f"device: {torch.cuda.get_device_name(0)}   backend={backend}")
    print(f"evaluation: {mode_label}   scored tokens={n}")
    print("=" * 72)
    print(f"teacher-forced agreement INT8 vs BF16: {agree}/{n} ({100 * agree / n:.2f}%)")
    ppl_label = "corpus PPL" if corpus_mode else "reference-continuation PPL (fidelity only)"
    print(f"{ppl_label}  BF16: {bf16_ppl:.6f}  INT8: {int8_ppl:.6f}  "
          f"Delta: {int8_ppl - bf16_ppl:+.6f} ({(int8_ppl / bf16_ppl - 1) * 100:+.4f}%)")
    print(f"NLL / token  BF16: {bf16_nll:.8f}  INT8: {int8_nll:.8f}  Delta: {int8_nll - bf16_nll:+.8f}")
    if tracker is not None:
        stats = tracker.report()
        print(f"BF16 top-{args.logits_topk} raw-logit   MAE: {stats['raw_mae']:.6f}  RMSE: {stats['raw_rmse']:.6f}")
        print(f"BF16 top-{args.logits_topk} log-prob    MAE: {stats['logprob_mae']:.6f}  RMSE: {stats['logprob_rmse']:.6f}")
        print(f"BF16 top-{args.logits_topk}+other KL(BF16 || INT8): {stats['topk_other_kl']:.8f}")


if __name__ == "__main__":
    main()
