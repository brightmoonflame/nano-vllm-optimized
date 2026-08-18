"""INT8 vs BF16 generation accuracy: greedy token-match rate.

Runs the same prompts through kv_quant=True and kv_quant=False (both on the
self-researched Triton kernels) and reports how faithfully the INT8 path
reproduces the BF16 greedy output — the "does quantization change the answer"
number.

Run:
    python bench_accuracy.py --model /root/model/Llama-3.2-3B-Instruct
"""
import argparse
import gc
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams

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


def generate(model: str, kv_quant: bool, prompts: list[str], max_tokens: int) -> list[list[int]]:
    llm = LLM(model, kv_quant=kv_quant, use_triton_attn=True, enforce_eager=True)
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)   # greedy → deterministic
    outs = llm.generate(prompts, sp, use_tqdm=False)
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return [o["token_ids"] for o in outs]


def main():
    p = argparse.ArgumentParser(description="INT8 vs BF16 greedy generation accuracy.")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()
    model = str(Path(args.model).expanduser().resolve())

    bf16 = generate(model, kv_quant=False, prompts=PROMPTS, max_tokens=args.max_tokens)
    int8 = generate(model, kv_quant=True, prompts=PROMPTS, max_tokens=args.max_tokens)

    exact = 0
    matched = 0
    total = 0
    for a, b in zip(bf16, int8):
        if a == b:
            exact += 1
        matched += sum(x == y for x, y in zip(a, b))
        total += len(a)

    print("=" * 56)
    print(f"device: {torch.cuda.get_device_name(0)}   prompts={len(PROMPTS)}  max_tokens={args.max_tokens}")
    print("=" * 56)
    print(f"exact-match rate:  {exact}/{len(PROMPTS)} ({100 * exact / len(PROMPTS):.1f}%)")
    print(f"token-match rate:  {matched}/{total} ({100 * matched / total:.2f}%)")


if __name__ == "__main__":
    main()
