"""End-to-end correctness of the self-researched Triton kernels on the
chunked-prefill and speculative-decode paths.

For each scenario, greedy output (temperature=0) is compared between
`use_triton_attn=True` (self-researched kernels) and `use_triton_attn=False`
(flash_attn baseline). Both must be token-identical:

  - chunked prefill: prompts of very different lengths force chunk splitting
    and mixed prefill-chunk/decode batches, exercising `run_chunked`'s split
    (paged prefill kernel for chunks, single-query kernel for decode).
  - speculative decode: draft-propose-then-verify routes the verification
    chunk through the paged prefill kernel.

Rejection sampling preserves the target distribution (greedy: token-for-token),
and the Triton kernels are drop-in replacements, so any mismatch is a bug in
the self-researched path — not an expected numerical difference.

Run:
  python bench_triton_e2e.py --model /root/model/Llama-3.2-3B-Instruct \
      --draft-model /root/model/Llama-3.2-3B-Instruct-Eagle3
"""
import argparse
import gc
import os
import random

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

from bench_spec_decode import REAL_TEXT


def free_llm(llm):
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def make_prompt(base_tokens: list[int], length: int, seed: int) -> list[int]:
    """Return a `length`-token prompt drawn from `base_tokens`, wrapping around."""
    rng = random.Random(seed)
    start = rng.randrange(len(base_tokens))
    tokens = base_tokens[start:] + base_tokens[:start]
    while len(tokens) < length:
        tokens.extend(base_tokens)
    return tokens[:length]


def run_once(model, prompts, sp, use_triton, **kw):
    llm = LLM(model, enforce_eager=True, use_triton_attn=use_triton, **kw)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    free_llm(llm)
    return [o["token_ids"] for o in outs]


def test_chunked(model, max_tokens):
    """chunked prefill ON, mixed-length prompts → mixed prefill/decode batches."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    base = tokenizer.encode(REAL_TEXT)
    lengths = [64, 400, 1024, 2048]     # divergent lengths force interleaving
    prompts = [make_prompt(base, L, i) for i, L in enumerate(lengths)]
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    print(f"\n[chunked prefill] 4 prompts (lens {lengths}), chunk_size=256, max_tokens={max_tokens}")
    ref = run_once(model, prompts, sp, use_triton=False,
                   enable_chunked_prefill=True, prefill_chunk_size=256)
    out = run_once(model, prompts, sp, use_triton=True,
                   enable_chunked_prefill=True, prefill_chunk_size=256)
    ok = all(a == b for a, b in zip(ref, out))
    print(f"  baseline vs triton: {'PASS (token-identical)' if ok else 'MISMATCH'}")
    if not ok:
        for i, (a, b) in enumerate(zip(ref, out)):
            if a != b:
                d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
                print(f"    seq {i}: first divergence at token {d}")
                break
    return ok


def test_spec(model, draft_model, max_tokens):
    """speculative decode ON → verify chunk routes through paged prefill kernel."""
    if draft_model is None or not os.path.isdir(os.path.expanduser(draft_model)):
        print(f"\n[speculative decode] SKIPPED (draft model not found: {draft_model})")
        return None

    tokenizer = AutoTokenizer.from_pretrained(model)
    base = tokenizer.encode(REAL_TEXT)
    num_seqs = 4
    rng = random.Random(0)
    prompts = [make_prompt(base, rng.randint(100, 512), i) for i in range(num_seqs)]
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    spec_cfg = {"model": os.path.expanduser(draft_model), "num_spec_tokens": 5}

    print(f"\n[speculative decode] {num_seqs} prompts, K=5, max_tokens={max_tokens}")
    ref = run_once(model, prompts, sp, use_triton=False, speculative_config=spec_cfg)
    out = run_once(model, prompts, sp, use_triton=True, speculative_config=spec_cfg)
    ok = all(a == b for a, b in zip(ref, out))
    print(f"  baseline vs triton: {'PASS (token-identical)' if ok else 'MISMATCH'}")
    if not ok:
        for i, (a, b) in enumerate(zip(ref, out)):
            if a != b:
                d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
                print(f"    seq {i}: first divergence at token {d}")
                break
    return ok


def main():
    p = argparse.ArgumentParser(description="Triton kernels end-to-end correctness (chunked + spec).")
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", default=None, help="EAGLE3 draft checkpoint (spec test).")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--skip-chunked", action="store_true")
    p.add_argument("--skip-spec", action="store_true")
    args = p.parse_args()

    print("=" * 60)
    print(f"Triton end-to-end correctness  (device: {torch.cuda.get_device_name(0)})")
    print("=" * 60)

    results = {}
    if not args.skip_chunked:
        results["chunked prefill"] = test_chunked(args.model, args.max_tokens)
    if not args.skip_spec:
        results["speculative decode"] = test_spec(args.model, args.draft_model, args.max_tokens)

    print("\n" + "=" * 60)
    failed = False
    for name, ok in results.items():
        status = "PASS" if ok else ("SKIPPED" if ok is None else "FAIL")
        print(f"  {name}: {status}")
        if ok is False:
            failed = True
    print("=" * 60)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
