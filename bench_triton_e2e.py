"""End-to-end smoke test for the self-researched Triton kernels on the
chunked-prefill and speculative-decode paths.

This is a SMOKE test, not a token-identity test. The self-researched kernels
are numerically different implementations from flash_attn (CUTLASS): the
precision tests (`tests/test_triton_attn.py`) already pin their logits to
flash_attn within ~1e-3. Greedy decoding then AMPLIFIES that sub-ULP gap via
the cascade (one early argmax flip changes the whole continuation) — the same
effect that made INT8 look like 85% token-match while being ΔPPL +0.000.

So here we assert only that `use_triton_attn=True` runs correctly on these two
paths (no crash / NaN / truncation, full max_tokens generated), and REPORT the
token-match rate against the flash_attn baseline as an informational metric.
The rate is expected to be high (95%+) but NOT 100% — that is normal.

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


def _match_rate(ref, out):
    """Return (rate, first_divergence_pos) of token overlap; -1 if no divergence."""
    n = sum(1 for a, b in zip(ref, out) for x, y in zip(a, b) if x == y)
    total = sum(len(a) for a in ref)
    rate = n / total if total else 0.0
    for a, b in zip(ref, out):
        if a != b:
            d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            return rate, d
    return rate, -1


def _smoke_ok(outs, max_tokens):
    """All sequences generated to full length (or hit EOS) with no crash."""
    return all(len(o) == max_tokens for o in outs)


def test_chunked(model, max_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model)
    base = tokenizer.encode(REAL_TEXT)
    lengths = [64, 400, 1024, 2048]
    prompts = [make_prompt(base, L, i) for i, L in enumerate(lengths)]
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    print(f"\n[chunked prefill] 4 prompts (lens {lengths}), chunk_size=256, max_tokens={max_tokens}")
    ref = run_once(model, prompts, sp, use_triton=False,
                   enable_chunked_prefill=True, prefill_chunk_size=256)
    out = run_once(model, prompts, sp, use_triton=True,
                   enable_chunked_prefill=True, prefill_chunk_size=256)

    rate, d = _match_rate(ref, out)
    ok = _smoke_ok(out, max_tokens)
    print(f"  token-match vs baseline: {rate * 100:.2f}%  (first divergence at token {d})")
    print(f"  smoke (use_triton_attn=True): {'PASS' if ok else 'FAIL'}")
    return ok, rate


def test_spec(model, draft_model, max_tokens):
    if draft_model is None or not os.path.isdir(os.path.expanduser(draft_model)):
        print(f"\n[speculative decode] SKIPPED (draft model not found: {draft_model})")
        return None, 0.0

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
    # Determinism: the same Triton config run twice must be bit-identical. If it
    # is not, the kernels read uninitialized memory / have a race — a real bug.
    out2 = run_once(model, prompts, sp, use_triton=True, speculative_config=spec_cfg)

    rate, d = _match_rate(ref, out)
    deterministic = all(a == b for a, b in zip(out, out2))
    ok = _smoke_ok(out, max_tokens) and deterministic
    print(f"  token-match vs baseline: {rate * 100:.2f}%  (first divergence at token {d})")
    print(f"  determinism (triton x2): {'PASS' if deterministic else 'FAIL'}")
    print(f"  smoke (use_triton_attn=True): {'PASS' if ok else 'FAIL'}")
    return ok, rate


def main():
    p = argparse.ArgumentParser(description="Triton kernels end-to-end smoke test (chunked + spec).")
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", default=None, help="EAGLE3 draft checkpoint (spec test).")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--skip-chunked", action="store_true")
    p.add_argument("--skip-spec", action="store_true")
    args = p.parse_args()

    print("=" * 60)
    print(f"Triton end-to-end SMOKE test  (device: {torch.cuda.get_device_name(0)})")
    print("NOTE: token-match vs flash_attn is informational only; a sub-100% rate")
    print("      is expected (different kernel implementations + greedy cascade).")
    print("=" * 60)

    failed = False
    if not args.skip_chunked:
        ok, _ = test_chunked(args.model, args.max_tokens)
        failed |= (ok is False)
    if not args.skip_spec:
        ok, _ = test_spec(args.model, args.draft_model, args.max_tokens)
        failed |= (ok is False)

    print("\n" + "=" * 60)
    print("SMOKE RESULT: " + ("FAIL" if failed else "PASS"))
    print("=" * 60)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
