# Nano-vLLM-optimized

This repository is a secondary-development fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Optimizations

1. **Installation workflow**: provides a Conda environment, offline matching FlashAttention wheel installation, and ModelScope model download commands.
2. **`serving_bench.py`**: benchmarks offline engine throughput and serving end-to-end goodput, including TTFT, TPOT, and request latency.
3. **Streaming output**: `LLM.generate_stream()` yields `(prompt_index, new_token_ids, is_finished)` after each engine step, so callers can consume tokens as soon as they are produced instead of waiting for the whole batch. `generate()` keeps its original behavior as a thin wrapper over the streaming primitive.
4. **Top-k / Top-p sampling**: extends `SamplingParams` with `top_k` and `top_p` (disabled by default) to filter low-probability candidates before sampling. The default path is unchanged; when filtering is requested, candidates are pruned by top-k then top-p before the existing exponential-race sampling.
5. **Multi-model support**: a dispatch table in `model_runner.py` selects the model class by `hf_config.model_type`. Adding a new architecture is a matter of dropping a `models/xxx.py` and registering one line.
6. **Chunked prefill**: an `enable_chunked_prefill` flag in `Config` splits long prompts into chunks that interleave with decode in the same step. Pure-decode steps automatically fall back to CUDA Graph, so TPOT stays unchanged while TTFT drops 32×. Run `python bench_chunked.py` to compare ON vs OFF.
7. **INT8 KV cache quantization**: a `kv_quant` flag quantizes KV cache to INT8 with per-(token, head, group) symmetric Min-Max scaling — head_dim is split into 8 groups of 16 dims, each with its own dynamic scale, so outlier dimensions no longer crush the precision of the other dims. Combined with the self-researched Triton fused-dequant kernel (item 9), decode reads the INT8 cache directly (no whole-cache dequant pass), so memory *and* throughput both improve. Teacher-forced eval verifies it is near-lossless (continuation ΔPPL **+0.000**) — see the Triton Attention Kernels section.
8. **Speculative decoding (EAGLE3)**: a chain-style EAGLE3 draft head (single Transformer decoder layer conditioned on low/mid/high target-layer features) proposes K candidate tokens that a standard rejection sampler verifies against the target in one pass. Supports greedy and temperature/top-k/top-p sampling; greedy output is token-for-token identical to non-spec decoding. Run `python bench_spec_decode.py`; see the [Speculative Decoding](#speculative-decoding-eagle3) section for numbers.
9. **Self-researched Triton attention kernels**: FlashAttention-2 prefill (causal + GQA + varlen), paged-attention decode, a fused INT8 dequant decode kernel, and Flash-Decoding (split-K). Gated behind a `use_triton_attn` flag (default off, `flash_attn` package untouched). See the [Triton Attention Kernels](#triton-attention-kernels) section for benchmarks.

## Supported Models

The engine selects the model implementation at runtime via `hf_config.model_type`. Each architecture lives in its own file under `nanovllm/models/` for easy comparison of architectural differences.

| Architecture | `model_type` | File | Status |
| --- | --- | --- | --- |
| Qwen3 (dense) | `qwen3` | `models/qwen3.py` | Supported |
| Qwen2 / Qwen2.5 | `qwen2` | `models/qwen2.py` | Supported |
| Llama 3.x | `llama` | `models/llama.py` | Supported |
| Gemma 3 | `gemma3` / `gemma3_text` | `models/gemma3.py` | Supported |

To add a new architecture, create `models/xxx.py` and register it in the `model_dict` table in `engine/model_runner.py`.

### Reference throughput

Hardware: single RTX 4090, `bench.py` (256 requests, random 100–1024 token inputs/outputs, `ignore_eos=True`).

| Model | Params | Throughput |
| --- | ---: | ---: |
| Qwen3-0.6B | 0.6B | ~5700 tok/s |
| Qwen2.5-0.5B | 0.5B | ~19500 tok/s |
| Llama-3.2-1B | 1B | ~10900 tok/s |
| Gemma-3-1B | 1B | ~9800 tok/s |

## Installation

This project requires an NVIDIA GPU and FlashAttention. The recommended platform is Linux or WSL2. Native Windows is not supported by the NCCL and FlashAttention dependencies used by the engine.

### 1. Clone the repository and create the Conda environment

```bash
git clone https://github.com/brightmoonflame/nano-vllm-optimized.git
cd nano-vllm-optimized
conda env create -f environment.yml
conda activate nano-vllm
```

### 2. Install a matching FlashAttention wheel

Download a prebuilt wheel from the [FlashAttention releases page](https://github.com/Dao-AILab/flash-attention/releases/). Run this command to print the matching filename:

```bash
FLASH_ATTN_VERSION=2.7.4.post1 python - <<'PY'
import os, platform, sys, torch

python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_tag = "torch" + ".".join(torch.__version__.split("+", 1)[0].split(".")[:2])
cuda_tag = "cu" + torch.version.cuda.split(".")[0]
abi_tag = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
arch = platform.machine().lower().replace("amd64", "x86_64")
print(f"flash_attn-{os.environ['FLASH_ATTN_VERSION']}+{cuda_tag}{torch_tag}cxx11abi{abi_tag}-{python_tag}-{python_tag}-linux_{arch}.whl")
PY
```

Download the printed file, then install it:

```bash
pip install --no-deps /path/to/flash_attn-<matching-version>.whl
```

Verify the wheel:

```bash
python -c "import torch, flash_attn; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('FlashAttention:', flash_attn.__version__)"
```

### 3. Install Nano-vLLM-optimized

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

`--no-deps` prevents pip from trying to download or rebuild FlashAttention after the matching offline wheel has been installed.

Verify the installation and CUDA:

```bash
python - <<'PY'
import torch, flash_attn, modelscope
from transformers import Qwen3Config
from nanovllm import LLM, SamplingParams

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("Imports OK")
PY
```

### 4. Download a model with ModelScope

ModelScope is installed by default. Download Qwen3-0.6B to the path used below:

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir /root/model/Qwen3-0.6B
```

Use `/root/model/Qwen3-0.6B` as the `--model` path. If you run files with hard-coded model paths, update them to this directory as well.

## Streaming Output

`generate_stream()` yields raw token ids incrementally after each engine step. For ready-to-print text, `generate_stream_text()` wraps it with incremental detokenization (BPE-safe, holds back half-formed characters):

```python
for index, text_delta, is_finished in llm.generate_stream_text(prompts, sampling_params):
    print(f"[req {index}] {text_delta}", flush=True)
```

`generate()` still returns full results at once. Run `python example.py` to see live streaming from two interleaved requests.

## Sampling Parameters

`SamplingParams` supports `temperature`, `top_k`, and `top_p`:

```python
SamplingParams(temperature=0.6, top_p=0.9)              # nucleus sampling
SamplingParams(temperature=0.6, top_k=50, top_p=0.9)    # top-k then top-p
```

- `top_k` (default `-1`): keep only the top-k logits; `-1` disables.
- `top_p` (default `1.0`): keep the smallest set of tokens whose cumulative probability reaches `top_p`; `1.0` disables.
- When both are disabled, the original full-vocab sampling path is used (no overhead).

With `top_p=0.9` on Qwen3-0.6B, low-probability tail noise is suppressed — e.g. self-introduction responses no longer emit placeholder tokens like `[Your Name]`, and reasoning tasks stay on track instead of drifting toward irrelevant tokens. The trade-off is ~32% higher TPOT due to the full-vocab sort; combining with `top_k=50` first bounds the sort cost.

`serving_bench.py` exposes `--top-k` and `--top-p` for benchmarking different sampling strategies.

## Serving Benchmark

`offline` measures engine throughput after all requests are queued. `serving`
simulates Poisson arrivals and measures end-to-end goodput and latency.

### Offline

```bash
python serving_bench.py \
  --model /root/model/Qwen3-0.6B \
  --mode offline \
  --num-requests 256 \
  --length-distribution fixed \
  --min-input-len 512 \
  --max-input-len 512 \
  --min-output-len 128 \
  --max-output-len 128 \
  --max-model-len 1024 \
  --seed 0 \
  --output-json results/offline-baseline.json
```

### Serving

```bash
python serving_bench.py \
  --model /root/model/Qwen3-0.6B \
  --mode serving \
  --num-requests 256 \
  --request-rate 8 \
  --length-distribution fixed \
  --min-input-len 512 \
  --max-input-len 512 \
  --min-output-len 128 \
  --max-output-len 128 \
  --max-model-len 1024 \
  --seed 0 \
  --output-json results/serving-rate8.json
```

### Results

Hardware: Qwen3-0.6B on a single RTX 4090. All features enabled, chunked prefill off by default. Streaming output is a capability addition, not a performance optimization — throughput remains neutral.

#### Engine throughput (`bench.py`)

256 requests, input 100–1024 tokens (uniform), output 100–1024 tokens (uniform), `temperature=0.6`, `ignore_eos=True`.

| Chunked prefill | Throughput |
| --- | ---: |
| OFF | ~5480 tok/s |
| ON | ~5740 tok/s (+5%) |

#### Chunked prefill TTFT (`bench_chunked.py`)

8 requests, ~474-token prompts, `enforce_eager=True`, `prefill_chunk_size=512`. Measures per-request TTFT (time from submission to first token) and TPOT (time per output token during decode).

| Chunked prefill | TTFT (mean) | TPOT (mean) |
| --- | ---: | ---: |
| OFF | 1547 ms | 28.0 ms |
| ON | **47 ms (32×)** | 28.2 ms |

The 32× TTFT reduction comes from interleaving: the first request only needs to prefill one 512-token chunk before producing output, instead of waiting for all 8 requests to finish prefill. TPOT stays flat because pure-decode steps automatically fall back to CUDA Graph.

#### INT8 KV cache quantization (`bench.py` + `bench_memory.py`)

`bench_memory.py` measures KV-cache capacity under the same GPU memory budget (see Triton Attention Kernels below for throughput, which now *improves* with the fused INT8 kernel).

| KV cache | Per-block memory | Blocks | Concurrent capacity |
| --- | ---: | ---: | ---: |
| BF16 (default) | 29.4 MB | 414 | 1× |
| INT8 (`kv_quant=True`) | **18.4 MB (-37%)** | **663 (+60%)** | **~1.6×** |

INT8 cuts per-block memory to ~63% of BF16, fitting ~1.6× the blocks (and therefore ~1.6× the max context or concurrent sequences) into the same GPU budget. The ratio is 1.6× rather than 2× because the group-wise scales add overhead: each (token, head) costs 128 B (INT8 data) + 32 B (8 × FP32 scales) = 160 B vs 256 B for BF16. With the fused Triton dequant kernel, this no longer costs throughput — see below.

#### CUDA Graph (`serving_bench.py` + `bench_prefill_graph.py`)

**Decode CUDA Graph** (256 requests, 512-token inputs, 128-token outputs, serving 8 req/s). The gain scales inversely with model size — launch overhead is a bigger fraction of per-step cost on small models:

| Model | Eager TPOT | CUDA Graph TPOT | Speedup |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 35.3 ms | 3.7 ms | **9.5×** |
| Llama-3.2-3B | 42.3 ms | 40.0 ms | ~1.1× |

Decode processes only 1 token per sequence per step, so kernel-launch overhead dominates on small models; on 3B the per-step compute is large enough that launch overhead shrinks to ~5–10%.

**Prefill CUDA Graph** (single-request first Dense Prefill):

| Prompt length | Graph OFF | Graph ON | Delta |
| --- | ---: | ---: | ---: |
| 1024 | 15.9 ms | 16.0 ms | +0.1 ms |
| 8192 | 18.1 ms | 17.3 ms | -0.7 ms |

Prefill processes hundreds to thousands of tokens per step, so kernel-launch overhead is a small fraction of total compute. On 0.6B–1B models the prefill itself takes only 3–5 ms, making graph savings (~1 ms) hard to measure. The feature is functionally correct and would show significant gains on 7B+ models where prefill exceeds 50 ms.

#### Serving (`serving_bench.py`)

256 requests, fixed 512-token inputs and 128-token outputs.

| Mode | Output throughput | Mean TTFT | Mean TPOT | Mean latency |
| --- | ---: | ---: | ---: | ---: |
| Offline | 5730 tok/s | 1647 ms | 26.4 ms | 4997 ms |
| Serving (8 req/s) | 902 tok/s | 66 ms | 3.8 ms | 543 ms |

## Triton Attention Kernels

Self-researched Triton kernels replace the `flash_attn` package on the prefill and decode paths, gated behind `use_triton_attn` (default off). Two unified entry points — `triton_flash_attn_varlen` (prefill: dense / paged / INT8) and `triton_paged_attention` (decode: BF16 / INT8) — backed by three Triton kernels (`_fwd_kernel`, `_paged_attn_decode_kernel`, `_split_reduce_kernel`). Together they cover FlashAttention-2 prefill, paged-attention decode, fused INT8 dequant, and Flash-Decoding (split-K). Variants are selected at compile time via `IS_PAGED` / `IS_INT8` `tl.constexpr` switches, so every compiled kernel contains only its own code path (zero runtime dispatch overhead).

Hardware: Llama-3.2-3B-Instruct (24 Q / 8 KV heads, GQA 3:1), single RTX 4090.

### Kernel-level latency

`bench_triton_prefill.py` / `bench_triton_decode.py` compare the raw kernels against `flash_attn` (ratio = triton / flash_attn, <1 means Triton is faster):

- **Prefill FA2**: ~94–110% of `flash_attn_varlen_func` (long sequences 80%+ after `BLOCK_M=128` tuning).
- **Decode paged (BF16)**: ~92–98% of `flash_attn_with_kvcache` at batch ≥32; small batch ~1.85× slower (Flash-Decoding improves this from ~2.4× but does not yet close the gap — a known limitation).
- **Decode fused INT8**: **~1.03–1.17× faster than `flash_attn` BF16** (reads half the bytes), and **~3–3.7× faster than the default `dequant + flash_attn` INT8 path** (eliminates the whole-cache dequant pass).

### Memory

`bench_memory.py`: under the same GPU budget, the INT8 KV cache holds **~1.6× the blocks/tokens** of BF16 (663 vs 414 blocks on Llama-3.2-3B) — see the INT8 KV cache quantization section above for the per-block breakdown.

### Accuracy

Two complementary metrics on Llama-3.2-3B-Instruct (16 prompts, 128-token continuations):

**Teacher-forced (cascade-free) — the number that matters.** `bench_teacher_forced.py` feeds both models the *same* reference continuation (the BF16 model's own greedy output) and scores only the next-token prediction at each step, so the contexts never diverge:

| Metric | Result |
| --- | ---: |
| next-token agreement (INT8 vs BF16) | **99.46%** (2022/2033) |
| continuation PPL — BF16 / INT8 | 1.259 / 1.259 (Δ **+0.000**) |

ΔPPL ≈ 0 means the INT8 KV cache is effectively lossless — the standard result to cite (KVQuant targets < 0.1 ΔPPL for 4-bit; 8-bit is well inside that). A BF16-vs-its-own-reference self-check runs at 99.70% (driver sanity; the residual is kernel reduction-order noise between batch sizes).

**Free-running greedy (`bench_accuracy.py`) — pessimistic by design.** Greedy decoding is autoregressive: a single early argmax flip changes the context for every later token, after which the two sequences generate different-but-valid continuations that all score as mismatches. That cascade makes a near-lossless quantizer look ~85%:

| Metric | Result |
| --- | ---: |
| exact-match rate | 68.8% (11/16) |
| token-match rate | 85.19% |
| avg tokens before first divergence | 107.9 / 128 |

Together they show the INT8 cache is near-lossless *per step* (99.46%); the ~85% figure is an artifact of greedy cascade, not quantization error.

### CUDA Graph

For models whose target Attention layers all use the fused global Triton path
(for example Llama/Qwen), INT8 KV Decode can also use CUDA Graph: the kernel
reads INT8 K/V plus scales directly and creates no whole-cache BF16 temporary.
Models with a FlashAttention fallback layer, such as Gemma sliding-window
Attention, stay eager for correctness. Compare eager and graph Decode with
`python bench_int8_cudagraph.py --model /root/model/Llama-3.2-3B-Instruct`.

### Commands

```bash
python bench_triton_prefill.py
python bench_triton_decode.py
python bench_int8_cudagraph.py --model /root/model/Llama-3.2-3B-Instruct
python bench_prefill_graph.py --model /root/model/Llama-3.2-3B-Instruct --use-triton-attn --kv-quant
python test_cudagraph.py --model /root/model/Llama-3.2-3B-Instruct
python bench_memory.py --model /root/model/Llama-3.2-3B-Instruct
python bench_accuracy.py --model /root/model/Llama-3.2-3B-Instruct
python bench_teacher_forced.py --model /root/model/Llama-3.2-3B-Instruct
python serving_bench.py --model /root/model/Llama-3.2-3B-Instruct --mode offline --use-triton-attn --kv-quant
```

## Speculative Decoding (EAGLE3)

EAGLE3 speculative decoding uses a chain-style draft head — a single Transformer decoder layer conditioned on low/mid/high target-layer features (an `fc: 3H→H` fusion) that shares the target embedding and maps a small draft vocabulary back to the target vocabulary via `d2t`. It covers greedy and temperature/top-k/top-p sampling: greedy output is token-for-token identical to non-spec decoding, and sampling output matches the target distribution via standard rejection sampling (`min(1, p/q)` acceptance + `max(0, p−q)` residual replacement).

**Requirements**: a Llama target plus its matching EAGLE3 draft checkpoint. The canonical pair is `meta-llama/Llama-3.2-3B-Instruct` (target) + `thoughtworks/Llama-3.2-3B-Instruct-Eagle3` (draft).

### Correctness

Under greedy (`temperature=0`), spec and non-spec produce token-for-token identical output:

```
[1] Correctness (greedy, temperature=0)
    spec == non-spec ✓ (32 tokens)
```

### Performance

Hardware: single RTX 4090, real-text prompts (100–512 tokens), 512 output tokens, `temperature=0`, `enforce_eager=True`. The spec path is eager-only, so this is an eager-vs-eager comparison that isolates the draft mechanism from CUDA-graph launch savings.

**K sweep (batch=1)** — acceptance rate rises monotonically with K, but speedup peaks around K=3 once the draft self-chain overhead outgrows the extra accepted tokens:

| K (`num_spec_tokens`) | Speedup | Accept rate (tokens/round) |
| --- | ---: | ---: |
| 1 | 1.26× | 0.65 |
| 3 | **1.54×** | 1.20 |
| 5 | 1.47× | 1.30 |
| 7 | 1.40× | 1.40 |

**Batch sweep (K=5)** — the gain holds across batch sizes:

| Batch | no-spec tok/s | spec tok/s | Speedup | Accept rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 45.2 | 61.4 | 1.36× | 1.30 |
| 2 | 82.6 | 109.6 | 1.33× | 1.29 |
| 4 | 161.5 | 166.3 | 1.03× | 0.53 |
| 8 | 314.1 | 525.0 | 1.67× | 1.42 |

Acceptance rate — the driver of speedup — depends on the specific prompt, so individual runs vary (e.g. the batch=4 point drew a harder prompt segment). The trend is the signal: speedup is roughly 1.3–1.7× across batch sizes rather than the "small-batch-only" behavior of random-token benchmarks.

### Why ~1.5× and not 3×+

EAGLE3's paper reports 3×+ on 7B–70B targets. This fork benchmarks on a **3B target**, where the single-layer draft is proportionally expensive relative to the target, so net speedup is ~1.5×. The mechanism is the same; on larger targets the draft overhead amortizes over a bigger target forward and speedup grows.
