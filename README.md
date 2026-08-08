# Nano-vLLM-optimized

This repository is a secondary-development fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Optimizations

1. **Installation workflow**: provides a Conda environment, offline matching FlashAttention wheel installation, and ModelScope model download commands.
2. **`serving_bench.py`**: benchmarks offline engine throughput and serving end-to-end goodput, including TTFT, TPOT, and request latency.
3. **Streaming output**: `LLM.generate_stream()` yields `(prompt_index, new_token_ids, is_finished)` after each engine step, so callers can consume tokens as soon as they are produced instead of waiting for the whole batch. `generate()` keeps its original behavior as a thin wrapper over the streaming primitive.
4. **Top-k / Top-p sampling**: extends `SamplingParams` with `top_k` and `top_p` (disabled by default) to filter low-probability candidates before sampling. The default path is unchanged; when filtering is requested, candidates are pruned by top-k then top-p before the existing exponential-race sampling.
5. **Multi-model support**: a dispatch table in `model_runner.py` selects the model class by `hf_config.model_type`. Adding a new architecture is a matter of dropping a `models/xxx.py` and registering one line.

## Supported Models

The engine selects the model implementation at runtime via `hf_config.model_type`. Each architecture lives in its own file under `nanovllm/models/` for easy comparison of architectural differences.

| Architecture | `model_type` | File | Status |
| --- | --- | --- | --- |
| Qwen3 (dense) | `qwen3` | `models/qwen3.py` | Supported |
| Qwen2 / Qwen2.5 | `qwen2` | `models/qwen2.py` | Supported |
| Llama 3.x | `llama` | `models/llama.py` | Supported |

To add a new architecture, create `models/xxx.py` and register it in the `model_dict` table in `engine/model_runner.py`.

### Reference throughput

Hardware: single RTX 4090, `bench.py` (256 requests, random 100–1024 token inputs/outputs, `ignore_eos=True`).

| Model | Params | Throughput |
| --- | ---: | ---: |
| Qwen3-0.6B | 0.6B | ~5700 tok/s |
| Qwen2.5-0.5B | 0.5B | ~19500 tok/s |
| Llama-3.2-1B | 1B | ~10900 tok/s |

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

Hardware: Qwen3-0.6B on a single RTX 4090. Streaming output is a capability addition, not a performance optimization; results below confirm it is throughput-neutral.

#### Engine throughput (`bench.py`)

256 requests, input 100–1024 tokens (uniform), output 100–1024 tokens (uniform), `temperature=0.6`, `ignore_eos=True`.

| Version | Total tokens | Total time | Throughput |
| --- | ---: | ---: | ---: |
| Baseline | 133966 | 23.61 s | 5674.12 tok/s |
| + streaming output | 133966 | 23.80 s | 5628.70 tok/s |

#### Serving (`serving_bench.py`)

256 requests, fixed 512-token inputs and 128-token outputs.

| Version | Mode | Total time | Output throughput | Mean TTFT | Mean TPOT | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | Offline | 5.76 s | 5688.38 tok/s | 1560.51 ms | 27.80 ms | 5091.67 ms |
| Baseline | Serving (8 req/s) | 36.32 s | 902.21 tok/s | 61.21 ms | 3.89 ms | 555.81 ms |
| + streaming output | Offline | 5.63 s | 5819.57 tok/s | 1599.96 ms | 26.68 ms | 4988.93 ms |
| + streaming output | Serving (8 req/s) | 36.31 s | 902.33 tok/s | 63.68 ms | 3.70 ms | 533.90 ms |
