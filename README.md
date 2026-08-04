# Nano-vLLM-optimized

This repository is a secondary-development fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Optimizations

1. **Installation workflow**: provides a Conda environment, offline matching FlashAttention wheel installation, and ModelScope model download commands.
2. **`serving_bench.py`**: benchmarks offline engine throughput and serving end-to-end goodput, including TTFT, TPOT, and request latency.

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

Results below are recorded on each optimization step to track progress. Hardware: Qwen3-0.6B on a single RTX 4090.

#### Engine throughput (`bench.py`)

256 requests, input 100–1024 tokens (uniform), output 100–1024 tokens (uniform), `temperature=0.6`, `ignore_eos=True`.

| Version | Total tokens | Total time | Throughput |
| --- | ---: | ---: | ---: |
| Baseline | 133966 | 23.61 s | 5674.12 tok/s |

#### Serving (`serving_bench.py`)

256 requests, fixed 512-token inputs and 128-token outputs.

| Version | Mode | Total time | Output throughput | Mean TTFT | Mean TPOT | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | Offline | 6.70 s | 4893.88 tok/s | 2507.97 ms | 27.75 ms | 6032.34 ms |
| Baseline | Serving (8 req/s) | 36.32 s | 902.32 tok/s | 60.45 ms | 3.86 ms | 550.56 ms |
