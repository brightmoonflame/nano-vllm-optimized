# Nano-vLLM-optimized

This repository is a secondary-development fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Installation

This project requires an NVIDIA GPU and FlashAttention. The recommended platform is Linux or WSL2. Native Windows is not supported by the NCCL and FlashAttention dependencies used by the engine.

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate nano-vllm
```

### 2. Install a matching FlashAttention wheel

Download a prebuilt `.whl` from the [FlashAttention releases page](https://github.com/Dao-AILab/flash-attention/releases/). Select an asset that matches the current environment's:

- Linux and `x86_64` platform;
- Python version (this environment uses Python 3.10, so use a `cp310` wheel);
- PyTorch version;
- CUDA version; and
- C++ ABI setting, when it is included in the wheel filename.

Install the downloaded wheel without resolving dependencies online:

```bash
pip install --no-deps /path/to/flash_attn-<matching-version>.whl
```

Verify the installation:

```bash
python -c "import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__)"
```

### 3. Install Nano-vLLM-optimized

```bash
git clone https://github.com/brightmoonflame/nano-vllm-optimized.git
cd nano-vllm-optimized
pip install -r requirements.txt
pip install -e . --no-deps
```

`--no-deps` prevents pip from trying to download or rebuild FlashAttention after the matching offline wheel has been installed.

## Serving Benchmark

Run the benchmark with a local Hugging Face model directory:

```bash
python serving_bench.py \
  --model /path/to/Qwen3-0.6B \
  --num-requests 256 \
  --request-rate 8 \
  --max-input-len 1024 \
  --max-output-len 1024 \
  --max-model-len 4096 \
  --seed 0
```

The script uses random token IDs to construct a synthetic workload. It is useful for observing scheduling, KV Cache behavior, and end-to-end request handling.
