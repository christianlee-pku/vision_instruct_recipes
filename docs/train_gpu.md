# Training on GPU (Cloud Scale)

This guide explains how to run high-performance training on GPU instances (e.g., Google Colab T4, A100 clusters). We use **QLoRA (4-bit quantization)** to fit the model into consumer-grade GPU memory (16GB-24GB) or **DeepSpeed** for multi-GPU scaling.

## Prerequisites

Ensure you have installed the GPU-optimized dependencies. This is critical for 4-bit quantization support.

**Note**: Ensure you have the correct CUDA version installed for your GPU to match the PyTorch and BitsAndBytes requirements.

```bash
# Install the GPU stack
pip install -r requirements_gpu.txt
```

## Configuration Profiles

We provide two main profiles for GPU training:

1.  **`t4_qlora` (Consumer GPU / Low VRAM)**:
    *   Optimized for 16GB VRAM.
    *   Uses 4-bit quantization (`load_in_4bit=true`).
    *   Batch size 1 with Gradient Accumulation 16.
    *   FP16 precision.

2.  **`cloud_gpu_scale` (Multi-GPU Cluster)**:
    *   Uses DeepSpeed ZeRO-2.
    *   BF16 precision (if supported).
    *   Larger batch sizes.

## Pre-downloading Models

To avoid network timeouts or "gated repo" errors during the training startup, we recommend pre-downloading the models to a local cache.

```bash
# This script downloads CLIP and the LLM to ./model_cache/
python scripts/download_models.py
```

## Running Training

### Single GPU or Custom Count

Use the provided helper script `scripts/train_cloud.sh`. It handles model caching and launching.

**Usage:**
```bash
bash scripts/train_cloud.sh [experiment_name] [num_gpus]
```

**Examples:**

```bash
# 1. Default: Run 'cloud_gpu_scale' on ALL available GPUs
bash scripts/train_cloud.sh

# 2. Run on exactly 4 GPUs
bash scripts/train_cloud.sh cloud_gpu_scale 4

# 3. Run single-GPU T4 profile
bash scripts/train_cloud.sh t4_qlora 1
```

**What this script does:**
*   Checks for `accelerate`.
*   Sets up WandB logging (online mode).
*   Points the training script to the local `./model_cache/` to avoid network issues.
*   Launches `scripts/train.py` with the specified experiment and GPU count.

### Manual Launch (Advanced)

To run on a cluster without the helper script, use `accelerate launch` directly:

```bash
accelerate launch scripts/train.py experiment=cloud_gpu_scale
```

Ensure you have configured accelerate (`accelerate config`) to use DeepSpeed before running this.

## Troubleshooting

*   **`AttributeError: 'weight' is not an nn.Module`**: This is a known issue with QLoRA loading in `transformers`. Our codebase includes a specific fix in `src/models/llava_arch.py` (`_initialize_missing_keys` override). If you see this, ensure you are using the latest code from the `main` branch.
*   **`NotImplementedError: Cannot copy out of meta tensor`**: This happens if the custom projector modules are initialized on the "meta" device. Our `LlavaModel` factory handles this by safely initializing them on CPU first.
*   **`OSError: Can't load model`**: Run `python scripts/download_models.py` again to ensure all weights (especially `safetensors` or `bin` files) are correctly cached.
