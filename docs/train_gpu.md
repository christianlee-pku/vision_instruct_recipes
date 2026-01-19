# Training on GPU (Cloud Scale)

This guide explains how to run high-performance training on GPU instances (e.g., Google Colab T4, A100 clusters). We use **QLoRA (4-bit quantization)** to fit the model into consumer-grade GPU memory (16GB-24GB) or **DeepSpeed** for multi-GPU scaling.

## Prerequisites

Ensure you have installed the GPU-optimized dependencies. This is critical for 4-bit quantization support.

```bash
# First, remove any conflicting CPU-only packages
pip uninstall -y torch torchvision torchaudio bitsandbytes accelerate transformers peft

# Install the GPU stack (Tested on CUDA 12.4 / Tesla T4)
pip install -r requirements_gpu.txt
```

## Configuration Profiles

We provide two main profiles for GPU training:

1.  **`t4_qlora` (Single T4/Consumer GPU)**:
    *   Optimized for 16GB VRAM.
    *   Uses 4-bit quantization (`load_in_4bit=true`).
    *   Batch size 1 with Gradient Accumulation 16.
    *   FP16 precision.

2.  **`cloud_scale` (Multi-GPU A100)**:
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

### Single GPU (e.g., Colab T4)

Use the provided helper script `scripts/train_cloud.sh`. It automatically detects the pre-downloaded models and launches the training with `accelerate`.

```bash
chmod +x scripts/train_cloud.sh
./scripts/train_cloud.sh
```

**What this script does:**
*   Checks for `accelerate`.
*   Sets up WandB logging (online mode).
*   Points the training script to the local `./model_cache/` to avoid network issues.
*   Launches `scripts/train.py` with the `t4_qlora` experiment profile.

### Multi-GPU (DeepSpeed)

To run on a cluster, use `accelerate launch` directly with the cloud scale config:

```bash
accelerate launch scripts/train.py experiment=cloud_scale
```

Ensure you have configured accelerate (`accelerate config`) to use DeepSpeed before running this.

## Troubleshooting

*   **`AttributeError: 'weight' is not an nn.Module`**: This is a known issue with QLoRA loading in `transformers`. Our codebase includes a specific fix in `src/models/llava_arch.py` (`_initialize_missing_keys` override). If you see this, ensure you are using the latest code from the `main` branch.
*   **`NotImplementedError: Cannot copy out of meta tensor`**: This happens if the custom projector modules are initialized on the "meta" device. Our `LlavaModel` factory handles this by safely initializing them on CPU first.
*   **`OSError: Can't load model`**: Run `python scripts/download_models.py` again to ensure all weights (especially `safetensors` or `bin` files) are correctly cached.
