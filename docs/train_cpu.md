# Training on CPU (Local Debugging)

This guide covers how to run the LLaVA training pipeline on a local machine without a GPU (CPU-only). This is primarily useful for code debugging, pipeline verification, and ensuring data loading works correctly before scaling up to expensive GPU resources.

## Prerequisites

Ensure you have installed the CPU-compatible dependencies:

```bash
pip install -r requirements_cpu.txt
```

This installs a lightweight version of PyTorch and avoids GPU-specific libraries like `bitsandbytes` (which are not needed for CPU training and can cause installation issues).

## Configuration: `local_lite` Profile

We have a dedicated experiment profile `configs/experiment/local_lite.yaml` designed for this purpose.

**Key Settings for CPU:**
*   `use_qlora: false`: Quantization is disabled (CPU doesn't support 4-bit/8-bit BNB efficiently).
*   `bf16: false`, `fp16: false`: Mixed precision is disabled (standard FP32 is used).
*   `per_device_train_batch_size: 1`: Kept minimal to reduce RAM usage.
*   `vision_tower`: Uses a repository with safetensors support (`jeromeku/clip-vit-large-patch14-336_safe`) to ensure compatibility with stricter local PyTorch security settings.

## Running Training

You can use the provided helper script `scripts/train_local.sh`, which automatically sets the correct flags for a CPU run.

```bash
# Make the script executable
chmod +x scripts/train_local.sh

# Run the script
./scripts/train_local.sh
```

**What the script does:**
1.  Sets `WANDB_MODE=disabled` to avoid logging spam during debugging.
2.  Explicitly disables GPU flags: `model.use_qlora=false`, `model.load_in_4bit=false`, `training.bf16=false`.
3.  Runs `scripts/train.py` with the `local_lite` experiment config.

## Expected Output

You should see logs indicating that:
1.  The model is loading (this might take a minute on CPU).
2.  The dataset is being processed.
3.  The training loop starts (`Epoch 1/3`).
4.  Since it's on CPU, it will be very slow, but it verifies that the code runs without crashing.

## Troubleshooting

*   **`ValueError: torch.load`**: If you see an error about insecure pickle loading, ensure you are using the `train_local.sh` script, which is configured to use the safetensors-compatible vision tower.
*   **Memory Issues**: If your machine runs out of RAM (not VRAM), reduce the batch size or the image size in `configs/data/default.yaml`.
