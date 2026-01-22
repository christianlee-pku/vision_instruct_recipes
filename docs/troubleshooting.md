# Troubleshooting Guide

This guide compiles common issues encountered during training, inference, and setup.

## Installation & Setup

### `ValueError: torch.load` with `weights_only=True`
*   **Cause**: Recent PyTorch versions (2.4+) default to `weights_only=True` for security, which may block loading older pickles or complex objects.
*   **Fix**: 
    1.  Use the provided `train_local.sh` or `train_cloud.sh` scripts, which set environment variables or handle loading correctly.
    2.  Downgrade PyTorch if absolutely necessary (not recommended).
    3.  Ensure you are using `safetensors` checkpoints where possible.

## Training Issues

### `AttributeError: 'weight' is not an nn.Module`
*   **Cause**: This is a known issue when loading 4-bit models (QLoRA) with specific versions of `transformers` and `accelerate`. The 4-bit linear layers are sometimes not recognized as standard modules during initialization.
*   **Fix**: 
    *   Ensure you are using the versions specified in `requirements_gpu.txt`.
    *   Our codebase includes a patch in `src/models/llava_arch.py` (`_initialize_missing_keys` override) to handle this. Ensure you are running the latest code from `main`.

### `NotImplementedError: Cannot copy out of meta tensor`
*   **Cause**: Occurs when initializing custom modules (like the Projector) on the "meta" device (used by DeepSpeed or Accelerate for zero-memory init) without subsequently assigning real weights.
*   **Fix**: Our `LlavaModel` factory handles this by safely initializing these specific layers on CPU/GPU before the main loading process. If you see this, check that your `config.yaml` is correctly pointing to the right model class.

### `OSError: Can't load model` or Connection Errors
*   **Cause**: Hugging Face Hub connection timeouts or missing files.
*   **Fix**: 
    *   Run `python scripts/download_models.py` to pre-download all necessary weights to `./model_cache/`.
    *   Check your internet connection and HF_TOKEN (if using gated models).

## CUDA / GPU Issues

### `CUDA out of memory` (OOM)
*   **Cause**: Batch size too large, or model too big for GPU.
*   **Fix**:
    *   **Reduce Batch Size**: set `training.per_device_train_batch_size=1`.
    *   **Enable Gradient Checkpointing**: Ensure `training.gradient_checkpointing=true`.
    *   **Use QLoRA**: Ensure `experiment=local_lite` is used (activates 4-bit loading).
    *   **Clean Cache**: Run `torch.cuda.empty_cache()` (though the trainer usually handles this).

### `NCCL Connection Refused` (Multi-GPU)
*   **Cause**: Firewall or network issues between nodes, or leftover processes.
*   **Fix**:
    *   Ensure all processes from previous runs are killed (`pkill -f python`).
    *   Check firewall settings if running on a cluster.
    *   Try setting `NCCL_P2P_DISABLE=1` if on consumer GPUs (e.g., RTX 3090/4090) without NVLink support.

## Inference

### Poor Generation Quality / Gibberish
*   **Cause**: 
    *   Incorrect chat template formatting (e.g., missing `<image>` token).
    *   Mismatched projector weights.
*   **Fix**: 
    *   Use the provided `scripts/demo.py` which handles the chat template automatically.
    *   Ensure the `mm_projector.bin` matches the base LLM being used.
