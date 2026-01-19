# Inference & Demo

Once you have trained a LLaVA model (or if you want to test the pipeline with a base model), you can use the inference script to launch an interactive web interface.

## Quick Start

You can launch the demo using the `scripts/demo.py` script. This script loads the base model, applies the LoRA adapter (if trained), and serves a Gradio UI.

```bash
python scripts/demo.py \
    --base_model HuggingFaceTB/SmolLM-135M \
    --adapter ./checkpoints/local_lite/checkpoint-final
```

*   **`--base_model`**: The identifier or path of the base LLM used during training.
*   **`--adapter`**: The path to your trained LoRA adapter (optional). If omitted, it runs the base model (which won't have multimodal capabilities unless it's already an LLaVA model).

## Features

*   **Interactive Chat**: Upload images and ask questions.
*   **Parameter Tuning**: Adjust `temperature` and `top_p` via the UI accordion.
*   **Public Sharing**: The script automatically launches with `share=True`, giving you a temporary public URL (e.g., `https://xxxx.gradio.live`) to share with others.

## Running on Different Hardware

### GPU (Recommended)
The script automatically detects CUDA. If available, it loads the model in 4-bit quantization (if dependencies allow) for efficiency.

```bash
python scripts/demo.py \
    --base_model HuggingFaceTB/SmolLM-135M \
    --adapter ./checkpoints/local_lite/checkpoint-final
```

### CPU (Local Testing)
If you are on a CPU machine, you can force CPU mode (though it will be slow):

```bash
python scripts/demo.py \
    --base_model HuggingFaceTB/SmolLM-135M \
    --adapter ./checkpoints/local_lite/checkpoint-final \
    --cpu \
    --no_quant
```

*   `--cpu`: Forces the model to run on CPU.
*   `--no_quant`: Disables 4-bit quantization (BNB is often GPU-only).

## Accessing the Demo in Cloud Environments (Colab)

When running in Google Colab, the local URL `http://0.0.0.0:7860` is not directly accessible. However, since `share=True` is enabled by default, look for the **Public URL** in the console output:

```text
Running on public URL: https://e38d428360e3b205ff.gradio.live
```

Click this link to open the interface in a new tab.

```