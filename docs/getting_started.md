# Vision Instruct Recipes: Scalable LLaVA Tuning

Welcome to the documentation for **Vision Instruct Recipes**, a comprehensive toolkit for fine-tuning LLaVA (Large Language and Vision Assistant) models. This documentation guides you through the process of setting up your environment, preparing data, training models (from local debugging to cloud-scale), and deploying them for inference.

## 📚 Table of Contents

1.  **[Introduction](#introduction)**
2.  **[Architecture Overview](#architecture-overview)**
3.  **[Installation & Setup](#installation--setup)**
4.  **[Data Preparation](#data-preparation)**
5.  **[Training](#training)**
    *   [Local-Lite (Single GPU/CPU)](#local-lite-single-gpucpu)
    *   [Cloud-Scale (Multi-GPU)](#cloud-scale-multi-gpu)
    *   [QLoRA & Quantization](#qlora--quantization)
6.  **[Inference & Demo](#inference--demo)**
7.  **[Configuration System](#configuration-system)**
8.  **[Troubleshooting](#troubleshooting)**

---

## Introduction

This project provides a robust, modular, and scalable pipeline for Visual Instruction Tuning. It is designed to be:

*   **Accessible**: Train on consumer hardware (e.g., RTX 3090, T4) using 4-bit quantization (QLoRA).
*   **Scalable**: Seamlessly transition to multi-GPU clusters using DeepSpeed ZeRO-2.
*   **Modular**: Clean separation of model logic, data processing, and training configurations using [Hydra](https://hydra.cc/).

---

## Architecture Overview

The core architecture follows the LLaVA design pattern:

*   **Base LLM**: LLaMA-3 (8B) or compatible variants (e.g., Vicuna, SmolLM).
*   **Vision Encoder**: OpenAI CLIP (ViT-L/14-336).
*   **Projector**: A Multi-Layer Perceptron (MLP) that aligns visual features with the LLM's embedding space.

We use **Late Fusion**: Image features are inserted into the text sequence at specific `<image>` placeholder tokens.

---

## Installation & Setup

### Prerequisites

*   Python 3.10+
*   CUDA 11.8+ (for GPU training)

### Step-by-Step Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/christianlee-pku/vision_instruct_recipes.git
    cd vision_instruct_recipes
    ```

2.  **Create Conda Environment**
    ```bash
    conda create -n llava python=3.10
    conda activate llava
    ```

3.  **Install Dependencies**
    
    *   **For Cloud/GPU Users (Recommended)**:
        ```bash
        # Installs torch 2.5.1+cu124, bitsandbytes 0.45.0, transformers 4.47.1
        pip install -r requirements_gpu.txt
        ```
    
    *   **For Local/CPU Users (Debugging)**:
        ```bash
        pip install -r requirements_cpu.txt
        ```

---

## Data Preparation

The training pipeline expects a JSON file for instructions and a folder containing images.

1.  **Images**: Download the COCO 2017 Train dataset and unzip it to `data/coco/images`.
2.  **Instructions**: Prepare a JSON file (e.g., `llava_instruct_150k.json`) in LLaVA format:
    ```json
    [
      {
        "id": "000000039769",
        "image": "000000039769.jpg",
        "conversations": [
          {"from": "human", "value": "<image>\nWhat is in the image?"},
          {"from": "gpt", "value": "Two cats sleeping on a pink blanket."}
        ]
      }
    ]
    ```
3.  **Config Update**: Ensure `configs/data/default.yaml` points to your files:
    ```yaml
    data_path: "data/coco/llava_instruct_150k.json"
    image_folder: "data/coco/images"
    ```

---

## Training

We support different "Experiment Profiles" managed by Hydra configs in `configs/experiment/`.

### Local-Lite (Single GPU/CPU)

Best for debugging code or running on limited hardware (e.g., Colab free tier, local RTX 3090, or even CPU).

*   **Features**: QLoRA (4-bit), Gradient Checkpointing, Batch Size 1 (accumulated).
*   **Command**:
    ```bash
    # Using the helper script (Recommended)
    bash scripts/train_local.sh
    
    # Or manually
    python scripts/train.py experiment=local_lite
    ```

### Cloud-Scale (Multi-GPU)

Best for full training runs on clusters (e.g., 8x A100).

*   **Features**: DeepSpeed ZeRO-2, BF16 precision, Higher Batch Size.
*   **Command**:
    ```bash
    # Using the helper script (auto-detects all GPUs)
    bash scripts/train_cloud.sh
    
    # Or specify GPU count (e.g., 4 GPUs)
    bash scripts/train_cloud.sh cloud_gpu_scale 4
    
    # Or manually
    accelerate launch scripts/train.py experiment=cloud_gpu_scale
    ```

### QLoRA & Quantization

To save memory, we use 4-bit quantization (QLoRA) by default in the `local_lite` profile. This allows loading a 7B model on ~6GB VRAM.
*   **Config**: Controlled by `model.use_qlora=true` and `model.load_in_4bit=true`.
*   **Compatibility**: Requires `bitsandbytes` and a CUDA GPU. CPU users must disable this (handled automatically in `train_local.sh`).

---

## Inference & Demo

Once trained, interact with your model using the Gradio web UI.

```bash
python scripts/demo.py \
    --base_model HuggingFaceTB/SmolLM-135M \
    --adapter ./checkpoints/local_lite/checkpoint-final
```

---

## Configuration System

We use **Hydra** for hierarchical configuration.

*   `configs/config.yaml`: The entry point. Defaults are defined here.
*   `configs/model/`: Model architecture settings (vision tower, projector type).
*   `configs/data/`: Dataset paths and processing parameters.
*   `configs/training/`: HF Trainer arguments (batch size, learning rate).
*   `configs/experiment/`: Overrides for specific scenarios (e.g., `local_lite.yaml`).

**Example: Overriding parameters via CLI**
```bash
python scripts/train.py experiment=local_lite training.learning_rate=5e-5
```

---

## Troubleshooting

For a comprehensive list of common issues and fixes, please refer to our **[Troubleshooting Guide](troubleshooting.md)**.

