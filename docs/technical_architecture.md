# Technical Architecture & Implementation Guide

## 1. System Overview

**Vision Instruct Recipes** is a production-grade pipeline designed for fine-tuning Large Multimodal Models (LMMs), specifically the **LLaVA (Large Language and Vision Assistant)** architecture. 

The core design philosophy is **"Scale-Invariant"**: the codebase allows seamless transitioning from debugging on a local laptop (CPU/Single GPU) to training on cloud clusters (Multi-GPU/TPU) without changing the core logic, managed entirely via **Hydra** configurations.

## 2. Core Architecture

The system implements the LLaVA-1.5 architecture, which connects a pre-trained Vision Encoder to a Large Language Model (LLM) via a simple Multi-Layer Perceptron (MLP) adapter.

### High-Level Components

| Component | Implementation | Description |
|-----------|----------------|-------------|
| **Base LLM** | `Llama-3` (e.g., 8B) | The reasoning engine. Can be frozen or fine-tuned (LoRA). |
| **Vision Tower** | `CLIP-ViT-L/336px` | Encodes images into feature patches. Usually frozen. |
| **Projector** | `LlavaMultiModalProjector` | A simple MLP (Linear $\to$ GELU $\to$ Linear) that maps CLIP features to the LLM's embedding space. |

### Multimodal Fusion Strategy
We employ **Late Fusion**:
1.  **Text Encoding**: Text is tokenized into IDs.
2.  **Image Encoding**: Images are processed by CLIP and the Projector to get visual embeddings.
3.  **Injection**: The `<image>` token in the text sequence is replaced by the sequence of visual embeddings.
4.  **Forward Pass**: The combined sequence is fed into the LLM as a single continuous stream of embeddings.

## 3. Implementation Details

The codebase is structured into three main domains: **Models**, **Data**, and **Training**.

### 3.1. Models (`src/models`)

We use a **Mixin Pattern** to extend Hugging Face's `LlamaModel` without breaking its compatibility with tools like `bitsandbytes` (quantization) or `accelerate`.

-   **`LlavaMetaModel` (Mixin)**: 
    -   Handles the initialization of the `vision_tower` and `mm_projector`.
    -   **Critical Implementation Detail**: In QLoRA settings (`low_cpu_mem_usage=True`), standard Transformers initialization often fails for custom modules. We force the projector to initialize on CPU first to ensure valid memory allocation before moving to GPU.

-   **`LlavaLlamaForCausalLM`**:
    -   Inherits from `LlamaForCausalLM`.
    -   **`prepare_inputs_for_generation`**: Overridden to accept `pixel_values` during inference.
    -   **`forward`**: Intercepts the call, computes image features, merges them with text embeddings (`_merge_multimodal_embeddings`), and passes the result to the LLM.

-   **`LlavaMultiModalProjector`**:
    -   A standard PyTorch `nn.Module`.
    -   Projects 1024-dim CLIP features to 4096-dim Llama-3 hidden states.

### 3.2. Data Pipeline (`src/data`)

The data pipeline is designed for high throughput and memory efficiency using **Lazy Loading**.

-   **`LazySupervisedDataset`**:
    -   **Input**: A JSON file containing conversation turns and image paths.
    -   **Lazy Loading**: Images are only opened (`PIL.Image.open`) inside `__getitem__`, preventing RAM explosion.
    -   **Robustness**: If an image is corrupt or missing, it falls back to a black dummy image rather than crashing the training run.

-   **Tokenization & Masking Strategy**:
    -   **User Turns**: `<image>\nUser Query` $\to$ **MASKED** (Label = -100). The model does not learn to reproduce user input.
    -   **Assistant Turns**: `Response` $\to$ **UNMASKED**. The model learns to predict these tokens.
    -   **Image Tokens**: Placeholder tokens are inserted and **MASKED**.

### 3.3. Training Loop (`src/training`)

-   **`LlavaTrainer`**:
    -   Extends Hugging Face `Trainer`.
    -   **Artifact Management**: The standard Trainer doesn't know about our custom `mm_projector`. We override `_save` to explicitly save the projector weights (`mm_projector.bin`) alongside the LoRA adapter. This is crucial for inference later.

## 4. Key Technologies

-   **Hydra**: For hierarchical configuration management. Allows composing experiments (e.g., `experiment=cloud_scale`) from modular config files.
-   **QLoRA (Peft + BitsAndBytes)**: Enables training large models (7B+) on consumer hardware (24GB VRAM) by loading the base model in 4-bit precision and training a Low-Rank Adapter.
-   **DeepSpeed ZeRO-2**: Used in cloud configurations to shard optimizer states across GPUs, reducing VRAM usage further.
-   **WandB**: Integrated for tracking training loss and potentially logging generated image captions during validation.

## 5. File Structure Reference

```text
src/
├── models/
│   ├── llava_arch.py       # Core modeling logic & Mixin
│   ├── projector.py        # MLP Adapter
│   └── encoder.py          # CLIP wrapper
├── data/
│   ├── dataset.py          # LazySupervisedDataset & Tokenization
│   └── collator.py         # Batch padding logic
├── training/
│   ├── trainer.py          # Custom save logic
│   └── callbacks.py        # WandB visualizers
└── utils/
    └── config_schema.py    # Type definitions for Hydra
```
