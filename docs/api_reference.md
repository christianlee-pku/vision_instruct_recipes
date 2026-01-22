# API Reference

This document provides a high-level overview of the core modules and classes in `vision_instruct_recipes`.

## `src.models`

Core model architecture definitions.

### `llava_arch.LlavaLlamaForCausalLM`
The main LLaVA model class wrapping a Llama base model.
*   **Key Methods**:
    *   `forward()`: Handles multimodal inputs (images + text).
    *   `prepare_inputs_labels_for_multimodal()`: Merges image embeddings into text embeddings.

### `encoder.CLIPVisionTower`
Wrapper for the OpenAI CLIP Vision Transformer (ViT-L/14-336).
*   **Purpose**: Encodes raw images into feature vectors.

### `projector.LlavaMultiModalProjector`
A Multi-Layer Perceptron (MLP) with GELU activation.
*   **Purpose**: Projects CLIP image features to the same dimension as the LLM's text embeddings.

---

## `src.data`

Data loading and processing utilities.

### `dataset.LazySupervisedDataset`
A PyTorch Dataset that loads instruction data from a JSON file.
*   **Features**: Lazy loading (does not load all data into RAM), supports complex conversation formatting.

### `collator.DataCollatorForSupervisedDataset`
Handles batch creation and padding.
*   **Key Logic**: 
    *   Pads input sequences to the maximum length in the batch.
    *   Creates `attention_mask` and `labels`.
    *   Masks out user instructions in `labels` so the model only learns to generate the assistant's response.

---

## `src.training`

Training loop customizations.

### `trainer.LlavaTrainer`
Subclass of Hugging Face `transformers.Trainer`.
*   **Customization**:
    *   Overrides `save_model` to handle LoRA/Projector weight saving separately.
    *   Integrates WandB image generation callbacks.
