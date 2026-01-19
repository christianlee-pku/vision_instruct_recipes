# 🌋 Vision Instruct Recipes

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-ready pipeline for training **LLaVA (Large Language and Vision Assistant)** models, designed to seamlessly scale from **local consumer GPUs** to **cloud clusters**.

## ✨ Features

- **🚀 Scalable Training**: Seamlessly switch between local debugging (single GPU) and cloud-scale training (multi-GPU with DeepSpeed ZeRO-2).
- **💡 Efficient Fine-Tuning**: Built-in support for **4-bit QLoRA** and **Gradient Checkpointing**, enabling training on consumer cards (e.g., RTX 3090/4090).
- **🛠️ Modular Design**: Clean separation of model architecture, data loading, and training logic using Hydra configuration.
- **📊 Experiment Tracking**: Integrated with **WandB** for real-time loss tracking and visual generation callbacks.

## 🏗️ Pipeline Overview

```mermaid
graph TD
    A[Configuration (Hydra)] --> B{Model Loading};
    B -->|LLaVA Arch| C[Base LLM (Llama-3)];
    B -->|Vision Mixin| D[Vision Tower (CLIP)];
    B -->|Adapter| E[Projector (MLP)];
    
    F[Data Processing] -->|Lazy Loading| G[Tokenization & Masking];
    G --> H[Training Loop];
    
    C & D & E --> H;
    
    H -->|Forward Pass| I[Multimodal Fusion];
    H -->|Backward Pass| J[Gradient Updates (QLoRA)];
    
    J --> K[Checkpointing];
    K --> L[WandB Logging];
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/christianlee-pku/vision_instruct_recipes.git
   cd vision_instruct_recipes
   ```

2. **Create a Conda environment**
   ```bash
   conda create -n llava python=3.10
   conda activate llava
   ```

3. **Install dependencies**
   
   - **For Cloud/GPU Users (Recommended)**:
     Optimized for Tesla T4 (CUDA 12.4) and QLoRA stability.
     ```bash
     # Clean existing env to avoid conflicts
     pip uninstall -y torch torchvision torchaudio bitsandbytes accelerate transformers peft
     
     # Install GPU-optimized stack
     pip install -r requirements_gpu.txt
     ```

   - **For Local/CPU Users (Debugging)**:
     Lightweight setup for code verification on laptops.
     ```bash
     pip install -r requirements_cpu.txt
     ```

## 🚀 Usage

### 1. Data Preparation
1. Download the COCO 2017 dataset images to `data/coco/images`.
2. Place your instruction tuning JSON file (e.g., `llava_instruct_150k.json`) in `data/coco/`.
3. Update `configs/data/default.yaml` if your paths differ.

### 2. Local Debugging (Lite Mode)
Run a quick training loop on a single GPU (or CPU) to verify the pipeline. This profile uses QLoRA and aggressive memory optimization.

```bash
# Run using the provided script
bash scripts/train_local.sh

# Or directly with python
python scripts/train.py experiment=local_lite
```

### 3. Cloud Scale Training
Run full-scale distributed training using DeepSpeed ZeRO-2.

```bash
# Run using the provided script
bash scripts/train_cloud.sh

# Or using accelerate directly
accelerate launch scripts/train.py experiment=cloud_scale
```

### 4. Interactive Demo
Launch a Gradio web interface to chat with your trained model.

```bash
python scripts/demo.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --adapter ./checkpoints/local_lite/checkpoint-final
```

## 📂 Project Structure

```text
vision_instruct_recipes/
├── configs/             # Hydra configuration files
│   ├── experiment/      # Experiment profiles (local_lite, cloud_scale)
│   └── ...
├── data/                # Dataset storage (not committed)
├── scripts/             # Execution scripts (train, demo)
├── src/                 # Source code
│   ├── models/          # LLaVA architecture & components
│   ├── data/            # Dataset & Collator logic
│   └── training/        # Trainer & Callbacks
└── tests/               # Unit tests
```

## 📜 License

This project is licensed under the Apache 2.0 License.

## 🖊️ Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{vision_instruct_recipes,
  author = {Christian Lee},
  title = {Vision Instruct Recipes: Scalable LLaVA Training Pipeline},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/christianlee-pku/vision_instruct_recipes}}
}
```
