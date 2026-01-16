# Scalable Visual Instruction Tuning Recipes

A production-ready pipeline for training LLaVA (Large Language and Vision Assistant) models, designed to scale from local consumer GPUs to cloud clusters.

## Features

- **Local-Lite Profile**: Train on single 24GB GPU using 4-bit QLoRA and Gradient Checkpointing.
- **Cloud-Scale Profile**: Scale to multi-GPU clusters using DeepSpeed ZeRO-2.
- **Modular Architecture**: Clean separation of configs, model code, and training logic.
- **Experiment Tracking**: Integrated WandB logging and image generation callbacks.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   - Download COCO 2017 images to `data/coco/train2017`.
   - Place LLaVA instruction JSON in `data/coco/llava_instruct_150k.json`.
   - Update `configs/data/default.yaml` with your paths if different.

## Training

### Local Debugging (Single GPU)
Run a quick test on a single GPU to verify the pipeline.
```bash
python scripts/train.py experiment=local_lite
```

### Full Training (Cloud Scale)
Run distributed training using DeepSpeed.
```bash
accelerate launch scripts/train.py experiment=cloud_scale
```

## Inference & Demo

Launch the Gradio web interface to interact with your trained model.
```bash
python scripts/demo.py --base_model <path_to_base_llama> --adapter <path_to_checkpoint>
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration.
- `configs/config.yaml`: Main entry point.
- `configs/experiment/`: Presets for different scenarios.
- `configs/deepspeed/`: DeepSpeed configurations.

## Development

- **Linting**: `ruff check .`
- **Tests**: `pytest tests/`
