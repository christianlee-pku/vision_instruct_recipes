#!/bin/bash

# Cloud Training Script (Production/Scale)
# Optimized for T4 GPU or GPU clusters.
# Usage: bash scripts/train_cloud.sh

echo "Initializing Cloud Training Pipeline..."

# 1. Environment Check
if ! command -v accelerate &> /dev/null;
then
    echo "Error: 'accelerate' not found. Please run 'pip install accelerate' first."
    exit 1
fi

# 2. Config & Env Setup
export WANDB_MODE=online
export PYTHONPATH=.
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 3. Pre-download Models to Local Directory
echo "📥 Pre-downloading models to ./model_cache/ "
python scripts/download_models.py

# Read paths from the generated env file
if [ -f "model_paths.env" ]; then
    export $(cat model_paths.env | xargs)
else
    # Fallback paths if python script didn't generate env file
    export LOCAL_CLIP_PATH="$(pwd)/model_cache/clip-vit-large-patch14-336"
    # Note: We need to override the model_name_or_path in the config too if it differs
fi

echo "✅ Using Local Models:"
echo "   CLIP: $LOCAL_CLIP_PATH"

# 4. Launch Training
# Override model paths to point to local directories
echo "🚀 Launching training via Accelerate..."
accelerate launch scripts/train.py \
    experiment=t4_qlora \
    model.vision_tower="$LOCAL_CLIP_PATH" \
    training.report_to="wandb" \
    hydra.run.dir="outputs/cloud/$(date +%Y-%m-%d_%H-%M-%S)"

echo "Training Process Dispatched."