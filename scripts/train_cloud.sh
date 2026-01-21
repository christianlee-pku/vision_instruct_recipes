#!/bin/bash

# Cloud Training Script (Production/Scale)
# Optimized for T4 GPU or GPU clusters.
# Usage: 
#   bash scripts/train_cloud.sh [experiment_name] [num_gpus]
# 
# Examples:
#   bash scripts/train_cloud.sh cloud_gpu_scale       (Default: All GPUs, DeepSpeed)
#   bash scripts/train_cloud.sh cloud_gpu_scale 4     (Use 4 GPUs)
#   bash scripts/train_cloud.sh t4_qlora              (Single T4, QLoRA)

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
# Default to cloud_gpu_scale if no experiment is provided as first arg
EXPERIMENT=${1:-"cloud_gpu_scale"}
NUM_GPUS=${2:-""}

# Construct accelerate arguments
ACCELERATE_ARGS=""
if [ -n "$NUM_GPUS" ]; then
    ACCELERATE_ARGS="--num_processes $NUM_GPUS"
    echo "⚙️  Configured for $NUM_GPUS GPUs"
else
    echo "⚙️  Auto-detecting all available GPUs"
fi

# Override model paths to point to local directories
echo "🚀 Launching training via Accelerate with experiment: $EXPERIMENT"
accelerate launch $ACCELERATE_ARGS scripts/train.py \
    experiment=$EXPERIMENT \
    model.vision_tower="$LOCAL_CLIP_PATH" \
    training.report_to="wandb"

echo "Training Process Dispatched."