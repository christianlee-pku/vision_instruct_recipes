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

# 3. Launch Training
# 使用 accelerate launch 自动处理多卡或单卡逻辑
# 默认使用 t4_qlora 配置，因为它在 16GB 环境下最稳健
echo "Launching training via Accelerate..."
accelerate launch scripts/train.py \
    experiment=t4_qlora \
    training.report_to="wandb" \
    hydra.run.dir="outputs/cloud/$(date +%Y-%m-%d_%H-%M-%S)"

echo "Training Process Dispatched."
