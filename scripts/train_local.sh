#!/bin/bash

# Local Training Script (Debug/Dev)
# Usage: ./scripts/train_local.sh

echo "Starting Local Training (Lite Mode)..."

# Set WandB to offline to avoid cloud sync during debug
# export WANDB_MODE=offline
export WANDB_MODE=disabled
export WANDB_PROJECT="vision-instruct-recipes-local"

# Ensure python path includes current directory
export PYTHONPATH=.

# Run training with local_lite configuration
# We disable QLoRA and quantization by default for local CPU debugging
python scripts/train.py \
    experiment=local_lite \
    training.report_to="none" \
    model.use_qlora=false \
    model.load_in_4bit=false \
    model.load_in_8bit=false \
    training.bf16=false \
    training.fp16=false \
    training.tf32=false

echo "Local Training Finished. Check outputs/ for logs."

