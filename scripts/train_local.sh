#!/bin/bash

# Local Training Script (Debug/Dev)
# Usage: ./scripts/train_local.sh

echo "Starting Local Training (Lite Mode)..."

# Set WandB to offline to avoid cloud sync during debug
export WANDB_MODE=offline
export WANDB_PROJECT="vision-instruct-recipes-local"

# Ensure python path includes current directory
export PYTHONPATH=.

# Run training with local_lite configuration
# We disable QLoRA by default for local CPU debugging to avoid errors, 
# unless you have a GPU and want to test it (add model.use_qlora=true)
python scripts/train.py \
    experiment=local_lite \
    training.report_to="wandb" \
    model.use_qlora=false \
    hydra.run.dir="outputs/local/$(date +%Y-%m-%d_%H-%M-%S)"

echo "Local Training Finished. Check outputs/local/ for logs."

