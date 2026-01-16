# Vision Instruct Recipes Constitution
<!--
Sync Impact Report:
- Version change: 1.0.0 -> 1.1.0 (Material Expansion)
- Refined Principle I: Emphasized "Portfolio-Ready" and engineering maturity.
- Refined Principle II: Mandated explicit component isolation (Vision Encoder, Projector, LLM).
- Added Principle III: Resource-Efficient Engineering (Gradient Checkpointing, Data Masking).
- Templates requiring updates: None (Generic placeholders remain compatible).
-->

## Core Principles

### I. Industry Standard R&D & Portfolio Readiness
Development MUST demonstrate engineering maturity and deep architectural understanding suitable for public GitHub review and potential employers. Codebase quality, documentation, and structure must mimic production-grade "Industry Standard R&D" environments rather than academic scripts.

### II. Config-Driven & Modular Architecture
A strict "Configuration-Driven" design pattern using YAML/Hydra is MANDATORY to decouple hyperparameters from logic. The codebase MUST explicitly isolate components: **Vision Encoder**, **Projector**, and **LLM**. Implementations must use type-hinted, fully documented Python compatible with the Hugging Face ecosystem (Trainer, PEFT/QLoRA), avoiding ad-hoc scripts.

### III. Resource-Efficient Engineering
Pipelines MUST mimic real-world corporate constraints for scalability and efficiency. Mandatory implementation details include **gradient checkpointing**, **4-bit quantization** (BitsAndBytes), and proper **multi-modal data masking** to prevent training on prompt tokens.

### IV. Scalable & Reproducible Structure
Maintain strict separation: `configs/` for parameters, `src/` for modular source code, and `scripts/` for minimal entry points. Adherence to MLOps best practices (WandB tracking, deterministic execution) is required to ensure reproducibility.

## Technology Stack & Constraints

**Core Frameworks**: PyTorch, Hugging Face Transformers, PEFT, BitsAndBytes (QLoRA), Hydra (Config), WandB (Logging), Gradio (UI).

**Architectural Components**: Mini-LLaVA (Vision Encoder + Projector + LLM).

**Hardware Target**: Consumer GPUs (e.g., NVIDIA RTX 30/40 series).

**Code Style**: Strict Type-hinted Python 3.x, Black/Ruff formatting, Google-style docstrings.

## Development Workflow

1.  **Config First**: Define experiment parameters and architectural choices in YAML/Hydra.
2.  **Component Isolation**: Implement model components (Encoder, Projector) as distinct, testable modules in `src/`.
3.  **Efficiency Verification**: Verify gradient checkpointing and memory usage before full training runs.
4.  **Track & Validate**: Log all metrics to WandB; validate instruction tuning visually via Gradio.

## Governance

This constitution supersedes all other practices. Amendments require documentation, approval, and a migration plan.

All PRs must be reviewed for "Portfolio Readiness"—code that looks amateurish or script-like will be rejected. Complexity is permitted only when it serves modularity or efficiency.

**Version**: 1.1.0 | **Ratified**: 2025-12-30 | **Last Amended**: 2025-12-30
