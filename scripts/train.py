import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer, TrainingArguments, set_seed

from src.models.llava_arch import LlavaLlamaForCausalLM
from src.data.dataset import LazySupervisedDataset
from src.data.collator import DataCollatorForSupervisedDataset
from src.training.trainer import LlavaTrainer
from src.training.callbacks import WandbImageGenerationCallback, SpecificLoggingCallback
from src.utils.logging import get_logger

logger = get_logger(__name__)

def validate_config(cfg: DictConfig):
    """Validates critical configuration parameters to fail fast."""
    if not os.path.exists(cfg.data.data_path):
        raise FileNotFoundError(f"Data path not found: {cfg.data.data_path}")
    
    if not os.path.exists(cfg.data.image_folder):
        raise FileNotFoundError(f"Image folder not found: {cfg.data.image_folder}")

    if cfg.model.get("use_qlora", False) and not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a GPU, but torch.cuda.is_available() is False.")

def setup_environment(cfg: DictConfig):
    """Sets up randomness and logging environments."""
    # 1. Reproducibility
    if cfg.get("seed"):
        set_seed(cfg.seed)
        
    # 2. WandB Setup
    if cfg.get("wandb"):
        if cfg.wandb.get("project"):
            os.environ["WANDB_PROJECT"] = cfg.wandb.project
        if cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        if cfg.wandb.get("run_name"):
            os.environ["WANDB_NAME"] = cfg.wandb.run_name
        else:
            os.environ["WANDB_NAME"] = cfg.experiment_name

    # 3. Device Info
    if torch.cuda.is_available():
        logger.info(f"CUDA Available. Device Count: {torch.cuda.device_count()}")
        logger.info(f"Current Device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available. Training on CPU/MPS.")

def load_components(cfg: DictConfig):
    """Loads tokenizer and model."""
    logger.info(f"Loading tokenizer: {cfg.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path,
        use_fast=cfg.model.get("use_fast_tokenizer", True),
        padding_side="right" # Required for generation tasks
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loading model via Hydra: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)
    
    return tokenizer, model

def apply_model_adaptations(model, cfg: DictConfig):
    """Applies freezing and PEFT/QLoRA adaptations."""
    # 1. Freeze Vision Tower
    if cfg.model.get("freeze_vision_tower", False):
        logger.info("Freezing Vision Tower...")
        vision_tower = model.get_vision_tower()
        if vision_tower:
            for param in vision_tower.parameters():
                param.requires_grad = False
    
    # 2. QLoRA / PEFT Setup
    use_qlora = cfg.model.get("use_qlora", False)
    if use_qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=cfg.training.get("gradient_checkpointing", False)
        )
        
        peft_config = LoraConfig(
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            target_modules=OmegaConf.to_container(cfg.model.lora_target_modules, resolve=True),
            lora_dropout=cfg.model.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["mm_projector"]
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        # Non-PEFT Observability
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable Params: {trainable_params:,d} || All Params: {all_params:,d} || Trainable%: {100 * trainable_params / all_params:.2f}%")
        
    # 3. Gradient Checkpointing Helper (Non-PEFT)
    if cfg.training.get("gradient_checkpointing", False) and not use_qlora:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
    return model

def prepare_data(cfg: DictConfig, tokenizer, model):
    """Loads and splits the dataset."""
    logger.info("Loading dataset...")
    
    full_dataset = LazySupervisedDataset(
        data_path=cfg.data.data_path,
        image_folder=cfg.data.image_folder,
        tokenizer=tokenizer,
        image_processor=model.get_vision_tower().image_processor,
        data_args=cfg.data # Pass full data config for flexibility
    )
    
    eval_size = max(1, int(len(full_dataset) * 0.01))
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])
    
    logger.info(f"Dataset split: Train={train_size}, Eval={eval_size}")
    
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        image_size=cfg.data.get("image_size", 336)
    )
    
    return train_dataset, eval_dataset, collator

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    try:
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        
        validate_config(cfg)
        setup_environment(cfg)
        
        tokenizer, model = load_components(cfg)
        model = apply_model_adaptations(model, cfg)
        
        train_dataset, eval_dataset, collator = prepare_data(cfg, tokenizer, model)
        
        # Callbacks
        callbacks = [SpecificLoggingCallback()]
        if cfg.training.get("report_to") == "wandb":
            wandb_callback = WandbImageGenerationCallback(
                tokenizer=tokenizer,
                image_processor=model.get_vision_tower().image_processor,
                eval_dataset=eval_dataset,
                num_samples=2
            )
            callbacks.append(wandb_callback)
        
        logger.info("Initializing Trainer...")
        training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
        training_args = TrainingArguments(**training_cfg)
        
        trainer = LlavaTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks
        )
        
        logger.info("Starting training...")
        if list(model.parameters())[0].device.type == "meta":
            logger.warning("Model is on meta device! This might indicate initialization issues.")
            
        trainer.train()
        trainer.save_model()
        
        logger.info("Training complete.")
        
    except Exception as e:
        logger.exception("Training failed with an exception!")
        sys.exit(1)

if __name__ == "__main__":
    main()
