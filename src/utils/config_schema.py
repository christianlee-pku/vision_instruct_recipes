from dataclasses import dataclass, field
from typing import List, Optional
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    _target_: str = "src.models.llava_arch.LlavaModel"
    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    vision_tower: str = "openai/clip-vit-large-patch14-336"
    freeze_vision_tower: bool = True
    use_qlora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

@dataclass
class DataConfig:
    data_path: str = "data/llava_instruct_150k.json"
    image_folder: str = "data/images"
    image_processor_name: str = "openai/clip-vit-large-patch14-336"
    lazy_preprocess: bool = True
    image_size: int = 336
    conversation_template: str = "llama_3"

@dataclass
class TrainingConfig:
    output_dir: str = "./checkpoints"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    logging_steps: int = 10
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "wandb"

@dataclass
class ExperimentConfig:
    defaults: List[dict] = field(default_factory=lambda: [
        {"model": "default"},
        {"data": "default"},
        {"training": "default"}
    ])
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "llava_tuning"
    seed: int = 42

def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=ExperimentConfig)
    cs.store(group="model", name="default", node=ModelConfig)
    cs.store(group="data", name="default", node=DataConfig)
    cs.store(group="training", name="default", node=TrainingConfig)
