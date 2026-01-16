import os
import torch
from transformers import Trainer
from src.utils.logging import get_logger

logger = get_logger(__name__)

class LlavaTrainer(Trainer):
    """
    Custom Trainer for LLaVA that handles specific saving logic for non-LoRA components 
    (like the projector) if they are being trained.
    """
    
    def _save(self, output_dir: str = None, state_dict=None):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        super()._save(output_dir, state_dict)
        
        # Save mm_projector separately for robustness and compatibility
        model = self.model
        # 1. Unwrap from DistributedDataParallel
        if hasattr(model, "module"):
            model = model.module
        # 2. Unwrap from PEFT (BaseModel wrapper)
        if hasattr(model, "base_model"):
            model = model.base_model
        
        # 3. Access inner model if wrapped in CausalLM
        if hasattr(model, "get_model"):
            model = model.get_model() # This should be LlavaLlamaModel

        if hasattr(model, "mm_projector"):
            logger.info(f"Saving mm_projector to {output_dir}")
            projector_path = os.path.join(output_dir, "mm_projector.bin")
            torch.save(model.mm_projector.state_dict(), projector_path)
        else:
            logger.warning("No mm_projector found in model. Skipping separate save.")

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        # Ensure 'pixel_values' is included if present
        if "pixel_values" not in self.label_names: 
            # This method usually sets signature based on model forward. 
            # Since we updated model forward, it should be fine.
            pass