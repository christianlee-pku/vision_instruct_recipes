import torch
import os
from transformers import TrainerCallback, TrainerState, TrainerControl
try:
    import wandb
except ImportError:
    wandb = None
import numpy as np
from src.utils.logging import get_logger

logger = get_logger(__name__)

class WandbImageGenerationCallback(TrainerCallback):
    """
    Callback to generate and log images/responses to WandB periodically.
    """
    def __init__(self, tokenizer, image_processor, eval_dataset, num_samples=4, freq=1):
        """
        Args:
            tokenizer: The tokenizer for decoding.
            image_processor: The image processor for raw image access (if needed) or just use dataset.
            eval_dataset: The dataset to sample from.
            num_samples: Number of samples to generate.
            freq: Frequency of generation (every `freq` eval steps? or just always on eval).
                  Since this is on_evaluate, it runs every time evaluation runs.
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        
        # Select indices once
        if len(eval_dataset) > num_samples:
            self.sample_indices = np.random.choice(len(eval_dataset), num_samples, replace=False)
        else:
            self.sample_indices = list(range(len(eval_dataset)))
            
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """
        Run generation on selected samples and log to WandB.
        """
        if not args.report_to or "wandb" not in args.report_to:
            return

        if state.is_world_process_zero:
            logger.info("Generating samples for WandB logging...")
            model.eval()
            
            table = wandb.Table(columns=["Image", "Prompt", "Target", "Prediction"])
            
            # Access underlying dataset if it's wrapped (e.g., Subset)
            dataset = self.eval_dataset
            if hasattr(dataset, "dataset"):
                dataset = dataset.dataset
                
            for idx in self.sample_indices:
                try:
                    # Get raw sample
                    if not hasattr(dataset, "list_data_dict"):
                        logger.warning("Dataset does not have list_data_dict, skipping generation.")
                        break
                        
                    sample = dataset.list_data_dict[idx]
                    
                    # 1. Prepare Image
                    image_file = sample.image
                    image = None
                    pixel_values = None
                    
                    if image_file:
                        image_path = os.path.join(dataset.image_folder, image_file)
                        try:
                            from PIL import Image
                            image = Image.open(image_path).convert('RGB')
                            if self.image_processor:
                                pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
                                pixel_values = pixel_values.to(model.device).to(model.dtype)
                        except Exception as e:
                            logger.warning(f"Failed to load image {image_path}: {e}")
                            continue

                    # 2. Prepare Text (Prompt)
                    # Use the first user turn as prompt
                    prompt = ""
                    target = ""
                    if sample.conversations:
                        # Assuming first turn is Human
                        prompt = sample.conversations[0].value
                        if len(sample.conversations) > 1:
                            target = sample.conversations[1].value
                            
                    # Formatting (naive for now, should match training template)
                    # We'll just use a simple format or raw text if template unknown
                    # <image>\nUser: {prompt}\nAssistant:
                    formatted_prompt = f"User: {prompt}\nAssistant:"
                    if image is not None:
                        formatted_prompt = f"<image>\n{formatted_prompt}"
                        
                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
                    input_ids = inputs.input_ids.to(model.device)
                    
                    # 3. Generate
                    with torch.no_grad():
                        generate_kwargs = {
                            "input_ids": input_ids,
                            "max_new_tokens": 64,
                            "do_sample": True,
                            "temperature": 0.7,
                            "top_p": 0.9,
                        }
                        if pixel_values is not None:
                            generate_kwargs["pixel_values"] = pixel_values
                            
                        # Handle case where model might not accept pixel_values in generate directly
                        # (depends on how LlavaLlamaForCausalLM wraps it, but usually fine)
                        outputs = model.generate(**generate_kwargs)
                    
                    # 4. Decode
                    prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Simple post-processing to remove input prompt from prediction if needed
                    # (generate usually includes input)
                    if prediction.startswith(formatted_prompt.replace("<image>", "")): # <image> token special handling
                         # Rough check, tokenizer might handle <image> differently
                         pass
                    
                    # Log
                    table.add_data(
                        wandb.Image(image) if image else None,
                        formatted_prompt,
                        target,
                        prediction
                    )
                
                except Exception as e:
                    logger.error(f"Error during WandB generation for idx {idx}: {e}")
            
            wandb.log({"eval_generations": table})
            model.train()
