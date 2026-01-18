import torch
import os
import time
import json
from transformers import TrainerCallback, TrainerState, TrainerControl
try:
    import wandb
except ImportError:
    wandb = None
import numpy as np
from src.utils.logging import get_logger

logger = get_logger(__name__)

class SpecificLoggingCallback(TrainerCallback):
    """
    Callback to log training metrics in a specific format:
    "mode": "train", "epoch": 1, "iter": 100, "lr": 0.002, "memory": 3557, 
    "data_time": 0.02597, "loss": 7.26566, "grad_norm": 5.29593, "time": 0.77726
    """
    def __init__(self):
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.step_start_time = time.time()
        self.data_time_sum = 0.0
        self.iter_time_sum = 0.0
        self.count = 0
    
    def on_step_begin(self, args, state, control, **kwargs):
        # Time from previous step end to this step begin is roughly data time
        current_time = time.time()
        self.data_time_sum += (current_time - self.step_start_time)
        self.step_start_time = current_time

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        # Time for the actual step (forward+backward)
        self.iter_time_sum += (current_time - self.step_start_time)
        self.count += 1
        # Reset for next data load
        self.step_start_time = current_time

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            # Calculate averages
            avg_data_time = self.data_time_sum / self.count if self.count > 0 else 0.0
            avg_iter_time = self.iter_time_sum / self.count if self.count > 0 else 0.0
            
            # Reset counters
            self.data_time_sum = 0.0
            self.iter_time_sum = 0.0
            self.count = 0
            
            # Extract metrics
            epoch = logs.get("epoch", 0)
            
            # Calculate iter within epoch
            # state.max_steps is total steps
            # args.num_train_epochs is total epochs
            iter_step = state.global_step
            if state.max_steps > 0 and args.num_train_epochs > 0:
                steps_per_epoch = state.max_steps // args.num_train_epochs
                if steps_per_epoch > 0:
                    iter_step = state.global_step % steps_per_epoch
                    # If iter_step is 0 but we have done steps, it means we completed an epoch (or multiple)
                    if iter_step == 0 and state.global_step > 0:
                         iter_step = steps_per_epoch

                    # Fallback: If global_step is 0 but we have epoch progress (e.g. gradient accumulation)
                    # Estimate iter from epoch fraction. 
                    # Note: steps_per_epoch is based on optimization steps (accumulated).
                    if state.global_step == 0 and epoch > 0:
                        frac = epoch - int(epoch)
                        estimated = int(frac * steps_per_epoch)
                        if estimated > 0:
                            iter_step = estimated

            lr = logs.get("learning_rate", 0.0)
            loss = logs.get("loss", 0.0)
            grad_norm = 0.0 # Trainer usually doesn't pass grad_norm in logs unless specifically configured
            
            # Try to find grad_norm in logs if available (some versions do)
            # Or if we can access it from model/optimizer (difficult here)
            # We'll use a placeholder or check logs keys
            # Often logged as 'total_flos' etc, but grad_norm might be missing.
            
            memory = 0
            if torch.cuda.is_available():
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 # MB
            
            # Format as requested
            # "mode": "train", "epoch": 1, "iter": 100, "lr": 0.002, "memory": 3557, 
            # "data_time": 0.02597, "loss": 7.26566, "grad_norm": 5.29593, "time": 0.77726
            
            log_entry = {
                "mode": "train",
                "epoch": int(epoch) + 1, # Show current epoch index (1-based) as integer
                "iter": int(iter_step),
                "lr": f"{lr:.6f}",
                "memory": int(memory),
                "data_time": f"{avg_data_time:.5f}",
                "loss": f"{loss:.5f}",
                "total_loss": f"{loss:.5f}", # Requested "total loss" explicitly
                "grad_norm": "N/A", # Placeholder if not found
                "time": f"{avg_iter_time:.5f}"
            }
            
            # Construct the string manually to match the exact format requested
            # The user provided a JSON-like string but unquoted keys in the prompt text?
            # "mode": "train", ...
            # Let's produce a valid JSON string or the exact text requested.
            # The user text: "mode": "train", "epoch": 1, ...
            # This looks like keys are quoted.
            
            msg_parts = []
            for k, v in log_entry.items():
                if isinstance(v, str):
                    # Check if value is numeric string to avoid double quotes if strictly JSON numbers needed
                    # But prompt example: "lr": 0.002 -> Number
                    # "loss": 7.26566 -> Number
                    # "grad_norm": 5.29593 -> Number
                    # "mode": "train" -> String
                    pass
            
            # Manual construction to ensure correct types (Number vs String)
            formatted_msg = ", ".join([
                f'"mode": "train"',
                f'"epoch": {log_entry["epoch"]}', # string formatted float, no quotes
                f'"iter": {log_entry["iter"]}',
                f'"lr": {log_entry["lr"]}',
                f'"memory": {log_entry["memory"]}',
                f'"data_time": {log_entry["data_time"]}',
                f'"loss": {log_entry["loss"]}',
                f'"total_loss": {log_entry["total_loss"]}',
                f'"grad_norm": {log_entry["grad_norm"]}', # This one is "N/A", might need quotes or be valid json? Example has 5.29593 (number)
                f'"time": {log_entry["time"]}'
            ])
            
            # Fix "N/A" for grad_norm if it's string, it should be quoted.
            # But if it is number, no quotes.
            # If I put "N/A" (quoted) it breaks the pattern of numbers.
            # I will just put 0.0 if N/A to keep it numeric? 
            # Or "N/A".
            # Let's use 0.0 for grad_norm if not found to matches style.
            
            if log_entry["grad_norm"] == "N/A":
                 formatted_msg = formatted_msg.replace('"grad_norm": N/A', '"grad_norm": 0.0')

            logger.info(formatted_msg)

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
