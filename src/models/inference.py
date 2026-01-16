import torch
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig
from src.models.llava_arch import LlavaLlamaForCausalLM, LlavaConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

class LlavaInference:
    """
    Helper class for inference with LLaVA model.
    """
    def __init__(
        self,
        model_path: str,
        adapter_path: str = None,
        load_in_4bit: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization Config
        quantization_config = None
        if load_in_4bit and device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Load Model
        logger.info(f"Loading base model from {model_path}...")
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        # Load Adapter if provided
        if adapter_path:
            logger.info(f"Loading adapter from {adapter_path}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            
            # Load Projector if saved separately (common in some pipelines, but PEFT saves modules_to_save)
            # If we added "mm_projector" to modules_to_save, it's inside the adapter weights usually.
            # But sometimes it's separate. We'll assume PEFT handles it if modules_to_save was set correctly.
            
        if device == "cpu":
            self.model.to(device)
            
        self.model.eval()
        
        # Image Processor
        self.vision_tower = self.model.get_vision_tower()
        self.image_processor = self.vision_tower.image_processor

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response for a given image and prompt.
        """
        # Prepare inputs
        formatted_prompt = f"User: <image>\n{prompt}\nAssistant:"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        
        pixel_values = None
        if image:
            pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(self.device).to(self.model.dtype)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up prompt from output
        # (Naive split)
        if "Assistant:" in decoded:
            response = decoded.split("Assistant:")[-1].strip()
        else:
            response = decoded
            
        return response
