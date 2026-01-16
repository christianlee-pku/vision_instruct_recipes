import json
import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from PIL import Image
from transformers import PreTrainedTokenizer

from src.data.entities import InstructionSample
from src.utils.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN_LENGTH = 576 # CLIP ViT-L-336

# Conversation Templates
CONVERSATION_TEMPLATES = {
    "llama_3": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "separator": "" # Templates include their own separators/special tokens
    },
    "vicuna_v1": {
        "system": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        "user": "USER: {content}",
        "assistant": "ASSISTANT: {content}</s>",
        "separator": " "
    },
    "plain": {
        "system": "",
        "user": "USER: {content}",
        "assistant": "ASSISTANT: {content}",
        "separator": "\n"
    }
}

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning that lazily loads images."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        image_folder: str,
        image_processor: Optional[Any] = None,
        data_args: Optional[Any] = None,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.data_args = data_args
        self.list_data_dict = self._load_data(data_path)
        
        # Determine template
        self.template_name = getattr(data_args, "conversation_template", "llama_3")
        if self.template_name not in CONVERSATION_TEMPLATES:
            logger.warning(f"Template {self.template_name} not found, falling back to 'plain'.")
            self.template_name = "plain"
        self.template = CONVERSATION_TEMPLATES[self.template_name]
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Loaded {len(self.list_data_dict)} samples from {data_path}")

    def _load_data(self, data_path: str) -> List[InstructionSample]:
        """Load data from JSON file."""
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
            return [InstructionSample.from_json(item) for item in data]
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise e

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Retrieves the i-th sample from the dataset and processes it for training.
        
        Steps:
        1. Load the image from disk and process it into a tensor (pixel_values).
        2. Tokenize the conversation text, applying masking (for User turns) and inserting image placeholder tokens.
        3. Add special tokens (BOS/EOS) if necessary.
        4. Truncate sequences to the model's maximum length.
        """
        sample = self.list_data_dict[i]
        
        # 1. Load and process image
        image = self._load_image_robust(sample.image)
        pixel_values = self._process_image(image, sample.image)

        # 2. Tokenize conversation with masking and image expansion
        input_ids, labels = self._tokenize_conversation(
            sample.conversations, 
            has_image=(image is not None)
        )

        # 3. Add special tokens (BOS/EOS) safely
        input_ids, labels = self._add_special_tokens(input_ids, labels)

        # 4. Truncation
        max_len = getattr(self.tokenizer, "model_max_length", 2048)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values
        }
    
    def _load_image_robust(self, image_file: str):
        """Robust image loading with fallback."""
        if not image_file:
            return None
            
        image_path = os.path.join(self.image_folder, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}. Using dummy black image.")
            # Fallback based on configured size
            size = getattr(self.data_args, "image_size", 336)
            return Image.new('RGB', (size, size), (0, 0, 0))

    def _process_image(self, image: Optional[Image.Image], image_name: str) -> torch.Tensor:
        """
        Processes the PIL image into a tensor using the image processor (e.g., CLIP).
        Handles resizing if necessary and provides a fallback tensor if processing fails.
        """
        target_size = getattr(self.data_args, "image_size", 336)
        
        if image and self.image_processor:
            try:
                # Resize image if it does not match target size
                if image.size != (target_size, target_size):
                    image = image.resize((target_size, target_size), resample=Image.BICUBIC)
                    
                return self.image_processor(image, return_tensors="pt")["pixel_values"][0]
            except Exception as e:
                logger.error(f"Image processing failed for {image_name}: {e}")
        
        # Fallback
        return torch.zeros((3, target_size, target_size))

    def _tokenize_conversation(self, conversations: List[Any], has_image: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes the conversation turns into input_ids and labels.
        
        Key Logic:
        1. Iterates through each turn (User/Assistant).
        2. Applies the conversation template (e.g., Llama-3 format).
        3. Enforces that the <image> token appears at the start of the first user turn.
        4. Splits the text by <image> to insert the expanded image tokens (placeholders).
        5. Generates `labels` for masking:
           - User turns are fully masked (IGNORE_INDEX).
           - Assistant turns are unmasked (model learns to predict them).
           - Image tokens are masked.
        """
        input_ids_list = []
        labels_list = []
        
        # Add System Prompt if exists (masked)
        if "system" in self.template and self.template["system"]:
            sys_text = self.template["system"] + self.template["separator"]
            sys_ids = self.tokenizer(sys_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
            input_ids_list.append(sys_ids)
            labels_list.append(torch.full_like(sys_ids, IGNORE_INDEX))

        for idx, turn in enumerate(conversations):
            role = turn.from_role
            content = turn.value
            
            # Prepare formatting
            if role == "human":
                # Enforce image token placement INSIDE content
                if idx == 0 and has_image:
                    # Ensure <image> tag is at the start of the first user turn's content
                    content = content.replace("<image>", "").strip()
                    content = "<image>\n" + content
                
                # Apply User Template
                formatted_text = self.template["user"].format(content=content)
                should_mask_content = True
            else:
                # Apply Assistant Template
                formatted_text = self.template["assistant"].format(content=content)
                should_mask_content = False

            # Add separator
            formatted_text += self.template["separator"]

            # 1. Split by <image> for expansion
            parts = formatted_text.split("<image>")
            
            for i, part in enumerate(parts):
                if part:
                    c_ids = self.tokenizer(part, add_special_tokens=False, return_tensors="pt").input_ids[0]
                    input_ids_list.append(c_ids)
                    
                    # Apply masking based on role
                    labels_list.append(torch.full_like(c_ids, IGNORE_INDEX) if should_mask_content else c_ids.clone())

                # Insert Image Expansion Placeholders
                if i < len(parts) - 1:
                    placeholder_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                    img_ids = torch.full((DEFAULT_IMAGE_TOKEN_LENGTH,), placeholder_id, dtype=torch.long)
                    
                    input_ids_list.append(img_ids)
                    # Image tokens are always masked
                    labels_list.append(torch.full((DEFAULT_IMAGE_TOKEN_LENGTH,), IGNORE_INDEX, dtype=torch.long))

        if not input_ids_list:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
            
        return torch.cat(input_ids_list), torch.cat(labels_list)

    def _add_special_tokens(self, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Safely adds BOS (Beginning Of Sentence) and EOS (End Of Sentence) tokens to the sequences.
        
        Core Logic:
        1. BOS Handling: Prepends the BOS token to input_ids if missing. The corresponding label 
           is set to IGNORE_INDEX (-100) because the model should not be penalized for 
           predicting the start token.
        2. EOS Handling: Appends the EOS token to input_ids if missing. The corresponding label 
           is set to the EOS token ID, allowing the model to learn when to stop generating 
           (especially important after an Assistant turn).
        """
        # 1. Prepend BOS token if defined and not already present at the start
        if self.tokenizer.bos_token_id is not None:
            if len(input_ids) == 0 or input_ids[0] != self.tokenizer.bos_token_id:
                input_ids = torch.cat([torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long), input_ids])
                # Always mask BOS in labels so model isn't trained to predict it
                labels = torch.cat([torch.tensor([IGNORE_INDEX], dtype=torch.long), labels])
             
        # 2. Append EOS token if defined and not already present at the end
        if self.tokenizer.eos_token_id is not None:
            if len(input_ids) == 0 or input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)])
                # Keep EOS in labels as a target so model learns the stopping condition
                labels = torch.cat([labels, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)])
        
        return input_ids, labels
