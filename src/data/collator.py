import torch
from dataclasses import dataclass
from typing import Dict, Sequence, List, Optional
from transformers import PreTrainedTokenizer

from src.utils.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    tokenizer: PreTrainedTokenizer
    ignore_index: int = IGNORE_INDEX
    image_size: Optional[int] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1. Safe extraction and conversion to Tensors
        input_ids = [
            torch.as_tensor(instance["input_ids"], dtype=torch.long) 
            for instance in instances
        ]
        labels = [
            torch.as_tensor(instance["labels"], dtype=torch.long) 
            for instance in instances
        ]
        
        # 2. Handle missing pad_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            # Fallback to EOS if PAD is missing
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        
        # 3. Dynamic Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        )
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_token_id),
        )

        # 4. Multi-modal handling with dynamic dummy image size
        if "pixel_values" in instances[0]:
            pixel_values = []
            
            # Find the target shape: first check Batch images, then configured size, then fallback
            target_shape = None
            for instance in instances:
                if instance.get("pixel_values") is not None:
                    target_shape = instance["pixel_values"].shape
                    break
            
            if target_shape is None:
                size = self.image_size if self.image_size is not None else 336
                target_shape = (3, size, size)

            for instance in instances:
                val = instance.get("pixel_values")
                if val is not None:
                    pixel_values.append(torch.as_tensor(val))
                else:
                    # Provide a dummy zero tensor matching the target shape
                    pixel_values.append(torch.zeros(*target_shape))
            
            batch["pixel_values"] = torch.stack(pixel_values)

        return batch
