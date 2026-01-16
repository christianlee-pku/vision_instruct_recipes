from typing import List, Optional, Tuple, Union, Dict, Any
import os

import torch
import torch.nn as nn

from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import cached_file

from src.models.encoder import CLIPVisionTower
from src.models.projector import LlavaMultiModalProjector
from src.utils.logging import get_logger

logger = get_logger(__name__)

LORA_ARGUMENTS = [
    "lora_r", 
    "lora_alpha", 
    "lora_dropout", 
    "lora_target_modules", 
    "lora_modules_to_save", 
    "lora_bias"
]

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"

    def __init__(
        self,
        vision_tower: str = "openai/clip-vit-large-patch14-336",
        mm_vision_select_layer: int = -2,
        mm_vision_select_feature: str = "patch",
        mm_hidden_size: int = 1024,
        unfreeze_mm_vision_tower: bool = False,
        pretrain_mm_mlp_adapter: Optional[str] = None,
        **kwargs,
    ):
        self.vision_tower = vision_tower
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_hidden_size = mm_hidden_size
        self.unfreeze_mm_vision_tower = unfreeze_mm_vision_tower
        self.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter
        super().__init__(**kwargs)

# ============================================================================
# Mixins for Multi-Modal Functionality
# ============================================================================

class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        
        if hasattr(config, "vision_tower") and config.vision_tower:
            self.vision_tower = CLIPVisionTower(config.vision_tower, delay_load=True, args=config)
            self.mm_projector = LlavaMultiModalProjector(config)

    def get_vision_tower(self):
        return getattr(self, 'vision_tower', None)

class LlavaMetaForCausalLM:
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_patches = image_features.shape[1]
            if (cur_input_ids == self.config.pad_token_id).sum() == 0:
                # No image tokens, just use text
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue

            # Found image placeholders
            indices = (cur_input_ids == self.config.pad_token_id).nonzero(as_tuple=True)[0]
            
            # Use the robust vectorized search we implemented earlier
            n = num_patches
            limit = len(indices) - n + 1
            if limit > 0:
                upper, lower = indices[n-1:], indices[:limit]
                matches = (upper - lower == (n - 1)).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    start_idx = indices[matches[0]]
                    cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                    
                    prefix = cur_input_embeds[:start_idx]
                    suffix = cur_input_embeds[start_idx + n:]
                    
                    fused = torch.cat([prefix, image_features[batch_idx], suffix], dim=0)
                    
                    # Align sequence length
                    if fused.shape[0] > cur_input_ids.shape[0]:
                        fused = fused[:cur_input_ids.shape[0]]
                    elif fused.shape[0] < cur_input_ids.shape[0]:
                        fused = torch.cat([fused, cur_input_embeds[fused.shape[0]:]], dim=0)
                    
                    new_input_embeds.append(fused)
                    if labels is not None:
                        new_labels.append(labels[batch_idx])
                else:
                    logger.warning(f"Batch {batch_idx}: contiguous block not found")
                    new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                    if labels is not None: new_labels.append(labels[batch_idx])
            else:
                logger.warning(f"Batch {batch_idx}: not enough pad tokens")
                new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None: new_labels.append(labels[batch_idx])

        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        if labels is not None:
            new_labels = torch.stack(new_labels, dim=0)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

# ============================================================================
# Model Implementation with Mixins
# ============================================================================

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        # We must call LlamaPreTrainedModel.__init__ directly to skip LlamaForCausalLM.__init__ 
        # which would instantiate a standard LlamaModel.
        # The parent of LlamaForCausalLM is LlamaPreTrainedModel.
        # Using super(LlamaForCausalLM, self) correctly bypasses LlamaForCausalLM.__init__.
        super(LlamaForCausalLM, self).__init__(config)
        
        self.model = LlavaLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        self._check_and_load_projector(config)

    def _check_and_load_projector(self, config):
        projector_path = getattr(config, "pretrain_mm_mlp_adapter", None)
        if projector_path:
            logger.info(f"Loading projector from {projector_path}")
            try:
                if not os.path.exists(projector_path):
                    for filename in ["mm_projector.bin", "pytorch_model.bin", "adapter_model.bin"]:
                        try:
                            resolved = cached_file(projector_path, filename)
                            if resolved:
                                projector_path = resolved
                                break
                        except: continue
                
                if os.path.exists(projector_path):
                    weights = torch.load(projector_path, map_location="cpu")
                    cleaned = {k.split("mm_projector.")[-1]: v for k, v in weights.items() if "mm_projector" in k} or weights
                    self.model.mm_projector.load_state_dict(cleaned, strict=False)
                    self.model.mm_projector.to(dtype=self.dtype)
                    logger.info("Projector weights loaded successfully.")
            except Exception as e:
                logger.error(f"Projector load failed: {e}")
                raise e

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images if images is not None else pixel_values
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        images = kwargs.get("images", None)
        pixel_values = kwargs.get("pixel_values", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if pixel_values is not None:
            _inputs['pixel_values'] = pixel_values
        return _inputs

# ============================================================================
# Factory
# ============================================================================

def LlavaModel(model_name_or_path: str, **kwargs):
    use_qlora = kwargs.pop("use_qlora", False)
    for k in LORA_ARGUMENTS + ["freeze_vision_tower", "unfreeze_mm_vision_tower"]:
        if k in kwargs: kwargs.pop(k)
    
    from transformers import AutoConfig
    base_config = AutoConfig.from_pretrained(model_name_or_path)
    
    llava_params = {
        "vision_tower": kwargs.get("vision_tower"),
        "mm_vision_select_layer": kwargs.get("mm_vision_select_layer"),
        "mm_vision_select_feature": kwargs.get("mm_vision_select_feature"),
        "mm_hidden_size": kwargs.get("mm_hidden_size"),
        "unfreeze_mm_vision_tower": kwargs.get("unfreeze_mm_vision_tower"),
        "pretrain_mm_mlp_adapter": kwargs.get("pretrain_mm_mlp_adapter"),
    }
    
    config_dict = base_config.to_dict()
    config_dict.update({k: v for k, v in llava_params.items() if v is not None})
    llava_config = LlavaConfig.from_dict(config_dict)
    
    quantization_config = None
    load_in_4bit = kwargs.get("load_in_4bit", False)
    load_in_8bit = kwargs.get("load_in_8bit", False)
    
    if (load_in_4bit or load_in_8bit) and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=kwargs.get("bnb_4bit_use_double_quant", False),
            bnb_4bit_quant_type=kwargs.get("bnb_4bit_quant_type", "nf4"),
        )
    
    for k in ["bnb_4bit_use_double_quant", "bnb_4bit_quant_type", "load_in_4bit", "load_in_8bit"]:
        if k in kwargs: kwargs.pop(k)

    return LlavaLlamaForCausalLM.from_pretrained(
        model_name_or_path, 
        config=llava_config,
        quantization_config=quantization_config,
        use_safetensors=True, 
        low_cpu_mem_usage=((load_in_4bit or load_in_8bit) and torch.cuda.is_available()),
        **kwargs
    )
