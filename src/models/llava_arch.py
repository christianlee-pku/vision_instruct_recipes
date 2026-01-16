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

# Arguments to be filtered out before passing to base Llama init
LORA_ARGUMENTS = [
    "lora_r", 
    "lora_alpha", 
    "lora_dropout", 
    "lora_target_modules", 
    "lora_modules_to_save", 
    "lora_bias"
]

LLAVA_SPECIFIC_ARGS = [
    "vision_tower",
    "mm_vision_select_layer",
    "mm_vision_select_feature",
    "mm_hidden_size",
    "unfreeze_mm_vision_tower",
    "freeze_vision_tower",
    "pretrain_mm_mlp_adapter",
    "load_in_4bit",
    "load_in_8bit",
    "bnb_4bit_use_double_quant",
    "bnb_4bit_quant_type"
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
# Multi-Modal Components Logic
# ============================================================================

class LlavaMetaModel:
    """Mixin to add vision tower and projector to a model."""
    def initialize_vision_modules(self, config: LlavaConfig):
        if not hasattr(self, "vision_tower") or self.vision_tower is None:
            self.vision_tower = CLIPVisionTower(config.vision_tower, delay_load=True, args=config)
        
        if not hasattr(self, "mm_projector") or self.mm_projector is None:
            self.mm_projector = LlavaMultiModalProjector(config)

# ============================================================================
# Implementation Classes
# ============================================================================

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config: LlavaConfig, **kwargs):
        # We must call LlamaPreTrainedModel.__init__ directly to avoid LlamaForCausalLM's 
        # default LlamaModel instantiation.
        super(LlamaForCausalLM, self).__init__(config)
        
        self.model = LlavaLlamaModel(config)
        self.model.initialize_vision_modules(config)
        
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        self._check_and_load_projector(config)

    def _check_and_load_projector(self, config: LlavaConfig) -> None:
        projector_path = getattr(config, "pretrain_mm_mlp_adapter", None)
        if not projector_path:
            return

        logger.info(f"Loading projector from: {projector_path}")
        try:
            if not os.path.exists(projector_path):
                for filename in ["mm_projector.bin", "pytorch_model.bin"]:
                    try:
                        resolved = cached_file(projector_path, filename)
                        if resolved:
                            projector_path = resolved
                            break
                    except: continue
            
            if os.path.exists(projector_path):
                state_dict = torch.load(projector_path, map_location="cpu")
                cleaned_state_dict = {
                    k.split("mm_projector.")[-1]: v for k, v in state_dict.items() if "mm_projector" in k
                } or state_dict
                self.model.mm_projector.load_state_dict(cleaned_state_dict, strict=False)
                self.model.mm_projector.to(dtype=self.dtype)
                logger.info("Projector weights loaded successfully.")
        except Exception as e:
            logger.error(f"Projector load failed: {e}")

    def get_model(self) -> LlavaLlamaModel:
        return self.model

    def get_vision_tower(self) -> Optional[CLIPVisionTower]:
        return self.model.vision_tower

    def encode_images(self, images: torch.FloatTensor) -> torch.FloatTensor:
        vision_tower = self.get_vision_tower()
        if vision_tower is None:
            raise RuntimeError("Vision Tower not initialized.")
        image_features = vision_tower(images)
        image_features = self.model.mm_projector(image_features)
        return image_features

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
        
        if images is None and pixel_values is not None:
            images = pixel_values

        if inputs_embeds is None:
            # 1. Get Text Embeddings
            inputs_embeds = self.model.embed_tokens(input_ids)

            # 2. Multi-modal Fusion
            if images is not None and input_ids is not None and past_key_values is None:
                image_features = self.encode_images(images)
                batch_size, seq_len, _ = inputs_embeds.shape
                num_patches = image_features.shape[1]
                
                vision_tower = self.get_vision_tower()
                n = vision_tower.num_patches if vision_tower else num_patches

                new_inputs_embeds = []
                for i in range(batch_size):
                    cur_input_ids = input_ids[i]
                    cur_inputs_embeds = inputs_embeds[i]
                    
                    indices = (cur_input_ids == self.config.pad_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(indices) >= n:
                        # Vectorized search for contiguous block
                        limit = len(indices) - n + 1
                        upper, lower = indices[n-1:], indices[:limit]
                        matches = (upper - lower == (n - 1)).nonzero(as_tuple=True)[0]
                        
                        if len(matches) > 0:
                            start_idx = indices[matches[0]]
                            prefix = cur_inputs_embeds[:start_idx]
                            suffix = cur_inputs_embeds[start_idx + n:]
                            fused = torch.cat([prefix, image_features[i], suffix], dim=0)
                            
                            if fused.shape[0] > seq_len: fused = fused[:seq_len]
                            elif fused.shape[0] < seq_len: 
                                fused = torch.cat([fused, cur_inputs_embeds[fused.shape[0]:]], dim=0)
                            new_inputs_embeds.append(fused)
                        else:
                             new_inputs_embeds.append(cur_inputs_embeds)
                    else:
                        new_inputs_embeds.append(cur_inputs_embeds)
                
                inputs_embeds = torch.stack(new_inputs_embeds, dim=0)

        return super().forward(
            input_ids=None,
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
# Factory Function for Hydra
# ============================================================================

def LlavaModel(model_name_or_path: str, **kwargs):
    # Determine quantization and PEFT state
    use_qlora = kwargs.get("use_qlora", False)
    load_in_4bit = kwargs.get("load_in_4bit", False)
    load_in_8bit = kwargs.get("load_in_8bit", False)
    
    # 1. Setup Configuration
    from transformers import AutoConfig
    base_config = AutoConfig.from_pretrained(model_name_or_path)
    
    # Extract multi-modal parameters
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
    
    # 2. Setup Quantization
    quantization_config = None
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
    
    # 3. CLEAN UP: Pop all non-standard arguments to avoid TypeError in model __init__
    for k in LORA_ARGUMENTS + LLAVA_SPECIFIC_ARGS + ["use_qlora"]:
        kwargs.pop(k, None)

    # 4. Instantiate and Load
    return LlavaLlamaForCausalLM.from_pretrained(
        model_name_or_path, 
        config=llava_config,
        quantization_config=quantization_config,
        use_safetensors=True, 
        low_cpu_mem_usage=((load_in_4bit or load_in_8bit) and torch.cuda.is_available()),
        **kwargs
    )