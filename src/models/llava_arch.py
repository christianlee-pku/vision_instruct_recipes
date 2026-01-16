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
    "lora_modules_to_save", # Matches YAML key
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
        pretrain_mm_mlp_adapter: Optional[str] = None,
        **kwargs,
    ):
        # Allow kwargs to override explicit arguments if they are present
        # This handles cases where from_pretrained might pass them in kwargs even if they match signature
        self.vision_tower = vision_tower
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_hidden_size = mm_hidden_size
        self.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter
        
        super().__init__(**kwargs)

# ============================================================================
# Inner Model: LLM Backbone + Vision Tower + Projector
# ============================================================================
class LlavaLlamaModel(LlamaModel):
    """
    Core LLaVA model that extends LlamaModel with a Vision Tower and Projector.
    """
    def __init__(self, config: LlavaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        
        # Initialize multi-modal components
        self.vision_tower = None
        if hasattr(config, "vision_tower") and config.vision_tower:
            self.vision_tower = CLIPVisionTower(config.vision_tower, delay_load=False, args=config)
            
        self.mm_projector = LlavaMultiModalProjector(config)

    def get_vision_tower(self) -> Optional[CLIPVisionTower]:
        return self.vision_tower

# ============================================================================
# Top-Level Model: The Causal LM Wrapper
# ============================================================================
class LlavaLlamaForCausalLM(LlamaForCausalLM):
    """
    LLaVA Model for Causal Language Modeling.
    Wraps LlavaLlamaModel and adds the LM head and multi-modal fusion logic.
    """
    config_class = LlavaConfig

    def __init__(self, config: LlavaConfig):
        # We don't call super().__init__ because it would create a standard LlamaModel.
        # Instead, we perform manual initialization to use LlavaLlamaModel.
        # This is the most stable pattern for QLoRA and weight loading.
        super(LlamaForCausalLM, self).__init__(config)
        
        self.model = LlavaLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        # Load external projector weights if specified
        self._check_and_load_projector(config)

    def _check_and_load_projector(self, config: LlavaConfig) -> None:
        """
        Checks and loads projector weights from a local file or HF Hub.
        """
        projector_path = getattr(config, "pretrain_mm_mlp_adapter", None)
        if not projector_path:
            return

        logger.info(f"Loading projector weights from: {projector_path}")
        try:
            if not os.path.exists(projector_path):
                resolved_path = None
                for filename in ["mm_projector.bin", "pytorch_model.bin", "adapter_model.bin"]:
                    try:
                        resolved_path = cached_file(projector_path, filename)
                        if resolved_path: break
                    except EnvironmentError:
                        continue
                projector_path = resolved_path

            if not projector_path:
                logger.warning(f"Could not resolve projector path: {config.pretrain_mm_mlp_adapter}")
                return

            state_dict = torch.load(projector_path, map_location="cpu")
            # Map keys to the projector module
            cleaned_state_dict = {
                k.split("mm_projector.")[-1]: v for k, v in state_dict.items() if "mm_projector" in k
            } or state_dict
            
            missing, unexpected = self.model.mm_projector.load_state_dict(cleaned_state_dict, strict=False)
            if missing:
                logger.warning(f"Projector load: missing keys {missing}")
            
            self.model.mm_projector.to(dtype=self.dtype)
            logger.info("Projector loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load projector: {e}")
            raise e

    def get_model(self) -> LlavaLlamaModel:
        return self.model

    def get_vision_tower(self) -> Optional[CLIPVisionTower]:
        return self.model.get_vision_tower()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ) -> Dict[str, Any]:
        """Supports multi-modal generation by preserving pixel_values."""
        images = kwargs.get("images", None)
        pixel_values = kwargs.get("pixel_values", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None: _inputs['images'] = images
        if pixel_values is not None: _inputs['pixel_values'] = pixel_values
        return _inputs

    def encode_images(self, images: torch.FloatTensor) -> torch.FloatTensor:
        """Encodes images through vision tower and projector."""
        image_features = self.model.vision_tower(images)
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
        """
        Multi-modal forward pass. Fuses visual features into text embeddings.
        """
        if images is None and pixel_values is not None:
            images = pixel_values

        if inputs_embeds is None:
            # 1. Base Text Embeddings
            inputs_embeds = self.model.embed_tokens(input_ids)

            # 2. Multi-modal Fusion
            if images is not None and input_ids is not None and past_key_values is None:
                image_features = self.encode_images(images)
                batch_size, seq_len, _ = inputs_embeds.shape
                num_patches = image_features.shape[1]
                
                # Check for contiguous placeholders
                expected_num_patches = self.model.vision_tower.num_patches if self.model.vision_tower else num_patches
                
                new_inputs_embeds = []
                for i in range(batch_size):
                    cur_input_ids = input_ids[i]
                    cur_inputs_embeds = inputs_embeds[i]
                    cur_image_features = image_features[i]
                    
                    indices = (cur_input_ids == self.config.pad_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(indices) >= num_patches:
                        # Vectorized search for contiguous block
                        n = expected_num_patches
                        limit = len(indices) - n + 1
                        upper, lower = indices[n-1:], indices[:limit]
                        matches = (upper - lower == (n - 1)).nonzero(as_tuple=True)[0]
                        
                        if len(matches) > 0:
                            start_idx = indices[matches[0]]
                            prefix = cur_inputs_embeds[:start_idx]
                            suffix = cur_inputs_embeds[start_idx + num_patches:]
                            fused = torch.cat([prefix, cur_image_features, suffix], dim=0)
                            
                            # Align sequence length
                            if fused.shape[0] > seq_len: fused = fused[:seq_len]
                            elif fused.shape[0] < seq_len: 
                                fused = torch.cat([fused, cur_inputs_embeds[fused.shape[0]:]], dim=0)
                            new_inputs_embeds.append(fused)
                        else:
                             logger.warning(f"Sample {i}: Contiguous image block not found.")
                             new_inputs_embeds.append(cur_inputs_embeds)
                    else:
                        logger.warning(f"Sample {i}: Insufficient PAD tokens.")
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

# Factory for Hydra
def LlavaModel(model_name_or_path: str, **kwargs):
    use_qlora = kwargs.pop("use_qlora", False)
    
    # Remove arguments that are NOT part of LlavaConfig or LlamaConfig
    for k in LORA_ARGUMENTS:
        if k in kwargs:
            kwargs.pop(k)
    
    # Note: freeze_vision_tower is NOT in LlavaConfig, so we pop it to avoid error?
    if "freeze_vision_tower" in kwargs:
        kwargs.pop("freeze_vision_tower")
    
    # Manually create config to ensure parameters are applied
    # This fixes the issue where from_pretrained might drop kwargs if they don't match the loaded config structure
    from transformers import AutoConfig
    base_config = AutoConfig.from_pretrained(model_name_or_path)
    
    # Update base config with LLaVA specific parameters from kwargs
    llava_params = {
        "vision_tower": kwargs.get("vision_tower"),
        "mm_vision_select_layer": kwargs.get("mm_vision_select_layer"),
        "mm_vision_select_feature": kwargs.get("mm_vision_select_feature"),
        "mm_hidden_size": kwargs.get("mm_hidden_size"),
        "pretrain_mm_mlp_adapter": kwargs.get("pretrain_mm_mlp_adapter"),
    }
    
    # Create LlavaConfig (inheriting from base properties)
    # We update the base config dict with our new params
    config_dict = base_config.to_dict()
    config_dict.update({k: v for k, v in llava_params.items() if v is not None})
    
    # Instantiate LlavaConfig
    llava_config = LlavaConfig.from_dict(config_dict)
    
    # Quantization Configuration (Decoupled from use_qlora)
    quantization_config = None
    load_in_4bit = kwargs.get("load_in_4bit", False)
    load_in_8bit = kwargs.get("load_in_8bit", False)
    
    if load_in_4bit or load_in_8bit:
        if not torch.cuda.is_available():
            logger.warning("Quantization requested but CUDA is not available. Falling back to full precision.")
            # We must set these to False to avoid from_pretrained erroring out on CPU
            if "load_in_4bit" in kwargs: kwargs["load_in_4bit"] = False
            if "load_in_8bit" in kwargs: kwargs["load_in_8bit"] = False
        else:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.float16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
                
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=kwargs.get("bnb_4bit_use_double_quant", False),
                bnb_4bit_quant_type=kwargs.get("bnb_4bit_quant_type", "nf4"),
            )
            # Do NOT assign the object to config.quantization_config directly as it might be serialized/accessed as dict
            # llava_config.quantization_config = quantization_config 

    # Clean up kwargs that are now in config to avoid duplicates/conflicts if passed to from_pretrained
    for k in llava_params.keys():
        if k in kwargs:
            kwargs.pop(k)
            
    # Clean up quantization args from kwargs as they are handled by quantization_config
    # Also remove load_in_4bit/8bit to avoid DeprecationWarning and conflicts
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
