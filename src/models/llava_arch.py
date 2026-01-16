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
# Top-Level Model: The Causal LM Wrapper (Standard HF Pattern)
# ============================================================================
class LlavaLlamaForCausalLM(LlamaForCausalLM):
    """
    LLaVA (Large Language-and-Vision Assistant) Model.
    Inherits from LlamaForCausalLM and adds vision capabilities.
    """
    config_class = LlavaConfig

    def __init__(self, config: LlavaConfig):
        """
        Initializes the LLaVA model.
        """
        # We call the parent constructor which sets up self.model (LlamaModel) and self.lm_head
        super(LlavaLlamaForCausalLM, self).__init__(config)
        
        # Key improvement: We no longer override self.model = LlavaLlamaModel(config).
        # Instead, we dynamically mount the vision components on the existing self.model (LlamaModel).
        # This ensures that bitsandbytes still sees the standard LlamaModel structure during initialization.
        if not hasattr(self.model, "vision_tower"):
            self.model.vision_tower = None
            if hasattr(config, "vision_tower") and config.vision_tower:
                self.model.vision_tower = CLIPVisionTower(config.vision_tower, delay_load=False, args=config)
        
        if not hasattr(self.model, "mm_projector"):
            self.model.mm_projector = LlavaMultiModalProjector(config)

        # Initialize weights and apply final processing
        self.post_init()
        
        # Check for Projector Initialization (Robustness Fix)
        self._check_and_load_projector(config)

    def _check_and_load_projector(self, config: LlavaConfig) -> None:
        """
        Checks if the projector needs to be loaded from an external file or HF Hub.
        """
        projector_path = getattr(config, "pretrain_mm_mlp_adapter", None)
        if projector_path:
            logger.info(f"Attempting to load projector weights from: {projector_path}")
            try:
                if not os.path.exists(projector_path):
                    resolved_path = None
                    for filename in ["mm_projector.bin", "pytorch_model.bin", "adapter_model.bin"]:
                        try:
                            resolved_path = cached_file(projector_path, filename)
                            if resolved_path:
                                logger.info(f"Resolved remote projector file: {filename}")
                                break
                        except EnvironmentError:
                            continue
                    
                    if resolved_path:
                        projector_path = resolved_path
                    else:
                        logger.warning(f"Projector path '{projector_path}' could not be resolved.")

                projector_weights = torch.load(projector_path, map_location="cpu")
                
                new_state_dict = {}
                for k, v in projector_weights.items():
                    # Handle different weight saving formats
                    if "mm_projector" in k:
                        key_suffix = k.split("mm_projector.")[-1]
                        new_state_dict[key_suffix] = v
                    else:
                        new_state_dict[k] = v
                        
                # Key: Access via self.model.mm_projector
                missing, unexpected = self.model.mm_projector.load_state_dict(new_state_dict, strict=False)
                if missing:
                    logger.warning(f"Missing keys when loading projector: {missing}")
                
                logger.info("Projector weights loaded successfully.")
                self.model.mm_projector.to(dtype=self.dtype)
                
            except Exception as e:
                logger.error(f"Failed to load projector: {e}")
                raise e

    def get_model(self):
        """Returns the inner model."""
        return self.model

    def get_vision_tower(self) -> Optional[CLIPVisionTower]:
        """Returns the vision tower component."""
        if hasattr(self.model, "vision_tower"):
            return self.model.vision_tower
        return None

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ) -> Dict[str, Any]:
        """Standard HF prepare_inputs_for_generation with multi-modal support."""
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

    def encode_images(self, images: torch.FloatTensor) -> torch.FloatTensor:
        """
        Passes raw images through the vision tower and projector.
        """
        # Access via self.model
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
        Standard forward pass, but performs fusion of text and image embeddings.
        """
        if images is None and pixel_values is not None:
            images = pixel_values

        if inputs_embeds is None:
            # 1. Get Text Embeddings
            inputs_embeds = self.get_model().embed_tokens(input_ids)

            # 2. Inject Image Features (Fusion)
            # Only fuse if images exist and we are in the prefill stage (no KV cache)
            if images is not None and input_ids is not None and past_key_values is None:
                image_features = self.encode_images(images)
                            
                batch_size, seq_len, _ = inputs_embeds.shape
                num_patches = image_features.shape[1]
                
                # Verify patch count against vision tower config
                if hasattr(self.model, "vision_tower") and self.model.vision_tower is not None:
                    expected_num_patches = self.model.vision_tower.num_patches
                    if num_patches != expected_num_patches:
                         logger.warning(f"Image features length {num_patches} != expected {expected_num_patches}. Check Preprocessing.")
                else:
                    expected_num_patches = num_patches

                new_inputs_embeds = []
                for i in range(batch_size):
                    cur_input_ids = input_ids[i]
                    cur_inputs_embeds = inputs_embeds[i]
                    cur_image_features = image_features[i]
                    
                    # Locate placeholders (PAD tokens)
                    indices = (cur_input_ids == self.config.pad_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(indices) >= num_patches:
                        # Robust search for the contiguous block of image tokens
                        n = expected_num_patches
                        limit = len(indices) - n + 1
                        
                        upper = indices[n-1:]
                        lower = indices[:limit]
                        diffs = upper - lower
                        matches = (diffs == (n - 1)).nonzero(as_tuple=True)[0]
                        
                        if len(matches) > 0:
                            start_idx = indices[matches[0]]
                            
                            prefix = cur_inputs_embeds[:start_idx]
                            suffix = cur_inputs_embeds[start_idx + num_patches:]
                            
                            # Surgery: Replace text embeddings at placeholder indices with image features
                            fused = torch.cat([prefix, cur_image_features, suffix], dim=0)
                            
                            # Preservation of sequence length for label/mask alignment
                            if fused.shape[0] > seq_len:
                                fused = fused[:seq_len]
                            elif fused.shape[0] < seq_len:
                                fused = torch.cat([fused, cur_inputs_embeds[fused.shape[0]:]], dim=0)
                                
                            new_inputs_embeds.append(fused)
                        else:
                             logger.warning(f"Sample {i}: Contiguous PAD block of length {n} not found. Skipping fusion.")
                             new_inputs_embeds.append(cur_inputs_embeds)
                    else:
                        logger.warning(f"Sample {i}: Insufficient PAD tokens ({len(indices)}) for {num_patches} patches.")
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
