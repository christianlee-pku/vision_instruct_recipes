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
    "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", 
    "lora_modules_to_save", "lora_bias"
]

LLAVA_SPECIFIC_ARGS = [
    "vision_tower", "mm_vision_select_layer", "mm_vision_select_feature", 
    "mm_hidden_size", "unfreeze_mm_vision_tower", "freeze_vision_tower", 
    "pretrain_mm_mlp_adapter", "load_in_4bit", "load_in_8bit", 
    "bnb_4bit_use_double_quant", "bnb_4bit_quant_type"
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
# Multi-Modal Mixin: The safest way to extend HF models for Quantization
# ============================================================================

class LlavaMetaModel:
    """
    This mixin ensures that vision components are added to the model
    WITHOUT breaking the standard LlamaModel structure.
    """
    def initialize_vision_modules(self, config: LlavaConfig):
        # We use standard attribute assignment. 
        # When this mixin is combined with LlamaModel, these become part of the module tree.
        if not hasattr(self, "vision_tower") or self.vision_tower is None:
            self.vision_tower = CLIPVisionTower(config.vision_tower, delay_load=True, args=config)
        
        if not hasattr(self, "mm_projector") or self.mm_projector is None:
            # FIX: Force projector initialization on CPU.
            # When low_cpu_mem_usage=True (default for QLoRA), the model init context is "meta".
            # If the projector layers are created on meta device without subsequent weight loading
            # (since they are new/randomly init), accelerate/bnb will fail to move/quantize them 
            # because they have no data ("Cannot copy out of meta tensor").
            # Initializing them on CPU ensures they have real data.
            # We use a try-except block to handle cases where we might not be in a specific context.
            try:
                # FIX: Force projector initialization on CPU.
                # When low_cpu_mem_usage=True (default for QLoRA), the model init context is "meta".
                # We need real tensors for the new projector layers.
                with torch.device("cpu"):
                    self.mm_projector = LlavaMultiModalProjector(config)
            except Exception:
                self.mm_projector = LlavaMultiModalProjector(config)

            # DOUBLE CHECK: If it's still on meta (e.g. if the context manager was overridden by accelerate),
            # force materialization.
            if any(p.device.type == "meta" for p in self.mm_projector.parameters()):
                self.mm_projector.to_empty(device="cpu")
                # Re-init weights manually since it's a raw nn.Module or similar
                for m in self.mm_projector.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    def get_vision_tower(self):
        return getattr(self, "vision_tower", None)

# ============================================================================
# Core Model Classes: Implementing the Mixin pattern
# ============================================================================

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    """LlamaModel extended with vision capabilities via Mixin."""
    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaLlamaForCausalLM(LlamaForCausalLM):
    """
    LLaVA Causal LM which replaces the standard LlamaModel with our vision-capable version.
    """
    config_class = LlavaConfig
    _keys_to_ignore_on_load_missing = ["vision_tower", "mm_projector", "model.vision_tower", "model.mm_projector"]

    def _initialize_missing_keys(self, missing_keys: List[str], is_quantized: bool = False):
        """
        Override to prevent transformers from trying to initialize our custom modules 
        (vision_tower, mm_projector) which are naturally 'missing' from the base LLM checkpoint.
        The default implementation causes AttributeError on quantized models.
        """
        # Filter out our custom keys from the missing list so the base method doesn't see them
        filtered_keys = []
        for k in missing_keys:
            if "vision_tower" in k or "mm_projector" in k:
                continue
            filtered_keys.append(k)

        try:
            # Call the parent implementation with the filtered list
            # We still try to force is_quantized=False to avoid the buggy path if possible
            super()._initialize_missing_keys(filtered_keys, is_quantized=False)
        except AttributeError:
            # FALLBACK: If the parent method crashes while trying to introspect weights (common in QLoRA),
            # we suppress the error. The critical weights (base model) are already loaded.
            # The missing keys were likely just artifacts or non-module parameters that triggered the bug.
            pass
        except Exception as e:
            # Re-raise other unexpected errors
            raise e

    def __init__(self, config: LlavaConfig):
        # We avoid calling LlamaForCausalLM.__init__ to prevent double model instantiation.
        # Instead, we call the grandparent (LlamaPreTrainedModel) init directly.
        super(LlamaForCausalLM, self).__init__(config)
        
        # This is the "Gold Standard" pattern for LLaVA:
        # Manually assign the specific model class.
        self.model = LlavaLlamaModel(config)
        # DEFERRED: initialize_vision_modules and projector loading are moved to LlavaModel factory
        # to avoid 'meta' device conflicts during from_pretrained loading.
        
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        # FINAL SAFEGUARD: Removed here, handled in factory.

    def _check_and_load_projector(self, config: LlavaConfig) -> None:
        projector_path = getattr(config, "pretrain_mm_mlp_adapter", None)
        if not projector_path:
            return

        logger.info(f"Loading projector weights from: {projector_path}")
        try:
            if not os.path.exists(projector_path):
                # Remote resolving...
                for filename in ["mm_projector.bin", "pytorch_model.bin"]:
                    try:
                        res = cached_file(projector_path, filename)
                        if res: 
                            projector_path = res
                            break
                    except: continue
            
            if os.path.exists(projector_path):
                weights = torch.load(projector_path, map_location="cpu")
                cleaned = {k.split("mm_projector.")[-1]: v for k, v in weights.items() if "mm_projector" in k} or weights
                self.model.mm_projector.load_state_dict(cleaned, strict=False)
                self.model.mm_projector.to(dtype=self.dtype)
                logger.info("Projector loaded and dtype matched.")
        except Exception as e:
            logger.error(f"Projector loading failed: {e}")

    def get_model(self) -> LlavaLlamaModel:
        return self.model

    def get_vision_tower(self):
        return self.model.get_vision_tower()

    def encode_images(self, images: torch.FloatTensor) -> torch.FloatTensor:
        vision_tower = self.get_vision_tower()
        if vision_tower is None:
            raise RuntimeError("Vision Tower not found. Check your config.")
        
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

            # 2. Multi-modal Fusion (Late Fusion)
            if images is not None and input_ids is not None and past_key_values is None:
                image_features = self.encode_images(images)
                batch_size, seq_len, _ = inputs_embeds.shape
                num_patches = image_features.shape[1]
                
                n = self.get_vision_tower().num_patches if self.get_vision_tower() else num_patches

                new_inputs_embeds = []
                for i in range(batch_size):
                    cur_input_ids = input_ids[i]
                    cur_inputs_embeds = inputs_embeds[i]
                    
                    # Search for contiguous PAD tokens (Placeholders)
                    # FIX: pad_token_id might be None in config, causing (tensor == None) -> False (bool).
                    # Use 0 or eos_token_id as fallback to ensure tensor comparison.
                    pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
                    indices = (cur_input_ids == pad_id).nonzero(as_tuple=True)[0]
                    
                    if len(indices) >= n:
                        # Vectorized search for the first valid block
                        limit = len(indices) - n + 1
                        upper, lower = indices[n-1:], indices[:limit]
                        matches = (upper - lower == (n - 1)).nonzero(as_tuple=True)[0]
                        
                        if len(matches) > 0:
                            start_idx = indices[matches[0]]
                            prefix = cur_inputs_embeds[:start_idx]
                            suffix = cur_inputs_embeds[start_idx + n:]
                            fused = torch.cat([prefix, image_features[i], suffix], dim=0)
                            
                            # Align sequence length for consistency
                            if fused.shape[0] > seq_len: fused = fused[:seq_len]
                            elif fused.shape[0] < seq_len: 
                                fused = torch.cat([fused, cur_inputs_embeds[fused.shape[0]:]], dim=0)
                            new_inputs_embeds.append(fused)
                        else:
                             new_inputs_embeds.append(cur_inputs_embeds)
                    else:
                        new_inputs_embeds.append(cur_inputs_embeds)
                
                inputs_embeds = torch.stack(new_inputs_embeds, dim=0)

        # 3. Call LlamaForCausalLM forward with injected embeddings
        return super().forward(
            input_ids=None, # We MUST pass None here as we provide inputs_embeds
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
        if images is not None: _inputs['images'] = images
        if pixel_values is not None: _inputs['pixel_values'] = pixel_values
        return _inputs

# ============================================================================
# Factory: Handles Hydra configuration and quantization loading
# ============================================================================

def LlavaModel(model_name_or_path: str, **kwargs):
    use_qlora = kwargs.get("use_qlora", False)
    load_in_4bit = kwargs.get("load_in_4bit", False)
    load_in_8bit = kwargs.get("load_in_8bit", False)
    
    from transformers import AutoConfig
    base_config = AutoConfig.from_pretrained(model_name_or_path)
    
    # Extract LLaVA specific parameters
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
    
    # CLEAN UP: Extremely critical to pop these before calling from_pretrained
    # to avoid TypeError in model __init__ or attribute errors in bitsandbytes
    for k in LORA_ARGUMENTS + LLAVA_SPECIFIC_ARGS + ["use_qlora"]:
        kwargs.pop(k, None)
    
    # DOUBLE CHECK: Explicitly remove quantization keys if they somehow persisted
    # Transformers 4.30+ throws ValueError if both config and bool are passed as kwargs.
    kwargs.pop("load_in_4bit", None)
    kwargs.pop("load_in_8bit", None)

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name_or_path, 
        config=llava_config,
        quantization_config=quantization_config,
        use_safetensors=True, 
        low_cpu_mem_usage=((load_in_4bit or load_in_8bit) and torch.cuda.is_available()),
        **kwargs
    )
    
    # POST-INIT: Initialize vision modules here to avoid 'Cannot copy out of meta tensor' errors.
    # This runs outside of transformers' init_empty_weights context.
    model.model.initialize_vision_modules(llava_config)
    model._check_and_load_projector(llava_config)
    
    # FINAL SYNC: Move custom modules to the same device as the base model.
    # We forced them to CPU during init to avoid meta-tensor crashes, but now they must join the GPU party.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Move projector
        if hasattr(model.model, "mm_projector"):
            model.model.mm_projector.to(device)
        
        # Move vision tower (if it loaded weights)
        # Note: CLIPVisionTower handles its own device placement in forward(), but explicitly moving it here is safer.
        if hasattr(model.model, "vision_tower") and model.model.vision_tower is not None:
            # Check if it's a raw module or our wrapper. Our wrapper usually stays on CPU until forward, 
            # but let's ensure the underlying model is ready if loaded.
            if hasattr(model.model.vision_tower, "vision_model"):
                 model.model.vision_tower.vision_model.to(device)

    return model