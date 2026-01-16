import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_name: str, args, delay_load: bool = False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_name
        
        # Load from args if available, else default
        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        elif args.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_model = CLIPVisionModel.from_pretrained(self.vision_tower_name, use_safetensors=True)
        self.vision_model.requires_grad_(False) # Default freeze
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:] # Remove CLS token
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select_feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
            self.vision_model.to(device=images.device, dtype=images.dtype)
        
        image_features = self.vision_model(images, output_hidden_states=True)
        image_features = self.feature_select(image_features)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_model.dtype

    @property
    def device(self):
        return self.vision_model.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_model.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        config = self.config
        patch_size = config.patch_size
        image_size = config.image_size
        patch_grid = (image_size // patch_size) ** 2
        
        if self.select_feature == "patch":
            return patch_grid
        elif self.select_feature == "cls_patch":
            return patch_grid + 1
        else:
            raise ValueError(f"Unknown select_feature: {self.select_feature}")
