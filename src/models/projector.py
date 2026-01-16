import torch.nn as nn
import re

class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Standard LLaVA-1.5 Projector: Linear -> GELU -> Linear
        
        # Infer sizes from config if available, else defaults
        mm_hidden_size = getattr(config, 'mm_hidden_size', 1024) # CLIP Large
        hidden_size = getattr(config, 'hidden_size', 4096)       # Llama 3 / 7B
        
        self.linear_1 = nn.Linear(mm_hidden_size, hidden_size)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
