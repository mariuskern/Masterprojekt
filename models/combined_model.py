import torch
import torch.nn as nn

from .clip_model import CLIP
from .dino_v2 import DINO_v2


class CombinedModel(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/32", clip_transform=None, dino_model_name: str = "dinov2_vitb14", dino_transform=None):
        super().__init__()
        self.clip_model = CLIP(model_name=clip_model_name, transform=clip_transform)
        self.dino_model = DINO_v2(model_name=dino_model_name, transform=dino_transform)

        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.dino_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        clip_features = self.clip_model(x)
        dino_features = self.dino_model(x)

        clip_features = torch.nn.functional.normalize(clip_features, p=2, dim=1)
        dino_features = torch.nn.functional.normalize(dino_features, p=2, dim=1)

        return torch.cat((clip_features, dino_features), dim=1)
