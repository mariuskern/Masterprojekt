import torch
import torch.nn as nn

from .clip_model import CLIP
from .dino_v2 import DINO_v2
from .convnext import ConvNeXtv2


class CombinedModel(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/32", clip_transform=None, dino_model_name: str = "dinov2_vitb14", dino_transform=None, dino_return_features: bool = True, convnext_model_name: str = "convnextv2_nano.fcmae_ft_in22k_in1k", convnext_transform=None):
        super().__init__()
        self.clip_model = CLIP(model_name=clip_model_name, transform=clip_transform)
        self.dino_model = DINO_v2(model_name=dino_model_name, transform=dino_transform, return_features=dino_return_features)
        self.convnext_model = ConvNeXtv2(model_name=convnext_model_name, transform=convnext_transform)

        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.dino_model.parameters():
            param.requires_grad = False
        for param in self.convnext_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        clip_features = self.clip_model(x)
        dino_features_cls, dino_features_patch = self.dino_model(x)
        convnext_features = self.convnext_model(x)

        clip_features = torch.nn.functional.normalize(clip_features, p=2, dim=1)
        dino_features_cls = torch.nn.functional.normalize(dino_features_cls, p=2, dim=1)
        dino_features_patch = torch.nn.functional.normalize(dino_features_patch, p=2, dim=1)
        convnext_features = torch.nn.functional.normalize(convnext_features, p=2, dim=1)

        return torch.cat((clip_features, dino_features_cls, dino_features_patch, convnext_features), dim=1)

# class DINO_CLIP(nn.Module):
#     def __init__(self, clip_model_name: str = "ViT-B/32", clip_transform=None, dino_model_name: str = "dinov2_vitb14", dino_transform=None):
#         super().__init__()
#         self.clip_model = CLIP(model_name=clip_model_name, transform=clip_transform)
#         for p in self.clip_model.parameters():
#             p.requires_grad = False
#         self.clip_model = self.clip_model.eval()

#         self.dino_model = DINO_v2(model_name=dino_model_name, transform=dino_transform)
#         for p in self.dino_model.parameters():
#             p.requires_grad = False
#         self.dino_model = self.dino_model.eval()
        
#         self.fusion_head = nn.Sequential(
#             nn.LayerNorm(512 + 768),
#             nn.Linear(512 + 768, 1024),
#             nn.GELU(),
#             nn.Linear(1024, 512),
#             nn.LayerNorm(512)
#         )
    
#     def forward(self, x):
#         clip_features = self.clip_model(x)
#         dino_features = self.dino_model(x)

#         # clip_features = torch.nn.functional.normalize(clip_features, p=2, dim=1)
#         # dino_features = torch.nn.functional.normalize(dino_features, p=2, dim=1)

#         features = torch.cat((clip_features, dino_features), dim=1)
#         features = self.fusion_head(features)

#         return features
