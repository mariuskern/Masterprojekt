import torch
import torch.nn as nn


class DINO_v2(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14", transform=None):
        super().__init__()

        self.transform = transform

        match model_name:
            case "dinov2_vits14":
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            case "dinov2_vitb14":
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            case "dinov2_vitl14":
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            case "dinov2_vitg14":
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            case _:
                raise ValueError(f"Model {model_name} not supported.")
    
    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        
        return self.model(x)