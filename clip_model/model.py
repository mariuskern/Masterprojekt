import torch
import torch.nn as nn
import clip


class CLIP(nn.Module):
    def __init__(self, model_name: str = "ViT-B/32", transform=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        self.transform = transform
        self.model, _ = clip.load(model_name, device=device)
        self.device = device
    
    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)

        return  self.model.encode_image(x)