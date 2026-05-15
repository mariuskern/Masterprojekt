import torch
import torch.nn as nn
import timm


class ConvNeXtv2(nn.Module):
    def __init__(self, model_name: str = "convnextv2_nano.fcmae_ft_in22k_in1k", transform=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        self.transform = transform
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            device=device
        )
        self.device = device
    
    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)

        x = self.model.forward_features(x)
        # output is unpooled, a (1, 640, 7, 7) shaped tensor
        x = self.model.forward_head(x, pre_logits=True)
        # output is a (1, num_features) shaped tensor

        return x