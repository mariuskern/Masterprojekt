import torch
import torch.nn as nn


class PoolingFusionHead(nn.Module):
    def __init__(self, input_dims:int = [512, 768], output_dim: int = 512, pooling: str = "max"):
        super(PoolingFusionHead, self).__init__()

        # clip_dim = 512
        # dino_dim = 768
        
        self.input_dims = input_dims

        self.proj_heads = nn.ModuleList([nn.Linear(dim, output_dim) for dim in input_dims])

        self.pool = None
        match pooling:
            case "max":
                self.pool = nn.MaxPool1d(kernel_size=len(input_dims))
            case "avg":
                self.pool = nn.AvgPool1d(kernel_size=len(input_dims))
            case _:
                raise ValueError("Pooling method not supported")
    
    def forward(self, *inputs):
        assert len(inputs) == len(self.input_dims)

        inputs = list(inputs)
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(torch.float32)
        
        for i in range(len(inputs)):
            inputs[i] = self.proj_heads[i](inputs[i])
        
        x = torch.stack(inputs, dim=1) # (B, num_features, proj_dim)
        x = x.transpose(1, 2) # (B, proj_dim, num_features)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = nn.functional.normalize(x, dim=1)
        return x
