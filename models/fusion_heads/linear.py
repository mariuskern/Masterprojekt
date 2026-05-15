import torch
import torch.nn as nn
from abc import ABC


class AbstractLinearFusionHead(nn.Module, ABC):
    def __init__(self, input_dims: int, use_weighted_concat: bool = False, use_proj: bool = False, proj_dim: int = 512):
        super(AbstractLinearFusionHead, self).__init__()

        # clip_dim = 512
        # dino_dim = 768
        
        self.input_dims = input_dims
        self.use_weighted_concat = use_weighted_concat
        self.proj_dim = proj_dim
        self.use_proj = use_proj

        self.alphas = nn.ParameterList()
        if self.use_weighted_concat:
            for i in range(len(input_dims)):
                alpha = nn.Parameter(torch.tensor(1.0))
                # setattr(self, f"alpha_{i}_{input_names[i]}", alpha)
                self.alphas.append(alpha)

        self.proj_heads = nn.ModuleList()
        if self.use_proj:
            for i in range(len(input_dims)):
                proj_head = nn.Linear(input_dims[i], self.proj_dim)
                # setattr(self, f"proj_{i}_{input_names[i]}", proj_head)
                self.proj_heads.append(proj_head)
    
    def forward(self, *inputs):
        assert len(inputs) == len(self.input_dims)

        inputs = list(inputs)
        
        if self.use_proj:
            for i in range(len(inputs)):
                inputs[i] = self.proj_heads[i](inputs[i])
        
        if self.use_weighted_concat:
            for i in range(len(inputs)):
                inputs[i] = self.alphas[i] * inputs[i]

        x = torch.cat(inputs, dim=-1)
        x = self.fusion_head(x)
        x = nn.functional.normalize(x, dim=1)
        return x


class LinearFusionHeadSmall(AbstractLinearFusionHead):
    def __init__(self, input_dims:int = [512, 768], output_dim: int = 512, use_weighted_concat: bool = False, use_proj: bool = False, proj_dim: int = 512):
        super(LinearFusionHeadSmall, self).__init__(input_dims=input_dims, use_weighted_concat=use_weighted_concat, use_proj=use_proj, proj_dim=proj_dim)

        self.output_dim = output_dim

        input_dim = self.proj_dim * len(input_dims) if self.use_proj else sum(input_dims)
        self.fusion_head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
        )


class LinearFusionHeadBase(AbstractLinearFusionHead):
    def __init__(self, input_dims:int = [512, 768], output_dim: int = 512, use_weighted_concat: bool = False, use_proj: bool = False, proj_dim: int = 512):
        super(LinearFusionHeadBase, self).__init__(input_dims=input_dims, use_weighted_concat=use_weighted_concat, use_proj=use_proj, proj_dim=proj_dim)

        self.output_dim = output_dim

        input_dim = self.proj_dim * len(input_dims) if self.use_proj else sum(input_dims)
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, output_dim),
            nn.ReLU()
        )


# class LinearFusionHeadBase(nn.Module):
#     def __init__(self, input_dims:int = [512, 768], output_dim: int = 512, use_weighted_concat: bool = False, use_proj: bool = False, proj_dim: int = 512):
#         super(LinearFusionHeadBase, self).__init__()

#         # clip_dim = 512
#         # dino_dim = 768
        
#         self.input_dims = input_dims
#         self.output_dim = output_dim
#         self.use_weighted_concat = use_weighted_concat
#         self.proj_dim = proj_dim
#         self.use_proj = use_proj

#         self.alphas = []
#         if self.use_weighted_concat:
#             for i in range(len(input_dims)):
#                 alpha = nn.Parameter(torch.tensor(1.0))
#                 setattr(self, f"alpha_{i}", alpha)
#                 self.alphas.append(alpha)

#         self.proj_heads = []
#         if self.use_proj:
#             for i in range(len(input_dims)):
#                 proj_head = nn.Linear(input_dims[i], self.proj_dim)
#                 setattr(self, f"proj_{i}", proj_head)
#                 self.proj_heads.append(proj_head)

#         input_dim = self.proj_dim * len(input_dims) if self.use_proj else sum(input_dims)
#         self.fusion_head = nn.Sequential(
#             nn.LayerNorm(input_dim),
#             nn.Linear(input_dim, 1024),
#             nn.GELU(),
#             nn.LayerNorm(1024),
#             nn.Linear(1024, 1024),
#             nn.GELU(),
#             nn.LayerNorm(1024),
#             nn.Linear(1024, output_dim),
#             nn.ReLU()
#         )
    
#     def forward(self, *inputs):
#         assert len(inputs) == len(self.input_dims)
        
#         if self.use_proj:
#             for i in range(len(inputs)):
#                 inputs[i] = self.proj_heads[i](inputs[i])
        
#         if self.use_weighted_concat:
#             for i in range(len(inputs)):
#                 inputs[i] = self.alphas[i] * inputs[i]

#         x = torch.cat(inputs, dim=-1)
#         x = self.fusion_head(x)
#         x = nn.functional.normalize(x, dim=1)
#         return x