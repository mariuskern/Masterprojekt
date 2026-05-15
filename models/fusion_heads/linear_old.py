import torch
import torch.nn as nn
from abc import ABC


class AbstractLinearFusionHeadOldNoAlpha(nn.Module, ABC):
    def __init__(self):
        super(AbstractLinearFusionHeadOldNoAlpha, self).__init__()
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        x = self.fusion_head(x)
        x = nn.functional.normalize(x, dim=1)
        return x


class LinearFusionHeadSmallOldNoAlpha(AbstractLinearFusionHeadOldNoAlpha):
    def __init__(self, output_dim: int = 512):
        super(LinearFusionHeadSmallOldNoAlpha, self).__init__()

        self.fusion_head = nn.Sequential(
            nn.Linear(512 + 768, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
        )


class LinearFusionHeadBaseOldNoAlpha(AbstractLinearFusionHeadOldNoAlpha):
    def __init__(self, output_dim: int = 512):
        super(LinearFusionHeadBaseOldNoAlpha, self).__init__()

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(512 + 768),
            nn.Linear(512 + 768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, output_dim),
            nn.ReLU()
        )


class AbstractLinearFusionHeadOldOneAlpha(nn.Module, ABC):
    def __init__(self, weighted_concat: bool = False):
        super(AbstractLinearFusionHeadOldOneAlpha, self).__init__()

        self.weighted_concat = weighted_concat

        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x1, x2):
        if self.weighted_concat:
            alpha = torch.sigmoid(self.alpha)
            x1 = alpha * x1
            x2 = (1 - alpha) * x2
        
        x = torch.cat([x1, x2], dim=-1)
        x = self.fusion_head(x)
        x = nn.functional.normalize(x, dim=1)
        return x


class LinearFusionHeadSmallOldOneAlpha(AbstractLinearFusionHeadOldOneAlpha):
    def __init__(self, output_dim: int = 512, weighted_concat: bool = False):
        super(LinearFusionHeadSmallOldOneAlpha, self).__init__(weighted_concat=weighted_concat)

        self.fusion_head = nn.Sequential(
            nn.Linear(512 + 768, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
        )


class LinearFusionHeadBaseOldOneAlpha(AbstractLinearFusionHeadOldOneAlpha):
    def __init__(self, output_dim: int = 512, weighted_concat: bool = False):
        super(LinearFusionHeadBaseOldOneAlpha, self).__init__(weighted_concat=weighted_concat)

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(512 + 768),
            nn.Linear(512 + 768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, output_dim),
            nn.ReLU()
        )


class LinearFusionHeadBaseOldThreeAlphas(nn.Module):
    def __init__(self, output_dim: int = 512, weighted_concat: bool = False):
        super(LinearFusionHeadBaseOldThreeAlphas, self).__init__()

        self.weighted_concat = weighted_concat
        self.clip_alpha = nn.Parameter(torch.tensor(1.0))
        self.dino_cls_alpha = nn.Parameter(torch.tensor(1.0))
        self.dino_patch_alpha = nn.Parameter(torch.tensor(1.0))

        # clip_dim = 512
        # dino_dim = 768

        proj_dim = 256
        self.clip_proj = nn.Linear(512, proj_dim)
        self.dino_cls_proj = nn.Linear(768, proj_dim)
        self.dino_patch_proj = nn.Linear(768, proj_dim)

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(proj_dim * 3),
            nn.Linear(proj_dim * 3, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, output_dim),
            nn.ReLU()
        )
    
    def forward(self, clip_features, dino_cls_features, dino_patch_features):
        clip_features = clip_features.to(torch.float32)
        dino_cls_features = dino_cls_features.to(torch.float32)
        dino_patch_features = dino_patch_features.to(torch.float32)

        clip_features = self.clip_proj(clip_features)
        dino_cls_features = self.dino_cls_proj(dino_cls_features)
        dino_patch_features = self.dino_patch_proj(dino_patch_features)

        if self.weighted_concat:
            clip_alpha = torch.sigmoid(self.clip_alpha)
            dino_cls_alpha = torch.sigmoid(self.dino_cls_alpha)
            dino_patch_alpha = torch.sigmoid(self.dino_patch_alpha)
            clip_features = clip_alpha * clip_features
            dino_cls_features = dino_cls_alpha * dino_cls_features
            dino_patch_features = dino_patch_alpha * dino_patch_features
        
        x = torch.cat([clip_features, dino_cls_features, dino_patch_features], dim=-1)
        x = self.fusion_head(x)
        x = nn.functional.normalize(x, dim=1)
        return x