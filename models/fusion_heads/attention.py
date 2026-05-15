import torch
import torch.nn as nn


class AttentionFusionHead(nn.Module):
    def __init__(self, output_dim: int = 512):
        super(AttentionFusionHead, self).__init__()

        self.proj_clip = nn.Linear(512, output_dim)
        self.proj_dino = nn.Linear(768, output_dim)

        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True)

        self.norm = nn.LayerNorm(output_dim)

        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU()
        )
    
    def forward(self, x1, x2):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        x1 = self.proj_clip(x1)
        x2 = self.proj_dino(x2)

        x = torch.stack([x1, x2], dim=1) # (b, 2, dim)
        x, _ = self.attention(x, x, x)
        x = x.mean(dim=1) # (b, dim)

        x = self.norm(x)
        x = x + self.mlp(x)

        x = nn.functional.normalize(x, dim=1)
    
        return x