import torch
import torch.nn as nn


class CLSFusionHead(nn.Module):
    def __init__(self, input_dims:int = [512, 768], output_dim: int = 512, num_heads=8, dropout=0.1):
        super(TransformerFusionHead, self).__init__()
        
        self.input_dims = input_dims

        self.proj_heads = nn.ModuleList([nn.Linear(dim, output_dim) for dim in input_dims])
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(output_dim)
        # self.ffn = nn.Sequential(
        #     nn.Linear(output_dim, output_dim * 4),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(output_dim * 4, output_dim),
        # )
        # self.norm2 = nn.LayerNorm(output_dim)
    
    def forward(self, *inputs):
        assert len(inputs) == len(self.input_dims)

        inputs = [inp.to(torch.float32) for inp in inputs]
        inputs = [proj(inp) for inp, proj in zip(inputs, self.proj_heads)]
        
        x = torch.stack(inputs, dim=1) # (B, N, D)
        
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) # (B, N + 1, D)

        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # ffn_out = self.ffn(x)
        # x = self.norm2(x + ffn_out)

        x = x[:, 0]

        x = nn.functional.normalize(x, dim=1)
        return x

class TransformerFusionHead(nn.Module):
    def __init__(self, input_dims:int = [512, 768], output_dim: int = 512, num_heads=8, dropout=0.1):
        super(TransformerFusionHead, self).__init__()
        
        self.input_dims = input_dims

        self.proj_heads = nn.ModuleList([nn.Linear(dim, output_dim) for dim in input_dims])
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
        )
        self.norm2 = nn.LayerNorm(output_dim)
    
    def forward(self, *inputs):
        assert len(inputs) == len(self.input_dims)

        inputs = [inp.to(torch.float32) for inp in inputs]
        inputs = [proj(inp) for inp, proj in zip(inputs, self.proj_heads)]
        
        x = torch.stack(inputs, dim=1) # (B, N, D)
        
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) # (B, N + 1, D)

        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        x = x[:, 0]

        x = nn.functional.normalize(x, dim=1)
        return x

class AttentionFusionHead(nn.Module):
    def __init__(self, input_dims:int = [512, 768], output_dim: int = 512, num_heads=8):
        super(AttentionFusionHead, self).__init__()
        
        self.input_dims = input_dims

        self.proj_heads = nn.ModuleList([nn.Linear(dim, output_dim) for dim in input_dims])
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            # nn.GELU(),
            # nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, *inputs):
        assert len(inputs) == len(self.input_dims)

        inputs = [inp.to(torch.float32) for inp in inputs]
        inputs = [proj(inp) for inp, proj in zip(inputs, self.proj_heads)]
        
        x = torch.stack(inputs, dim=1) # (B, N, D)

        # x, _ = self.attention(x, x, x)
        # x = x.mean(dim=1) # (b, dim)
        # x = self.norm(x)

        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = x.mean(dim=1) # (b, dim)

        x = self.mlp(x)
        
        x = nn.functional.normalize(x, dim=1)
        return x

# class AttentionFusionHead(nn.Module):
#     def __init__(self, output_dim: int = 512):
#         super(AttentionFusionHead, self).__init__()

#         self.proj_clip = nn.Linear(512, output_dim)
#         self.proj_dino = nn.Linear(768, output_dim)

#         self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True)

#         self.norm = nn.LayerNorm(output_dim)

#         self.mlp = nn.Sequential(
#             nn.Linear(output_dim, output_dim),
#             nn.GELU()
#         )
    
#     def forward(self, x1, x2):
#         x1 = x1.to(torch.float32)
#         x2 = x2.to(torch.float32)

#         x1 = self.proj_clip(x1)
#         x2 = self.proj_dino(x2)

#         x = torch.stack([x1, x2], dim=1) # (b, 2, dim)
#         x, _ = self.attention(x, x, x)
#         x = x.mean(dim=1) # (b, dim)

#         x = self.norm(x)
#         x = x + self.mlp(x)

#         x = nn.functional.normalize(x, dim=1)
    
#         return x