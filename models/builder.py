# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .clip_model import CLIP
from .dino_v2 import DINO_v2


class LinearFusionHead(nn.Module):
    def __init__(self, output_dim: int = 512):
        super(LinearFusionHead, self).__init__()

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(512 + 768),
            nn.Linear(512 + 768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, output_dim),
            # nn.ReLU()
        )
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        return self.fusion_head(x)


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
    
        return x


class DINO_CLIP(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        transform=None,
        clip_model_name: str = "ViT-B/32",
        dino_model_name: str = "dinov2_vitb14",
        clip_transform=None,
        dino_transform=None,
        dim: int = 512,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
        # mlp: bool = False,
        fusion_head = "Linear"
    ) -> None:
        """
        dim: feature dimension (default: 512)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(DINO_CLIP, self).__init__()

        self.transform = transform
        self.clip_transform = clip_transform
        self.dino_transform = dino_transform
        self.K = K
        self.m = m
        self.T = T

        self.clip_model = CLIP(model_name=clip_model_name, transform=clip_transform)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model = self.clip_model.eval()

        self.dino_model = DINO_v2(model_name=dino_model_name, transform=dino_transform)
        for p in self.dino_model.parameters():
            p.requires_grad = False
        self.dino_model = self.dino_model.eval()

        # create the encoders
        # num_classes is the output fc dimension
        match fusion_head:
            case "Linear":
                self.encoder_q = LinearFusionHead(output_dim=dim)
                self.encoder_k = LinearFusionHead(output_dim=dim)
            case "Attention":
                self.encoder_q = AttentionFusionHead(output_dim=dim)
                self.encoder_k = AttentionFusionHead(output_dim=dim)

        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
        #     )
        #     self.encoder_k.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
        #     )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, keys) -> None:
    #     # gather keys before updating queue
    #     keys = concat_all_gather(keys)

    #     batch_size = keys.shape[0]

    #     ptr = int(self.queue_ptr)
    #     assert self.K % batch_size == 0  # for simplicity

    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.queue[:, ptr : ptr + batch_size] = keys.T
    #     ptr = (ptr + batch_size) % self.K  # move pointer

    #     self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _dequeue_and_enqueue_without_dpp(self, keys) -> None:
        # Single-GPU: keine Gather-Operation
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     """
    #     Batch shuffle, for making use of BatchNorm.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]

    #     num_gpus = batch_size_all // batch_size_this

    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda()

    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)

    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)

    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    #     return x_gather[idx_this], idx_unshuffle

    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     """
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]

    #     num_gpus = batch_size_all // batch_size_this

    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    #     return x_gather[idx_this]

    @torch.no_grad()
    def _batch_shuffle_without_dpp(self, x):
        # Shuffle only within the batch (no DDP)
        # idx_shuffle = torch.randperm(x.shape[0]).to(x.device)
        idx_shuffle = torch.randperm(x.shape[0])
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_without_dpp(self, x, idx_unshuffle):
        # Undo shuffle (no DDP)
        return x[idx_unshuffle]

    def forward(self, im_q, im_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if self.transform is not None:
            im_q = self.transform(im_q)
            if im_k is not None:
                im_k = self.transform(im_k)

        img_q_clip = self.clip_model(im_q)
        img_q_dino = self.dino_model(im_q)

        img_q_clip = nn.functional.normalize(img_q_clip, p=2, dim=1)
        img_q_dino = nn.functional.normalize(img_q_dino, p=2, dim=1)

        # compute query features
        q = self.encoder_q(img_q_clip, img_q_dino)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        if im_k is None:
            return q

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_without_dpp(im_k)

            img_k_clip = self.clip_model(im_k)
            img_k_dino = self.dino_model(im_k)

            img_k_clip = nn.functional.normalize(img_k_clip, p=2, dim=1)
            img_k_dino = nn.functional.normalize(img_k_dino, p=2, dim=1)

            k = self.encoder_k(img_k_clip, img_k_dino)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_without_dpp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue_without_dpp(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
