# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from models import CLIP, DINO_v2, ConvNeXtv2
from models.fusion_heads import AttentionFusionHead, LinearFusionHeadSmallOldNoAlpha, LinearFusionHeadBaseOldNoAlpha, LinearFusionHeadSmallOldOneAlpha, LinearFusionHeadBaseOldOneAlpha, LinearFusionHeadBaseOldThreeAlphas, LinearFusionHeadSmall, LinearFusionHeadBase, PoolingFusionHead


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        # transform=None,
        clip_model_name: str = "ViT-B/32",
        dino_model_name: str = "dinov2_vitb14",
        use_dino_cls_and_patch_tokens: bool = False,
        convnext_model_name: str = "convnextv2_nano.fcmae_ft_in22k_in1k",
        clip_transform=None,
        dino_transform=None,
        convnext_transform=None,
        dim: int = 512,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
        # mlp: bool = False,
        fusion_head = "Linear",
        use_weighted_concat = False,
        use_proj = False,
        proj_dim = 512,
        pooling = None,
        projection_head_dims = None,
    ) -> None:
        """
        dim: feature dimension (default: 512)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        if "Linear" not in fusion_head and use_weighted_concat:
            raise ValueError("use_weighted_concat is only supported for Linear fusion head")

        if (fusion_head == "LinearSmallOld" or fusion_head == "LinearBaseOld") and use_dino_cls_and_patch_tokens:
            raise ValueError("use_dino_cls_and_patch_tokens is not supported for LinearSmallOld and LinearBaseOld fusion heads")

        # self.transform = transform
        self.clip_transform = clip_transform
        self.dino_transform = dino_transform
        self.convnext_transform = convnext_transform
        self.K = K
        self.m = m
        self.T = T
        self.use_weighted_concat = use_weighted_concat
        self.use_dino_cls_and_patch_tokens = use_dino_cls_and_patch_tokens
        self.use_proj = use_proj
        self.proj_dim = proj_dim

        self.input_dims = []
        self.input_names = []
        self.models = []

        if clip_model_name is not None:
            self.clip_model = CLIP(model_name=clip_model_name, transform=clip_transform)
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_model = self.clip_model.eval()
            self.input_dims.append(512)
            self.input_names.append("clip")
            self.models.append(self.clip_model)

        if dino_model_name is not None:
            self.dino_model = DINO_v2(model_name=dino_model_name, transform=dino_transform, return_features=use_dino_cls_and_patch_tokens)
            for p in self.dino_model.parameters():
                p.requires_grad = False
            self.dino_model = self.dino_model.eval()
            self.models.append(self.dino_model)
            if use_dino_cls_and_patch_tokens:
                self.input_dims.extend([768, 768])
                self.input_names.extend(["dino_cls", "dino_patch"])
            else:
                self.input_dims.append(768)
                self.input_names.append("dino")

        if convnext_model_name is not None:
            self.convnext_model = ConvNeXtv2(model_name=convnext_model_name, transform=convnext_transform)
            for p in self.convnext_model.parameters():
                p.requires_grad = False
            self.convnext_model = self.convnext_model.eval()
            self.input_dims.append(640)
            self.input_names.append("convnext")
            self.models.append(self.convnext_model)

        # create the encoders
        # num_classes is the output fc dimension

        # if use_dino_cls_and_patch_tokens:
        #     input_dims = [512, 768, 768]
        # else:
        #     input_dims = [512, 768]

        match fusion_head:
            case "LinearSmallOldNoAlpha":
                self.encoder_q = LinearFusionHeadSmallOldNoAlpha(output_dim=dim)
                self.encoder_k = LinearFusionHeadSmallOldNoAlpha(output_dim=dim)
            case "LinearBaseOldNoAlpha":
                self.encoder_q = LinearFusionHeadBaseOldNoAlpha(output_dim=dim)
                self.encoder_k = LinearFusionHeadBaseOldNoAlpha(output_dim=dim)
            case "LinearSmallOldOneAlpha":
                self.encoder_q = LinearFusionHeadSmallOldOneAlpha(output_dim=dim, weighted_concat=use_weighted_concat)
                self.encoder_k = LinearFusionHeadSmallOldOneAlpha(output_dim=dim, weighted_concat=use_weighted_concat)
            case "LinearBaseOldOneAlpha":
                self.encoder_q = LinearFusionHeadBaseOldOneAlpha(output_dim=dim, weighted_concat=use_weighted_concat)
                self.encoder_k = LinearFusionHeadBaseOldOneAlpha(output_dim=dim, weighted_concat=use_weighted_concat)
            case "LinearBaseOldThreeAlphas":
                self.encoder_q = LinearFusionHeadBaseOldThreeAlphas(output_dim=dim, weighted_concat=use_weighted_concat)
                self.encoder_k = LinearFusionHeadBaseOldThreeAlphas(output_dim=dim, weighted_concat=use_weighted_concat)
            case "LinearSmall":
                self.encoder_q = LinearFusionHeadSmall(input_dims=self.input_dims, output_dim=dim, use_weighted_concat=use_weighted_concat, use_proj=use_proj, proj_dim=proj_dim)
                self.encoder_k = LinearFusionHeadSmall(input_dims=self.input_dims, output_dim=dim, use_weighted_concat=use_weighted_concat, use_proj=use_proj, proj_dim=proj_dim)
            case "LinearBase":
                self.encoder_q = LinearFusionHeadBase(input_dims=self.input_dims, output_dim=dim, use_weighted_concat=use_weighted_concat, use_proj=use_proj, proj_dim=proj_dim)
                self.encoder_k = LinearFusionHeadBase(input_dims=self.input_dims, output_dim=dim, use_weighted_concat=use_weighted_concat, use_proj=use_proj, proj_dim=proj_dim)
            case "Pooling":
                self.encoder_q = PoolingFusionHead(input_dims=self.input_dims, output_dim=dim, pooling=pooling)
                self.encoder_k = PoolingFusionHead(input_dims=self.input_dims, output_dim=dim, pooling=pooling)
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

        if projection_head_dims is not None:
            projection_head_dims.insert(0, dim)

            layers_q = []
            for i in range(len(projection_head_dims) - 2):
                layers_q.append(nn.Linear(projection_head_dims[i], projection_head_dims[i + 1]))
                layers_q.append(nn.BatchNorm1d(projection_head_dims[i + 1]))
                layers_q.append(nn.ReLU(True))
            layers_q.append(nn.Linear(projection_head_dims[-2], projection_head_dims[-1], bias=False))
            self.projector_q = nn.Sequential(*layers_q)

            layers_k = []
            for i in range(len(projection_head_dims) - 2):
                layers_k.append(nn.Linear(projection_head_dims[i], projection_head_dims[i + 1]))
                layers_k.append(nn.BatchNorm1d(projection_head_dims[i + 1]))
                layers_k.append(nn.ReLU(True))
            layers_k.append(nn.Linear(projection_head_dims[-2], projection_head_dims[-1], bias=False))
            self.projector_k = nn.Sequential(*layers_k)
        else:
            self.projector_q = nn.Identity()
            self.projector_k = nn.Identity()

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # # create the queue
        # self.register_buffer("queue", torch.randn(dim, K))
        # self.queue = nn.functional.normalize(self.queue, dim=0)

        if projection_head_dims is None:
            self.register_buffer("queue", torch.randn(dim, K))
        else:
            self.register_buffer("queue", torch.randn(projection_head_dims[-1], K))
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
        
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
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
    
    def get_alphas(self):
        if not hasattr(self.encoder_q, "alphas"):
            return {}
        
        alphas = {}
        # for i, name in enumerate(self.input_names):
        #     alpha_name = f"alpha_{name}"
        #     if hasattr(self.encoder_q, alpha_name):
        #         alphas[name] = getattr(self.encoder_q, alpha_name).item()
        #     else:
        #         alphas[name] = None
        for i, alpha in enumerate(self.encoder_q.alphas):
            name = f"{self.input_names[i]}_alpha"
            alphas[name] = alpha
        return alphas
    
    def _forward_helper(self, im):
        im_models = []
        for model in self.models:
            im_model = model(im)

            if isinstance(im_model, tuple):
                im_model = [nn.functional.normalize(x, p=2, dim=1) for x in im_model]
                im_models.extend(im_model)
            else:
                im_model = nn.functional.normalize(im_model, p=2, dim=1)
                im_models.append(im_model)
        return im_models

    def forward(self, im_q, im_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # if self.transform is not None:
        #     im_q = self.transform(im_q)
        #     if im_k is not None:
        #         im_k = self.transform(im_k)
        
        # im_q_models = self._forward_helper(im_q)
        # q = self.encoder_q(*im_q_models)
        # features = q.clone()

        # img_q_clip = self.clip_model(im_q)
        # img_q_clip = nn.functional.normalize(img_q_clip, p=2, dim=1)

        # img_q_convnext = self.convnext_model(im_q)
        # img_q_convnext = nn.functional.normalize(img_q_convnext, p=2, dim=1)

        # if self.use_dino_cls_and_patch_tokens:
        #     img_q_dino_cls, img_q_dino_patch = self.dino_model(im_q)
        #     img_q_dino_cls = nn.functional.normalize(img_q_dino_cls, p=2, dim=1)
        #     img_q_dino_patch = nn.functional.normalize(img_q_dino_patch, p=2, dim=1)
        #     # compute query features
        #     q = self.encoder_q(img_q_clip, img_q_dino_cls, img_q_dino_patch, img_q_convnext)  # queries: NxC
        # else:
        #     img_q_dino = self.dino_model(im_q)
        #     img_q_dino = nn.functional.normalize(img_q_dino, p=2, dim=1)
        #     # compute query features
        #     q = self.encoder_q(img_q_clip, img_q_dino, img_q_convnext)  # queries: NxC

        # # compute query features
        # q = self.encoder_q(img_q_clip, img_q_dino_cls, img_q_dino_patch)  # queries: NxC
        # # q = nn.functional.normalize(q, dim=1)

        # if im_k is None:
        #     return features
        
        # q = self.projector_q(q)

        if im_k is None:
            im_q_models = self._forward_helper(im_q)
            q = self.encoder_q(*im_q_models)
            return q

        im_q_models = self._forward_helper(im_q)
        q = self.encoder_q(*im_q_models)
        features = q.clone()
        q = self.projector_q(q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_without_dpp(im_k)

            im_k_models = self._forward_helper(im_k)
            k = self.encoder_k(*im_k_models)
            k = self.projector_k(k)

            # img_k_clip = self.clip_model(im_k)
            # img_k_clip = nn.functional.normalize(img_k_clip, p=2, dim=1)

            # img_k_convnext = self.convnext_model(im_k)
            # img_k_convnext = nn.functional.normalize(img_k_convnext, p=2, dim=1)

            # if self.use_dino_cls_and_patch_tokens:
            #     img_k_dino_cls, img_k_dino_patch = self.dino_model(im_k)
            #     img_k_dino_cls = nn.functional.normalize(img_k_dino_cls, p=2, dim=1)
            #     img_k_dino_patch = nn.functional.normalize(img_k_dino_patch, p=2, dim=1)
            #     k = self.encoder_k(img_k_clip, img_k_dino_cls, img_k_dino_patch, img_k_convnext)  # keys: NxC
            # else:
            #     img_k_dino = self.dino_model(im_k)
            #     img_k_dino = nn.functional.normalize(img_k_dino, p=2, dim=1)
            #     k = self.encoder_k(img_k_clip, img_k_dino, img_k_convnext)  # keys: NxC

            # img_k_clip = nn.functional.normalize(img_k_clip, p=2, dim=1)
            # img_k_dino_cls = nn.functional.normalize(img_k_dino_cls, p=2, dim=1)
            # img_k_dino_patch = nn.functional.normalize(img_k_dino_patch, p=2, dim=1)

            # k = self.encoder_k(img_k_clip, img_k_dino_cls, img_k_dino_patch)  # keys: NxC
            # # k = nn.functional.normalize(k, dim=1)

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

        return logits, labels, features


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
