# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets

# import augmentations as aug
# from distributed import init_distributed_mode

# import resnet

from models import CLIP, DINO_v2, ConvNeXtv2
from models.fusion_heads import LinearFusionHeadSmall, LinearFusionHeadBase, AttentionFusionHead, PoolingFusionHead, TransformerFusionHead


# def get_arguments():
#     parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

#     # Data
#     parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
#                         help='Path to the image net dataset')

#     # Checkpoints
#     parser.add_argument("--exp-dir", type=Path, default="./exp",
#                         help='Path to the experiment folder, where all logs/checkpoints will be stored')
#     parser.add_argument("--log-freq-time", type=int, default=60,
#                         help='Print logs to the stats.txt file every [log-freq-time] seconds')

#     # Model
#     parser.add_argument("--arch", type=str, default="resnet50",
#                         help='Architecture of the backbone encoder network')
#     parser.add_argument("--mlp", default="8192-8192-8192",
#                         help='Size and number of layers of the MLP expander head')

#     # Optim
#     parser.add_argument("--epochs", type=int, default=100,
#                         help='Number of epochs')
#     parser.add_argument("--batch-size", type=int, default=2048,
#                         help='Effective batch size (per worker batch size is [batch-size] / world-size)')
#     parser.add_argument("--base-lr", type=float, default=0.2,
#                         help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
#     parser.add_argument("--wd", type=float, default=1e-6,
#                         help='Weight decay')

#     # Loss
#     parser.add_argument("--sim-coeff", type=float, default=25.0,
#                         help='Invariance regularization loss coefficient')
#     parser.add_argument("--std-coeff", type=float, default=25.0,
#                         help='Variance regularization loss coefficient')
#     parser.add_argument("--cov-coeff", type=float, default=1.0,
#                         help='Covariance regularization loss coefficient')

#     # Running
#     parser.add_argument("--num-workers", type=int, default=10)
#     parser.add_argument('--device', default='cuda',
#                         help='device to use for training / testing')

#     # Distributed
#     parser.add_argument('--world-size', default=1, type=int,
#                         help='number of distributed processes')
#     parser.add_argument('--local_rank', default=-1, type=int)
#     parser.add_argument('--dist-url', default='env://',
#                         help='url used to set up distributed training')

#     return parser


# def main(args):
#     torch.backends.cudnn.benchmark = True
#     init_distributed_mode(args)
#     print(args)
#     gpu = torch.device(args.device)

#     if args.rank == 0:
#         args.exp_dir.mkdir(parents=True, exist_ok=True)
#         stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
#         print(" ".join(sys.argv))
#         print(" ".join(sys.argv), file=stats_file)

#     transforms = aug.TrainTransform()

#     dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
#     sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
#     assert args.batch_size % args.world_size == 0
#     per_device_batch_size = args.batch_size // args.world_size
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=per_device_batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         sampler=sampler,
#     )

#     model = VICReg(args).cuda(gpu)
#     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
#     optimizer = LARS(
#         model.parameters(),
#         lr=0,
#         weight_decay=args.wd,
#         weight_decay_filter=exclude_bias_and_norm,
#         lars_adaptation_filter=exclude_bias_and_norm,
#     )

#     if (args.exp_dir / "model.pth").is_file():
#         if args.rank == 0:
#             print("resuming from checkpoint")
#         ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
#         start_epoch = ckpt["epoch"]
#         model.load_state_dict(ckpt["model"])
#         optimizer.load_state_dict(ckpt["optimizer"])
#     else:
#         start_epoch = 0

#     start_time = last_logging = time.time()
#     scaler = torch.cuda.amp.GradScaler()
#     for epoch in range(start_epoch, args.epochs):
#         sampler.set_epoch(epoch)
#         for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
#             x = x.cuda(gpu, non_blocking=True)
#             y = y.cuda(gpu, non_blocking=True)

#             lr = adjust_learning_rate(args, optimizer, loader, step)

#             optimizer.zero_grad()
#             with torch.cuda.amp.autocast():
#                 loss = model.forward(x, y)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             current_time = time.time()
#             if args.rank == 0 and current_time - last_logging > args.log_freq_time:
#                 stats = dict(
#                     epoch=epoch,
#                     step=step,
#                     loss=loss.item(),
#                     time=int(current_time - start_time),
#                     lr=lr,
#                 )
#                 print(json.dumps(stats))
#                 print(json.dumps(stats), file=stats_file)
#                 last_logging = current_time
#         if args.rank == 0:
#             state = dict(
#                 epoch=epoch + 1,
#                 model=model.state_dict(),
#                 optimizer=optimizer.state_dict(),
#             )
#             torch.save(state, args.exp_dir / "model.pth")
#     if args.rank == 0:
#         torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")


# def adjust_learning_rate(args, optimizer, loader, step):
#     max_steps = args.epochs * len(loader)
#     warmup_steps = 10 * len(loader)
#     base_lr = args.base_lr * args.batch_size / 256
#     if step < warmup_steps:
#         lr = base_lr * step / warmup_steps
#     else:
#         step -= warmup_steps
#         max_steps -= warmup_steps
#         q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
#         end_lr = base_lr * 0.001
#         lr = base_lr * q + end_lr * (1 - q)
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
#     return lr


class VICReg(nn.Module):
    def __init__(
            self,
            batch_size,
            # sim_coeff,
            # std_coeff,
            # cov_coeff,
            transform=None,
            clip_model_name: str = "ViT-B/32",
            dino_model_name: str = "dinov2_vitb14",
            convnext_model_name: str = "convnextv2_nano.fcmae_ft_in22k_in1k",
            use_dino_cls_and_patch_tokens: bool = False,
            clip_transform = None,
            dino_transform = None,
            convnext_transform = None,
            fusion_head = "Linear",
            dim: int = 512,
            use_weighted_concat = False,
            use_proj = False,
            proj_dim = 512,
            pooling = None,
            projection_head_dims = [2048, 2048, 256]
        ):

        super().__init__()

        self.transform = transform
        self.batch_size = batch_size
        # self.sim_coeff = sim_coeff
        # self.std_coeff = std_coeff
        # self.cov_coeff = cov_coeff
        self.num_features = dim

        # self.args = args
        # self.num_features = int(args.mlp.split("-")[-1])
        # self.backbone, self.embedding = resnet.__dict__[args.arch](
        #     zero_init_residual=True
        # )
        # self.projector = Projector(args, self.embedding)

        self.input_dims = []
        self.input_names = []
        self.models = nn.ModuleList()
        # self.transforms = []

        if clip_model_name is not None:
            clip_model = CLIP(model_name=clip_model_name, transform=clip_transform)
            for p in clip_model.parameters():
                p.requires_grad = False
            clip_model = clip_model.eval()
            self.input_dims.append(512)
            self.input_names.append("clip")
            self.models.append(clip_model)
            # self.transforms.append(clip_transform)

        if dino_model_name is not None:
            dino_model = DINO_v2(model_name=dino_model_name, transform=dino_transform, return_features=use_dino_cls_and_patch_tokens)
            for p in dino_model.parameters():
                p.requires_grad = False
            dino_model = dino_model.eval()
            self.models.append(dino_model)
            # self.transforms.append(dino_transform)
            if use_dino_cls_and_patch_tokens:
                self.input_dims.extend([768, 768])
                self.input_names.extend(["dino_cls", "dino_patch"])
            else:
                self.input_dims.append(768)
                self.input_names.append("dino")

        if convnext_model_name is not None:
            convnext_model = ConvNeXtv2(model_name=convnext_model_name, transform=convnext_transform)
            for p in convnext_model.parameters():
                p.requires_grad = False
            convnext_model = convnext_model.eval()
            self.input_dims.append(640)
            self.input_names.append("convnext")
            self.models.append(convnext_model)
            # self.transforms.append(convnext_transform)
        
        match fusion_head:
            case "LinearSmall":
                self.fusion_head = LinearFusionHeadSmall(input_dims=self.input_dims, output_dim=dim, use_weighted_concat=use_weighted_concat, use_proj=use_proj, proj_dim=proj_dim)
            case "LinearBase":
                self.fusion_head = LinearFusionHeadBase(input_dims=self.input_dims, output_dim=dim, use_weighted_concat=use_weighted_concat, use_proj=use_proj, proj_dim=proj_dim)
            case "Pooling":
                self.fusion_head = PoolingFusionHead(input_dims=self.input_dims, output_dim=dim, pooling=pooling)
            case "Transformer":
                self.fusion_head = TransformerFusionHead(input_dims=self.input_dims, output_dim=dim)
            case "Attention":
                self.fusion_head = AttentionFusionHead(input_dims=self.input_dims, output_dim=dim)
        
        # self.projector = nn.Sequential(
        #     nn.Linear(dim, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.GELU(),

        #     nn.Linear(2048, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.GELU(),

        #     nn.Linear(2048, 256)
        # )
        layers = []
        projection_head_dims.insert(0, dim)
        for i in range(len(projection_head_dims) - 2):
            layers.append(nn.Linear(projection_head_dims[i], projection_head_dims[i + 1]))
            layers.append(nn.BatchNorm1d(projection_head_dims[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(projection_head_dims[-2], projection_head_dims[-1], bias=False))
        self.projector = nn.Sequential(*layers)

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
    
    # def _forward_helper(self, im):
    #     im_models = []
    #     for model, transform in zip(self.models, self.transforms):
    #         im_input = im

    #         if transform is not None:
    #             im_input = transform(im_input)

    #         im_model = model(im_input)

    #         if isinstance(im_model, tuple):
    #             im_model = [nn.functional.normalize(x, p=2, dim=1) for x in im_model]
    #             im_models.extend(im_model)
    #         else:
    #             im_model = nn.functional.normalize(im_model, p=2, dim=1)
    #             im_models.append(im_model)
    #     return im_models

    def forward(self, x, y = None, return_loss_and_features = False):
        if y is None:
            x_models = self._forward_helper(x)
            x = self.fusion_head(*x_models)
            return x
        
        x_models = self._forward_helper(x)
        x = self.fusion_head(*x_models)
        features = x.clone()
        x = self.projector(x)

        y_models = self._forward_helper(y)
        y = self.fusion_head(*y_models)
        y = self.projector(y)

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        # loss = (
        #     self.sim_coeff * repr_loss
        #     + self.std_coeff * std_loss
        #     + self.cov_coeff * cov_loss
        # )
        if return_loss_and_features:
            return repr_loss, std_loss, cov_loss, features
        else:
            return repr_loss, std_loss, cov_loss
    
    def get_alphas(self):
        if not hasattr(self.fusion_head, "alphas"):
            return {}
        
        alphas = {}
        for i, alpha in enumerate(self.fusion_head.alphas):
            name = f"{self.input_names[i]}_alpha"
            alphas[name] = alpha
        return alphas

# def Projector(args, embedding):
#     mlp_spec = f"{embedding}-{args.mlp}"
#     layers = []
#     f = list(map(int, mlp_spec.split("-")))
#     for i in range(len(f) - 2):
#         layers.append(nn.Linear(f[i], f[i + 1]))
#         layers.append(nn.BatchNorm1d(f[i + 1]))
#         layers.append(nn.ReLU(True))
#     layers.append(nn.Linear(f[-2], f[-1], bias=False))
#     return nn.Sequential(*layers)


# def exclude_bias_and_norm(p):
#     return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


# def batch_all_gather(x):
#     x_list = FullGatherLayer.apply(x)
#     return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        # If distributed is not available/initialized, just return the input
        if not dist.is_available() or not dist.is_initialized():
            return (x,)

        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        # If distributed not initialized, gradients are passed through
        if not dist.is_available() or not dist.is_initialized():
            return grads[0]

        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


# def handle_sigusr1(signum, frame):
#     os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
#     exit()


# def handle_sigterm(signum, frame):
#     pass


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
#     args = parser.parse_args()
#     main(args)
