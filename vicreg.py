import torch
from torch import nn
import torchvision
from torch.utils import data
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import wandb
import os
import math
from tqdm import tqdm

from datasets import ImageNet, Places365, ArtPlaces, iNaturalist
from evaluation_utils import evaluate_features
from models import VICReg, LARS, TrainTransform
from config import VicRegConfig as Config



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(torch.cuda.get_device_name())

    cfg = Config()


    wandb_login(cfg.LOG_RUN)

    data_loader, data_loader_val = get_data_loader(cfg)
    model = get_model(cfg, device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer, data_loader)


    wandb_init(project="vicreg", config=cfg.get_as_wandb_dict(), model=model)
    save_config(cfg)


    batch_max = len(data_loader) * cfg.EPOCHS
    batch_global = 0
    EVAL_AFTER = 1024 # Number of batches after which to evaluate on the validation set (= number of batches in an virtual epoch)
    SAVE_AFTER = 10240 # 20480

    # scaler = torch.amp.GradScaler("cuda")

    for epoch in range(cfg.EPOCHS):
        losses = []
        val_losses = []

        tqdm_data_loader = tqdm(data_loader, unit="batch")
        tqdm_data_loader.set_description(f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for i, ((x, y), _) in enumerate(tqdm_data_loader):
            x = x.to(device)
            y = y.to(device)

            # lr = adjust_learning_rate(EPOCHS, LR, BATCH_SIZE, optimizer, data_loader, i)

            # optimizer.zero_grad()
            # with torch.amp.autocast("cuda"):
            #     loss = model.forward(x, y)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # loss = model.forward(x, y)
            repr_loss, std_loss, cov_loss = model.forward(x, y)
            loss_vicreg = (
                cfg.SIM_COEFF * repr_loss
                + cfg.STD_COEFF * std_loss
                + cfg.COV_COEFF * cov_loss
            )

            alphas = model.get_alphas()
            alpha_reg = torch.zeros((), device=device) # 0.0
            for alpha in alphas.values():
                alpha_reg += (alpha - 1.0) ** 2
            
            loss = loss_vicreg + cfg.LAMBDA_ALPHA_REG * alpha_reg

            if (i+1) % 10 == 0 and cfg.LOG_RUN:
                wandb.log({
                    "loss": loss.item(),
                    "vicreg_loss": loss_vicreg.item(),
                    "repr_loss": repr_loss.item(),
                    "std_loss": std_loss.item(),
                    "cov_loss": cov_loss.item(),
                    "alpha_reg": alpha_reg.item(),
                    "learning_rate": scheduler.get_last_lr()[-1] if scheduler is not None else LR, 
                    "epoch": epoch + 1,
                    "batch": i + 1,
                    "batch_global": batch_global,
                    **{name: alpha.item() for name, alpha in alphas.items()}
                })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            postfix = {
                "Loss": loss.item(),
            }
            tqdm_data_loader.set_postfix(postfix)

            # Val
            if (batch_global + 1) % EVAL_AFTER == 0 or (batch_global + 1) == batch_max or batch_global == 5:
                result = {}

                model.eval()
                with torch.no_grad():
                    features = []
                    labels = []

                    tqdm_data_loader_val = tqdm(data_loader_val, unit="batch")
                    tqdm_data_loader_val.set_description(f"Epoch {epoch+1}/{cfg.EPOCHS}")
                    for i, ((x, y), labels_batch) in enumerate(tqdm_data_loader_val):
                        x = x.to(device)
                        y = y.to(device)

                        # loss = model.forward(x, y)
                        # features_batch = model.forward(x)
                        repr_loss, std_loss, cov_loss, features_batch = model.forward(x, y, return_loss_and_features=True)
                        loss = (
                            cfg.SIM_COEFF * repr_loss
                            + cfg.STD_COEFF * std_loss
                            + cfg.COV_COEFF * cov_loss
                        )

                        val_losses.append(loss.item())
                        features.extend(features_batch.cpu().tolist())
                        labels.extend(labels_batch.tolist())
                    features = np.array(features).astype("float32")
                    labels = np.array(labels)

                    result = evaluate_features(features, labels, cfg.EVALUATION_DISTANCE, cfg.EVALUATION_K)
                model.train()

                if cfg.LOG_RUN:
                    wandb.log({
                        "loss_avg": np.mean(losses),
                        "val_loss_avg": np.mean(val_losses),
                        **{f"acc@{k}": result[f"Accuracy@{k}"].item() for k in cfg.EVALUATION_K},
                        **{f"per_class_mean_acc@{k}": result["per_class"]["mean"][f"Accuracy@{k}"].item() for k in cfg.EVALUATION_K},
                        **{f"per_class_mean_precision@{k}": result["per_class"]["mean"][f"Precision@{k}"].item() for k in cfg.EVALUATION_K},
                        **{f"per_class_mean_recall@{k}": result["per_class"]["mean"][f"Recall@{k}"].item() for k in cfg.EVALUATION_K}
                    })

                losses = []
                val_losses = []

            if (batch_global + 1) % SAVE_AFTER == 0 or (batch_global + 1) == batch_max:
                save_model(cfg, model,epoch=epoch + 1, batch=i + 1, batch_global=batch_global + 1)
            
            if scheduler is not None:
                scheduler.step()
            
            batch_global += 1
    

    finish_wandb()

def wandb_login(log_run):
    if log_run:
        wandb.login()

def get_data_loader(cfg):
    transform = TrainTransform()
    clip_transform = torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    dino_transform = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    convnext_transform = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))

    match cfg.DATASET:
        case "ImageNet":
            dataset = ImageNet(root=cfg.LOCAL_DATA_DIR, transform=transform)
            dataset_val = ImageNet(root=cfg.LOCAL_DATA_DIR, transform=transform, split="val")
        case "Places365":
            dataset = Places365(root=cfg.LOCAL_DATA_DIR, transform=transform)
            dataset_val = Places365(root=cfg.LOCAL_DATA_DIR, transform=transform, split="val")
        case "ArtPlaces":
            dataset = ArtPlaces(root=cfg.LOCAL_DATA_DIR, transform=transform)
            dataset_val = ArtPlaces(root=cfg.LOCAL_DATA_DIR, transform=transform, split="val")
        case "iNaturalist":
            dataset = iNaturalist(root=cfg.LOCAL_DATA_DIR, transform=transform, target_type=cfg.TARGET_TYPE, split="train_mini")
            dataset_val = iNaturalist(root=cfg.LOCAL_DATA_DIR, transform=transform, target_type=cfg.TARGET_TYPE, split="val")
            dataset_val = dataset_val.getSubset(32000, classes=list(dataset_val.class_to_idx.keys()))

    data_loader = data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.DATA_LOADER_NUM_WORKERS, pin_memory=True, drop_last=True)
    data_loader_val = data.DataLoader(dataset_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.DATA_LOADER_NUM_WORKERS, pin_memory=True, drop_last=True)

    return data_loader, data_loader_val

def get_model(cfg, device):
    model = VICReg(
        cfg.BATCH_SIZE,
        # SIM_COEFF,
        # STD_COEFF,
        # COV_COEFF,
        clip_model_name=cfg.CLIP_MODEL_NAME,
        dino_model_name=cfg.DINO_MODEL_NAME,
        convnext_model_name = cfg.CONVNEXT_MODEL_NAME,
        use_dino_cls_and_patch_tokens = cfg.USE_DINO_CLS_AND_PATCH_TOKENS,
        clip_transform = None,
        dino_transform = None,
        convnext_transform = None,
        fusion_head=cfg.FUSION_HEAD,
        dim = cfg.DIM,
        use_weighted_concat = cfg.USE_WEIGHTED_CONCAT,
        use_proj = cfg.USE_PROJ,
        proj_dim = cfg.PROJ_DIM,
        projection_head_dims = cfg.PROJECTION_HEAD_DIMS
    )
    model = model.train()
    model = model.to(device)

    return model

def wandb_init(project, config, model):
    config["fusion_head_architecture"] = str(model.fusion_head)
    
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        # dir=r"C:\Users\mariu\Desktop",

        # track hyperparameters and run metadata
        config=config
    )

    wandb.watch(model.fusion_head, log="all", log_freq=100)

def finish_wandb():
    wandb.finish()

def _get_dest(cfg):
    dest = os.path.join(cfg.RESULT_DIR, cfg.DINO_MODEL_NAME.lower() + "_" + cfg.CLIP_MODEL_NAME.lower().replace("/", "") + "_" + cfg.DATASET.lower() + "_" + datetime.today().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(dest)

    return dest

def save_config(cfg):
    with open(os.path.join(_get_dest(cfg), "constants.json"), "w") as file:
        json.dump(cfg.get_as_dict(), file)

def save_model(cfg, model, epoch=0, batch=0, batch_global=0):
    torch.save(model.state_dict(), os.path.join(_get_dest(cfg), "state_dict_epoch_" + str(epoch) + "_batch_" + str(batch) + "_batch_global_" + str(batch_global) + ".pt"))

def get_optimizer(cfg, model):
    def _exclude_bias_and_norm(p):
        return p.ndim == 1

    optimizer = LARS(
        model.parameters(),
        # lr=0,
        lr = cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        weight_decay_filter=_exclude_bias_and_norm,
        lars_adaptation_filter=_exclude_bias_and_norm,
    )

    return optimizer

def get_scheduler(cfg, optimizer, data_loader):
    scheduler = None

    match cfg.SCHEDULER:
        # case "step":
        #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        case "cosine":
            t_max = len(data_loader) * cfg.EPOCHS
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=cfg.SCHEDULER_MIN_LR)
    
    return scheduler

if __name__ == "__main__":
    main()