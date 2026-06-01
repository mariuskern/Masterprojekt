import dataclasses
from dataclasses import dataclass
from typing import List, Optional
import os

class Config:
    # logging
    LOG_RUN: bool = True

    TRAIN_LOCATION = "cluster"

    # dataset
    DATASET: str = "iNaturalist"
    TARGET_TYPE: Optional[str] = None
    DATA_LOADER_NUM_WORKERS: int = 8
    GLOBAL_DATA_DIR: Optional[str] = None
    LOCAL_DATA_DIR: Optional[str] = None

    RESULT_DIR: str = None

    EVALUATION_DISTANCE: str = "l2"
    EVALUATION_K: List[int] = None

    def __post_init__(self):
        match (self.TRAIN_LOCATION, self.DATASET):
            case ("local", "ImageNet"):
                pass
            case ("cluster", _):
                self.GLOBAL_DATA_DIR = os.path.join(os.getenv("WORK"), self.DATASET)
                self.LOCAL_DATA_DIR = os.path.join(os.getenv("TMPDIR"))
        
        match self.TRAIN_LOCATION:
            case "local":
                self.RESULT_DIR = os.path.join(r"D:\Dokumente\Studium\Masterprojekt\Gewichte", "vicreg")
            case "cluster":
                self.RESULT_DIR = os.path.join(os.getenv("HOME"), "weights", "vicreg")
        
        match self.DATASET:
            case "iNaturalist":
                self.TARGET_TYPE = "genus"
            case _:
                self.TARGET_TYPE = None

@dataclass
class VicRegConfig(Config):
    # models
    CLIP_MODEL_NAME: str = "ViT-B/32"
    DINO_MODEL_NAME: str = "dinov2_vitb14"
    CONVNEXT_MODEL_NAME: str = "convnextv2_nano.fcmae_ft_in22k_in1k"
    FUSION_HEAD: str = "LinearBase"
    LAMBDA_ALPHA_REG: float = 0
    USE_WEIGHTED_CONCAT: bool = False
    USE_DINO_CLS_AND_PATCH_TOKENS: bool = True
    USE_PROJ: bool = False
    PROJ_DIM: int = 512
    PROJECTION_HEAD_DIMS: List[int] = [512, 256] # [2048, 2048, 256]
    DIM: int = 1024

    # training
    BATCH_SIZE: int = 64
    EPOCHS: int = 5
    LR: float = 0.001
    WEIGHT_DECAY: float = 1e-6
    SCHEDULER: str = "cosine"
    SCHEDULER_MIN_LR: Optional[float] = None

    # loss coefficients
    SIM_COEFF: float = 25.0
    STD_COEFF: float = 25.0
    COV_COEFF: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        match self.SCHEDULER:
            case "cosine":
                self.SCHEDULER_MIN_LR = 0.0001
    
    def get_as_dict(self):
        return dataclasses.asdict(self)

    def get_as_wandb_dict(self):
        # return dataclasses.asdict(self)

        config = {
            "clip_model_name": self.CLIP_MODEL_NAME,
            "dino_model_name": self.DINO_MODEL_NAME,
            "convnext_model_name": self.CONVNEXT_MODEL_NAME,
            "fusion_head": self.FUSION_HEAD,
            "lambda_alpha_reg": self.LAMBDA_ALPHA_REG,
            "use_weighted_concat": self.USE_WEIGHTED_CONCAT,
            "use_dino_cls_and_patch_tokens": self.USE_DINO_CLS_AND_PATCH_TOKENS,
            "use_proj": self.USE_PROJ,
            "proj_dim": self.PROJ_DIM,
            "projection_head_dims": self.PROJECTION_HEAD_DIMS,
            "dataset": self.DATASET,
            "target_type": self.TARGET_TYPE,
            "batch_size": self.BATCH_SIZE,
            "epochs": self.EPOCHS,
            "learning_rate": self.LR,
            "sim_coeff": self.SIM_COEFF,
            "std_coeff": self.STD_COEFF,
            "cov_coeff": self.COV_COEFF,
            "optimizer_weight_decay": self.WEIGHT_DECAY,
            "scheduler": self.SCHEDULER,
            "scheduler_min_lr": self.SCHEDULER_MIN_LR,
            "dim": self.DIM,
        }

        return config