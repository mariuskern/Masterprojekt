DEST = r"C:\Users\mariu\Documents\Studium\Masterprojekt\Evaluation"

DISTANCE = "l2"

# K = [1, 5, 10, 15, 20, 50]
K = [1, 5]

DATASETS = ["ImageNet", "Places365", "ArtPlaces"]
# DATASETS = ["ArtPlaces"]

MODELS = [
    {
        "name": "CLIP",
        "architecture": "CLIP",
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
    {
        "name": "DINO_v2",
        "architecture": "DINO_v2",
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
    {
        "name": "CombinedModel",
        "architecture": "CombinedModel",
        "dataset": ["ImageNet", "Places365", "ArtPlaces"]
    },
    {
        "name": "Linear Base (CLS & Patch)",
        "architecture": "DINO_CLIP",
        "weights": r"C:\Users\mariu\Documents\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260515_100629\state_dict_epoch_1_batch_570_batch_global_10240.pt",
        "clip_model_name": "ViT-B/32",
        "dino_model_name": "dinov2_vitb14",
        "dim": 512,
        "K": 8192,
        "fusion_head": "LinearBase",
        "use_weighted_concat": False,
        "use_dino_cls_and_patch_tokens": True,
        "use_proj": False,
        "proj_dim": 0,
        "dataset": ["Places365", "ArtPlaces"]
    },
    {
        "name": "Linear Base (CLS & Patch, Three Alphas)",
        "architecture": "DINO_CLIP",
        "weights": r"C:\Users\mariu\Documents\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260514_225944\state_dict_epoch_1_batch_570_batch_global_28179.pt",
        "clip_model_name": "ViT-B/32",
        "dino_model_name": "dinov2_vitb14",
        "dim": 512,
        "K": 8192,
        "fusion_head": "LinearBaseOldThreeAlphas",
        "use_weighted_concat": True,
        "use_dino_cls_and_patch_tokens": True,
        "use_proj": False,
        "proj_dim": 0,
        "dataset": ["Places365", "ArtPlaces"]
    },
    {
        "name": "Linear Small (14.05. 10:37)",
        "architecture": "DINO_CLIP",
        "weights": r"C:\Users\mariu\Documents\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260514_103708\state_dict_epoch_1_batch_570_batch_global_28179.pt",
        "clip_model_name": "ViT-B/32",
        "dino_model_name": "dinov2_vitb14",
        "dim": 512,
        "K": 8192,
        "fusion_head": "LinearSmallOldOneAlpha",
        "use_weighted_concat": False,
        "use_dino_cls_and_patch_tokens": False,
        "use_proj": False,
        "proj_dim": 0,
        "dataset": ["Places365", "ArtPlaces"]
    },
    {
        "name": "Linear Base (One Alpha)",
        "architecture": "DINO_CLIP",
        "weights": r"C:\Users\mariu\Documents\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260511_205255\state_dict_epoch_1_batch_570_batch_global_28179.pt",
        "clip_model_name": "ViT-B/32",
        "dino_model_name": "dinov2_vitb14",
        "dim": 512,
        "K": 8192,
        "fusion_head": "LinearBaseOldOneAlpha",
        "use_weighted_concat": True,
        "use_dino_cls_and_patch_tokens": False,
        "use_proj": False,
        "proj_dim": 0,
        "dataset": ["Places365", "ArtPlaces"]
    },
    {
        "name": "Linear Base (10.05. 15:42)",
        "architecture": "DINO_CLIP",
        "weights": r"C:\Users\mariu\Documents\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260510_154300\state_dict_epoch_1_batch_570_batch_global_28179.pt",
        "clip_model_name": "ViT-B/32",
        "dino_model_name": "dinov2_vitb14",
        "dim": 512,
        "K": 8192,
        "fusion_head": "LinearBaseOldNoAlpha",
        "use_weighted_concat": False,
        "use_dino_cls_and_patch_tokens": False,
        "use_proj": False,
        "proj_dim": 0,
        "dataset": ["Places365", "ArtPlaces"]
    },
    {
        "name": "Linear Base (Old Training)",
        "architecture": "DINO_CLIP",
        "weights": r"C:\Users\mariu\Documents\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260509_155607\state_dict_25.pt",
        "clip_model_name": "ViT-B/32",
        "dino_model_name": "dinov2_vitb14",
        "dim": 512,
        "K": 8192,
        "fusion_head": "LinearBaseOldNoAlpha",
        "use_weighted_concat": False,
        "use_dino_cls_and_patch_tokens": False,
        "use_proj": False,
        "proj_dim": 0,
        "dataset": ["Places365", "ArtPlaces"]
    },
    {
        "name": "Attention (Old)",
        "architecture": "DINO_CLIP",
        "weights": r"C:\Users\mariu\Documents\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260315_113734\state_dict_epoch_2_batch_570_batch_global_56358.pt",
        "clip_model_name": "ViT-B/32",
        "dino_model_name": "dinov2_vitb14",
        "dim": 512,
        "K": 8192,
        "fusion_head": "Attention",
        "use_weighted_concat": False,
        "use_dino_cls_and_patch_tokens": False,
        "use_proj": False,
        "proj_dim": 0,
        "dataset": ["Places365", "ArtPlaces"]
    },
    {
        "name": "Linear Base (14.03. 17:14)",
        "architecture": "DINO_CLIP",
        "weights": r"C:\Users\mariu\Documents\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260314_174008\state_dict_epoch_2_batch_570_batch_global_56358.pt",
        "clip_model_name": "ViT-B/32",
        "dino_model_name": "dinov2_vitb14",
        "dim": 512,
        "K": 8192,
        "fusion_head": "LinearBaseOldNoAlpha",
        "use_weighted_concat": False,
        "use_dino_cls_and_patch_tokens": False,
        "use_proj": False,
        "proj_dim": 0,
        "dataset": ["Places365", "ArtPlaces"]
    },
]
