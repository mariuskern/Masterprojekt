DEST = r"D:\Dokumente\Studium\Masterprojekt\Evaluation"

DISTANCE = "l2"

# K = [1, 5, 10, 15, 20, 50]
K = [1, 5]

# DATASETS = ["ImageNet", "Places365", "ArtPlaces", "iNaturalist_Family"]
DATASETS = ["iNaturalist_Family"]

MODELS = [
    {
        "name": "CLIP",
        "architecture": "CLIP"
    },
    {
        "name": "DINO_v2",
        "architecture": "DINO_v2",
    },
    {
        "name": "ConvNeXt_v2",
        "architecture": "ConvNeXt_v2",
    },
    {
        "name": "CombinedModel",
        "architecture": "CombinedModel",
    },

    # # ImageNet Final
    # {
    #     "name": "Linear MoCo",
    #     "architecture": "MoCo",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_imagenet_20260607_084618\state_dict_epoch_1_batch_781_batch_global_20018.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "K": 8192,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "pooling": None,
    #     "projection_head_dims": None
    # },
    # {
    #     "name": "Linear VICReg",
    #     "architecture": "VICReg",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\vicreg\dinov2_vitb14_vit-b32_imagenet_20260606_200835\state_dict_epoch_1_batch_781_batch_global_20018.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "pooling": None,
    #     "projection_head_dims": [2048, 2048, 256]
    # },

    # # Places365 Final
    # {
    #     "name": "Linear MoCo",
    #     "architecture": "MoCo",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260606_103020\state_dict_epoch_1_batch_570_batch_global_28179.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "K": 8192,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "pooling": None,
    #     "projection_head_dims": None
    # },
    # {
    #     "name": "Linear VICReg",
    #     "architecture": "VICReg",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\vicreg\dinov2_vitb14_vit-b32_places365_20260606_004644\state_dict_epoch_1_batch_570_batch_global_28179.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "pooling": None,
    #     "projection_head_dims": [2048, 2048, 256]
    # },

    # # ArtPlaces Final
    # {
    #     "name": "Linear MoCo",
    #     "architecture": "MoCo",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_artplaces_20260605_191236\state_dict_epoch_75_batch_71_batch_global_5325.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "K": 1024,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "pooling": None,
    #     "projection_head_dims": None
    # },
    # {
    #     "name": "Linear VICReg",
    #     "architecture": "VICReg",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\vicreg\dinov2_vitb14_vit-b32_artplaces_20260604_181317\state_dict_epoch_75_batch_71_batch_global_5325.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "pooling": None,
    #     "projection_head_dims": [2048, 2048, 256]
    # },

    # # iNaturlaist Final
    # {
    #     "name": "Linear MoCo",
    #     "architecture": "MoCo",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_inaturalist_20260608_175736\state_dict_epoch_4_batch_350_batch_global_31248.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "K": 8192,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "pooling": None,
    #     "projection_head_dims": None
    # },
    # {
    #     "name": "Linear VICReg",
    #     "architecture": "VICReg",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\vicreg\dinov2_vitb14_vit-b32_inaturalist_20260607_221420\state_dict_epoch_4_batch_350_batch_global_31248.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "pooling": None,
    #     "projection_head_dims": [2048, 2048, 256]
    # },









    # # Places365
    # {
    #     "name": "Linear Base (CLS & Patch)",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260515_100629\state_dict_epoch_1_batch_570_batch_global_10240.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": None,
    #     "dim": 512,
    #     "K": 8192,
    #     "fusion_head": "LinearBase",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "dataset": ["Places365", "ArtPlaces"]
    # },
    # {
    #     "name": "Linear Base (CLS & Patch, Three Alphas)",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260514_225944\state_dict_epoch_1_batch_570_batch_global_28179.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": None,
    #     "dim": 512,
    #     "K": 8192,
    #     "fusion_head": "LinearBaseOldThreeAlphas",
    #     "use_weighted_concat": True,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "dataset": ["Places365", "ArtPlaces"]
    # },
    # {
    #     "name": "Linear Small (14.05. 10:37)",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260514_103708\state_dict_epoch_1_batch_570_batch_global_28179.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": None,
    #     "dim": 512,
    #     "K": 8192,
    #     "fusion_head": "LinearSmallOldOneAlpha",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": False,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "dataset": ["Places365", "ArtPlaces"]
    # },
    # {
    #     "name": "Linear Base (One Alpha)",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260511_205255\state_dict_epoch_1_batch_570_batch_global_28179.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": None,
    #     "dim": 512,
    #     "K": 8192,
    #     "fusion_head": "LinearBaseOldOneAlpha",
    #     "use_weighted_concat": True,
    #     "use_dino_cls_and_patch_tokens": False,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "dataset": ["Places365", "ArtPlaces"]
    # },
    # {
    #     "name": "Linear Base (10.05. 15:42)",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260510_154300\state_dict_epoch_1_batch_570_batch_global_28179.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": None,
    #     "dim": 512,
    #     "K": 8192,
    #     "fusion_head": "LinearBaseOldNoAlpha",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": False,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "dataset": ["Places365", "ArtPlaces"]
    # },
    # {
    #     "name": "Linear Base (Old Training)",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260509_155607\state_dict_25.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": None,
    #     "dim": 512,
    #     "K": 8192,
    #     "fusion_head": "LinearBaseOldNoAlpha",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": False,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "dataset": ["Places365", "ArtPlaces"]
    # },
    # {
    #     "name": "Attention (Old)",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260315_113734\state_dict_epoch_2_batch_570_batch_global_56358.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": None,
    #     "dim": 512,
    #     "K": 8192,
    #     "fusion_head": "Attention",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": False,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "dataset": ["Places365", "ArtPlaces"]
    # },
    # {
    #     "name": "Linear Base (14.03. 17:14)",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260314_174008\state_dict_epoch_2_batch_570_batch_global_56358.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": None,
    #     "dim": 512,
    #     "K": 8192,
    #     "fusion_head": "LinearBaseOldNoAlpha",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": False,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "dataset": ["Places365", "ArtPlaces"]
    # },

    # iNaturalist
    # {
    #     "name": "Linear Small (CLS + Patch, iNaturalist) Hellblau",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_inaturalist_20260518_000247\state_dict_epoch_2_batch_350_batch_global_10240.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "K": 8192,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 512,
    #     "dataset": ["iNaturalist_Family"]
    # },
    # {
    #     "name": "Linear Small (CLS + Patch, iNaturalist)",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_inaturalist_20260518_000247\state_dict_epoch_4_batch_350_batch_global_30720.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "K": 8192,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    #     "dataset": ["iNaturalist_Family"]
    # },
    # {
    #     "name": "Linear Small (CLS + Patch, Proj, Alphas, iNaturalist) Orange",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_inaturalist_20260517_020531\state_dict_epoch_5_batch_350_batch_global_39060.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "K": 16384,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": True,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": True,
    #     "proj_dim": 512,
    #     "dataset": ["iNaturalist_Family"]
    # },
    # {
    #     "name": "Linear Small (CLS + Patch, Proj, Alphas, iNaturalist) Aqua",
    #     "architecture": "DINO_CLIP",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_inaturalist_20260516_182956\state_dict_epoch_3_batch_350_batch_global_20480.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "K": 8192,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": True,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": True,
    #     "proj_dim": 512,
    #     "dataset": ["iNaturalist_Family"]
    # },

    # VICReg
    # {
    #     "name": "VICReg (18.05.2026 - 20:21)",
    #     "architecture": "VICReg",
    #     "weights": r"D:\Dokumente\Studium\Masterprojekt\Gewichte\vicreg\dinov2_vitb14_vit-b32_places365_20260518_202116\state_dict_epoch_2_batch_570_batch_global_51200.pt",
    #     "clip_model_name": "ViT-B/32",
    #     "dino_model_name": "dinov2_vitb14",
    #     "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
    #     "dim": 1024,
    #     "fusion_head": "LinearSmall",
    #     "use_weighted_concat": False,
    #     "use_dino_cls_and_patch_tokens": True,
    #     "use_proj": False,
    #     "proj_dim": 0,
    # },
]
