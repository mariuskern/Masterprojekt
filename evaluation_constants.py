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
        "name": "DINO_CLIP (trained on Places365, linear)",
        "architecture": "DINO_CLIP",
        "weights": r"C:\Users\mariu\Documents\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260311_222710\state_dict_epoch_2_batch_570_batch_global_56358.pt",
        "clip_model_name": "ViT-B/32",
        "dino_model_name": "dinov2_vitb14",
        "dim": 1024,
        "K": 16384,
        "fusion_head": "Linear",
        "dataset": ["Places365", "ArtPlaces"]
    },
    {
        "name": "DINO_CLIP (trained on Places365, attention)",
        "architecture": "DINO_CLIP",
        "weights": r"C:\Users\mariu\Documents\Studium\Masterprojekt\Gewichte\dino_clip\dinov2_vitb14_vit-b32_places365_20260312_113216\state_dict_epoch_2_batch_570_batch_global_56358.pt",
        "clip_model_name": "ViT-B/32",
        "dino_model_name": "dinov2_vitb14",
        "dim": 1024,
        "K": 16384,
        "fusion_head": "Attention",
        "dataset": ["Places365", "ArtPlaces"]
    },
]
