import torch
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix

from models import CLIP, DINO_v2, CombinedModel, DINO_CLIP
from datasets import Transforms


def create_model(model_info, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Create and return a model based on the provided model information.
    """

    match model_info["architecture"]:
        case "CLIP":
            model = CLIP(transform=Transforms.CLIP.value)
        case "DINO_v2":
            model = DINO_v2(transform=Transforms.DINO_v2.value)
        case "CombinedModel":
            model = CombinedModel(clip_transform=Transforms.CLIP.value, dino_transform=Transforms.DINO_v2.value)
        case "DINO_CLIP":
            model = DINO_CLIP(
                transform = Transforms.DINO_CLIP.value,
                clip_model_name = model_info["clip_model_name"],
                dino_model_name = model_info["dino_model_name"],
                clip_transform=Transforms.CLIP.value,
                dino_transform=Transforms.DINO_v2.value,
                dim=model_info["dim"],
                K=model_info["K"],
                # m=model_info["m"],
                # T=model_info["T"],
                fusion_head=model_info["fusion_head"]
            )
            if model_info["weights"] is not None:
                model.load_state_dict(torch.load(model_info["weights"]))
    
    for p in model.parameters():
        p.requires_grad = False
    model = model.eval()
    model = model.to(device)
    return model

def extract_features(model, dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Extract features and labels from the dataloader using the provided model.
    """

    features = []
    labels = []

    for images_batch, labels_batch in dataloader:
        images_batch = images_batch.to(device)
        features_batch = model(images_batch)
        
        features_batch = features_batch.cpu().tolist()
        labels_batch = labels_batch.tolist()

        features.extend(features_batch)
        labels.extend(labels_batch)
    
    return features, labels

def calculate_accuracy(I, labels, k):
    """
    Calculate overall accuracy and per-class accuracy.
    """

    I = I[:, :k]

    predictions_labels = labels[I]
    predictions = np.array([Counter(i).most_common(1)[0][0] for i in predictions_labels])
    
    # matches = predictions == labels
    # correct = matches.sum()
    # all_predictions = matches.size
    # accuracy2 = correct / all_predictions

    overall_accuracy = np.mean(predictions == labels)

    matrix = confusion_matrix(labels, predictions)
    accuracy = matrix.diagonal()/matrix.sum(axis=1)

    return overall_accuracy, accuracy.mean(), accuracy.std(), accuracy.tolist(), matrix.tolist()

def calculate_precision(I, labels, k):
    """
    Calculate per-class precision.
    """

    I = I[:, :k]

    predictions_labels = labels[I]
    predictions = np.array([Counter(i).most_common(1)[0][0] for i in predictions_labels])
    tp, fp, _, _ = confusion_per_class(predictions, labels)

    matrix = confusion_matrix(labels, predictions)

    precision = tp / (tp + fp + 1e-8)
    return precision.mean(), precision.std(), precision.tolist(), matrix.tolist()

def calculate_recall(I, labels, k):
    """
    Calculate per-class recall.
    """

    I = I[:, :k]

    predictions_labels = labels[I]
    predictions = np.array([Counter(i).most_common(1)[0][0] for i in predictions_labels])
    tp, _, _, fn = confusion_per_class(predictions, labels)

    matrix = confusion_matrix(labels, predictions)

    recall = tp / (tp + fn + 1e-8)
    return recall.mean(), recall.std(), recall.tolist(), matrix.tolist()

def confusion_per_class(predictions, labels):
    """
    Helper method to compute true positives, false positives, true negatives, and false negatives per class.
    """

    num_classes = max(labels.max(), predictions.max()) + 1
    tp = np.zeros(num_classes, dtype=int)
    fp = np.zeros(num_classes, dtype=int)
    fn = np.zeros(num_classes, dtype=int)
    tn = np.zeros(num_classes, dtype=int)
    
    for c in range(num_classes):
        tp[c] = np.sum((predictions == c) & (labels == c))
        fp[c] = np.sum((predictions == c) & (labels != c))
        tn[c] = np.sum((predictions != c) & (labels != c))
        fn[c] = np.sum((predictions != c) & (labels == c))
    
    return tp, fp, tn, fn
