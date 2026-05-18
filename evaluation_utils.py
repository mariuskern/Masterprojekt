import torch
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import faiss

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
                convnext_model_name = model_info["convnext_model_name"],
                clip_transform=Transforms.CLIP.value,
                dino_transform=Transforms.DINO_v2.value,
                convnext_transform = Transforms.CONVNEXT.value,
                dim=model_info["dim"],
                K=model_info["K"],
                # m=model_info["m"],
                # T=model_info["T"],
                fusion_head=model_info["fusion_head"],
                use_weighted_concat = model_info["use_weighted_concat"],
                use_dino_cls_and_patch_tokens = model_info["use_dino_cls_and_patch_tokens"],
                use_proj = model_info["use_proj"],
                proj_dim = model_info["proj_dim"]
            )
            if model_info["weights"] is not None:
                model.load_state_dict(torch.load(model_info["weights"]))
    
    for p in model.parameters():
        p.requires_grad = False
    model = model.eval()
    model = model.to(device)
    return model

def extract_and_evaluate_features(model, model_name, dataloader, dataset_name, distance, ks):
    tqdm_dataloader = tqdm(dataloader, unit="batch")
    tqdm_dataloader.set_description(f"Extracting features ({model_name}, {dataset_name})")

    features, labels = extract_features(model, tqdm_dataloader)

    result = evaluate_features(features, labels, distance, ks)

    return result

def extract_features(model, dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Extract features and labels from the dataloader using the provided model. Returns features and labels as a numpy array.
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

    features = np.array(features).astype("float32")
    labels = np.array(labels)
    
    return features, labels

def evaluate_features(features, labels, distance, ks):
    result = {
        "per_class": {
            "mean": {},
            "std": {},
            "matrix": {}
        }
    }

    match distance:
        case "l2":
            index = faiss.IndexFlatL2(features.shape[1])
        case "cosine":
            faiss.normalize_L2(features)
            index = faiss.IndexFlatIP(features.shape[1])
    
    index.add(features)

    _, I = index.search(features, max(ks) + 1)  # + 1 to account for self-match
    I = I[:, 1:]  # Remove self-match

    for k in ks:
        overall_accuracy, accuracy_mean, accuracy_std, accuracy_per_class, accuracy_matrix = calculate_accuracy(I, labels, k)
        precision_mean, precision_std, precision_per_class, precision_matrix = calculate_precision(I, labels, k)
        recall_mean, recall_std, recall_per_class, recall_matrix = calculate_recall(I, labels, k)

        result["Accuracy@" + str(k)] = overall_accuracy
        result["per_class"]["mean"]["Accuracy@" + str(k)] = accuracy_mean
        result["per_class"]["mean"]["Precision@" + str(k)] = precision_mean
        result["per_class"]["mean"]["Recall@" + str(k)] = recall_mean
        result["per_class"]["std"]["Accuracy@" + str(k)] = accuracy_std
        result["per_class"]["std"]["Precision@" + str(k)] = precision_std
        result["per_class"]["std"]["Recall@" + str(k)] = recall_std
        result["per_class"]["matrix"]["Accuracy@" + str(k)] = accuracy_matrix
        result["per_class"]["matrix"]["Precision@" + str(k)] = precision_matrix
        result["per_class"]["matrix"]["Recall@" + str(k)] = recall_matrix
        result["per_class"]["Accuracy@" + str(k)] = accuracy_per_class
        result["per_class"]["Precision@" + str(k)] = precision_per_class
        result["per_class"]["Recall@" + str(k)] = recall_per_class
    
    return result

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
