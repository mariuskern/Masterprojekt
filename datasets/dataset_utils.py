import torchvision
from enum import Enum


class Transforms(Enum):
    DINO_v2 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        # torchvision.transforms.ToTensor(),
    ])

    CLIP = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        torchvision.transforms.CenterCrop(size=(224, 224)),
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    DINO_CLIP = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    T224 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
    ])

    DEFAULT = torchvision.transforms.Compose([
        torchvision.transforms.Resize((400, 400)),
        torchvision.transforms.ToTensor(),
    ])