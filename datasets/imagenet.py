import torchvision
from torch.utils import data
import os
import random
from abc import ABC, abstractmethod
import csv
from PIL import Image

from .dataset_utils import Transforms


class AbstractImageNet(data.Dataset, ABC):
    def __init__(
            self,
            root:str = r"C:\Users\mariu\Documents\Development\Datasets\imagenet-object-localization-challenge",
            split:str = "train",
            transform = Transforms.DEFAULT.value
        ):

        super().__init__()

        if split != "train" and split != "val":
            raise ValueError("Split can only be train or val")

        self.root = root
        self.split = split
        self.transform = transform

        self.class_to_idx = {}
        self.idx_to_class = {}
        self.id_to_idx = {}
        self.id_to_class = {}
        file = open(os.path.join(self.root, "LOC_synset_mapping.txt"), "r")
        for i, line in enumerate(file.readlines()):
            synset_id = line.split()[0]
            c = " ".join(line.split(", ")[0].split()[1:])
            self.class_to_idx[c] = i
            self.idx_to_class[i] = c
            self.id_to_idx[synset_id] = i
            self.id_to_class[synset_id] = c


        image_folder = os.path.join(self.root, "ILSVRC", "Data", "CLS-LOC", split)
        
        self.len = 0
        self.class_to_idxs = {}
        self.imgs = []
        if split == "train":
            synset_ids = os.listdir(image_folder)
            for synset_id in synset_ids:
                l = len(os.listdir(os.path.join(image_folder, synset_id)))
                if l == 0:
                    self.class_to_idxs[self.id_to_class[synset_id]] = {
                        "n": 0,
                        "start": None,
                        "end": None
                    }
                else:
                    self.class_to_idxs[self.id_to_class[synset_id]] = {
                        "n": l,
                        "start": self.len,
                        "end": self.len + l - 1
                    }
                    self.len += l
            
            self.imagenet = torchvision.datasets.ImageFolder(image_folder, transform=transform, allow_empty=True)

            self.imgs = self.imagenet.imgs
        else:
            file = open(os.path.join(self.root, "LOC_val_solution.csv"))
            csv_reader = csv.reader(file)
            _ = next(csv_reader)
            for row in csv_reader:
                image_name, synset_id = row[0], row[1].split()[0]
                self.imgs.append((os.path.join(image_folder, image_name + ".JPEG"), self.id_to_idx[synset_id]))
            self.len = len(self.imgs)
    
    def __len__(self):
        return self.len

    @abstractmethod
    def __getitem__(self, index):
        pass


class ImageNet(AbstractImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        if self.split == "train":
            image, label = self.imagenet[index]
        else:
            image_path, label = self.imgs[index]
            image = Image.open(image_path).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        return image, label
    
    def getSubset(self, image_num, classes=None):
        if self.split == "val":
            raise ValueError("Subset for validation split not supported yet")
        
        if classes == None:
            indices = random.sample(range(0, self.__len__()), min(image_num, self.__len__()))
            return data.Subset(self, indices) #, data.Subset(self.untransformed_image_folder, indices)

        a = image_num // len(classes)
        b = image_num % len(classes)
        images_per_class = [a for _ in range(len(classes) - 1)] + [a + b]

        indices = []

        for l, num in zip(classes, images_per_class):
            if isinstance(l, int):
                l = self.idx_to_class[l]
            start, end = self.class_to_idxs[l]["start"], self.class_to_idxs[l]["end"]
            if start is None or end is None:
                continue
            indices = indices + random.sample(range(start, end+1), min(num, (end+1)-start))
        
        return data.Subset(self, indices) #, data.Subset(self.untransformed_image_folder, indices)