import torchvision
from torch.utils import data
import os
import random
from enum import Enum
from abc import ABC, abstractmethod
import csv
from PIL import Image
from .github.places365_without_integrity_check import Places365 as Places365_pytorch

from .dataset_utils import Transforms


class AbstractPlaces365(data.Dataset, ABC):
    def __init__(
            self,
            root: str = r"C:\Users\mariu\Documents\Development\Datasets\Places365",
            split: str = "train",
            transform = Transforms.DEFAULT.value,
        ):

        super().__init__()

        if split != "train" and split != "val":
            raise ValueError("Split can only be train or val")

        if split == "train":
            split = "train-standard"

        self.root = root
        self.split = split
        self.transform = transform

        self.places365 = Places365_pytorch(root=self.root, split=self.split, transform=self.transform)

        self.class_to_idx = {}
        self.idx_to_class = {}
        for key, value in self.places365.class_to_idx.items():
            key = key.split("/", 2)[2]

            self.class_to_idx[key] = value
            self.idx_to_class[value] = key
        
        self.len = 0
        self.class_to_idxs = {}
        if self.split == "train-standard":
            letters = os.listdir(os.path.join(root, "data_large_standard"))
            for letter in letters:
                self.class_to_idxs.update(self._get_class_to_idxs_by_letter(root, letter))
        elif self.split == "val":
            self.len = len(self.places365)
        
        self.imgs = self.places365.imgs
    
    def _get_class_to_idxs_by_letter(self, root, letter):
        class_to_idxs_part = {}

        for class_part in os.listdir(os.path.join(root, "data_large_standard", letter)):
            p = self._get_class_to_idxs_by_letter_helper(root, letter, [class_part])
            class_to_idxs_part.update(p)
        
        return class_to_idxs_part
    
    def _get_class_to_idxs_by_letter_helper(self, root, letter, class_parts):
        class_to_idxs_part = {}

        c = os.listdir(os.path.join(root, "data_large_standard", letter, *class_parts))
        if len(c) == 0:
            class_to_idxs_part["/".join(class_parts)] = {
                "n": 0,
                "start": None,
                "end": None
            }
            return class_to_idxs_part
        elif os.path.isfile(os.path.join(root, "data_large_standard", letter, *class_parts, c[0])):
            n = len(c)
            class_to_idxs_part["/".join(class_parts)] = {
                "n": n,
                "start": self.len,
                "end": self.len + n - 1
            }
            self.len += n
            return class_to_idxs_part

        for class_part in c:
            class_to_idxs_part.update(self._get_class_to_idxs_by_letter_helper(root, letter, class_parts + [class_part]))
        
        return class_to_idxs_part
    
    def __len__(self):
        return self.len
    
    @abstractmethod
    def __getitem__(self, index):
        pass

class Places365(AbstractPlaces365):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        return self.places365[index]
    
    def getSubset(self, image_num, classes=None):
        if classes == None:
            indices = random.sample(range(0, self.__len__()), min(image_num, self.__len__()))
            return data.Subset(self, indices)

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
        
        return data.Subset(self, indices)