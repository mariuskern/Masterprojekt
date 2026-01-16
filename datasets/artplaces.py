from torch.utils import data
import os
import random
from abc import ABC, abstractmethod
import csv
from PIL import Image

from .dataset_utils import Transforms


class AbstractArtPlaces(data.Dataset, ABC):
    def __init__(
            self,
            root:str = r"C:\Users\mariu\Documents\Development\Datasets\ArtPlaces_13371280",
            split:str = "train",
            transform = Transforms.DEFAULT.value
    ):
        super().__init__()

        if split != "train" and split != "val":
            raise ValueError("Split can only be train or val")
        
        self.root = root
        self.split = split
        self.transform = transform
        
        match split:
            case "train":
                artplace_csv = os.path.join(root, "Artplace_Train.csv")
            case "val":
                artplace_csv = os.path.join(root, "Artplace_Test.csv")

        self.class_to_idx = {}
        self.idx_to_class = {}
        categories = open(os.path.join(root, "categories_places365.txt"), "r")
        for line in categories:
            key, value = line.split(" ")
            key = key.split("/", 2)[2]

            self.class_to_idx[key] = int(value)
            self.idx_to_class[int(value)] = key
        
        self.len = 0
        self.imgs = []

        file = open(artplace_csv, "r")
        reader = csv.reader(file)
        _ = next(reader) # skip header
        for row in reader:
            image_name, label = row
            label = int(label)
            
            if "WASD" in image_name:
                image_name = os.path.join("WASD", "wikidata", image_name.split("/")[2])
            self.imgs.append((os.path.join(root, image_name), label))
        
        self.imgs.sort(key=lambda x: x[1])

        self.class_to_idxs = {}

        for class_name in self.class_to_idx.keys():
            self.class_to_idxs[class_name] = {
                "n": 0,
                "start": None,
                "end": None
            }

        for i, (image_name, label) in enumerate(self.imgs):
            class_name = self.idx_to_class[label]

            self.class_to_idxs[class_name]["n"] += 1
            if self.class_to_idxs[class_name]["start"] is None:
                self.class_to_idxs[class_name]["start"] = i
            self.class_to_idxs[class_name]["end"] = i
                
        self.len = len(self.imgs)
    
    def __len__(self):
        return self.len

    @abstractmethod
    def __getitem__(self, index):
        pass

class ArtPlaces(AbstractArtPlaces):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        image_path, label = self.imgs[index]
        image = Image.open(os.path.abspath(image_path)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
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