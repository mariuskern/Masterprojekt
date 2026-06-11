# import torchvision
from torch.utils import data
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from .github.inaturalist.inaturalist_without_integrity_check import INaturalist as INaturalist_pytorch

from .dataset_utils import Transforms


class AbstractINaturalist(data.Dataset, ABC):
    def __init__(
            self,
            root: str = r"D:\Dokumente\Development\Datasets",
            split: str = "train",
            target_type: str = "genus",
            transform = Transforms.DEFAULT.value,
        ):

        super().__init__()

        '''
        kingdom, phylum, class, order, family, genus, species

        all_categories: ['00000_Animalia_Annelida_Clitellata_Haplotaxida_Lumbricidae_Lumbricus_terrestris', '00001_Animalia_Annelida_Polychaeta_Sabellida_Sabellidae_Sabella_spallanzanii', ...]
        categories_index: {'kingdom': {'Animalia': 0, 'Fungi': 1, 'Plantae': 2}, 'phylum': {'Annelida': 0, 'Arthropoda': 1, 'Chordata': 2, 'Cnidaria': 3, ...}, ...}
        categories_map: [{'kingdom': 0, 'phylum': 0, 'class': 0, 'order': 0, 'family': 0, 'genus': 0}, {'kingdom': 0, 'phylum': 0, 'class': 1, 'order': 1, 'family': 1, 'genus': 1}, ...]
        '''

        if split != "train" and split != "train_mini" and split != "val":
            raise ValueError("Split can only be train or val")
        
        if split == "train":
            raise ValueError("Train split not downloaded")

        self.root = root
        self.split = split
        self.target_type = target_type
        self.transform = transform

        if self.split == "train":
            self.version = "train"
        elif self.split == "train_mini":
            self.version = "2021_train_mini"
        else:
            self.version = "2021_valid"

        # self.iNaturalist = torchvision.datasets.INaturalist(root=self.root, version=self.version, target_type=self.target_type, transform=self.transform, download=False)
        self.iNaturalist = INaturalist_pytorch(root=self.root, version=self.version, target_type=self.target_type, transform=self.transform, download=False)
        
        self.class_to_idx = {}
        self.idx_to_class = {}
        for key, value in self.iNaturalist.categories_index[self.target_type].items():
            self.class_to_idx[key] = value
            self.idx_to_class[value] = key
        
        self.len = 0

        self.class_to_idxs = {}
        self.imgs = []

        self.tmp = {
            "class_to_idxs": defaultdict(list),
        }

        for i, category in enumerate(self.iNaturalist.all_categories):
            l = len(os.listdir(os.path.join(self.root, self.version, category)))

            for img in os.listdir(os.path.join(self.root, self.version, category)):
                self.imgs.append(os.path.join(self.version, category, img))

            start = self.len
            end = self.len + l - 1

            category_map = self.iNaturalist.categories_map[i]
            idx = category_map[self.target_type]
            c = self.idx_to_class[idx]

            self.tmp["class_to_idxs"][c].extend(range(start, end + 1))
            
            self.len += l
    
        self.class_to_idxs = self._build_range_dict(self.tmp["class_to_idxs"])

    def _build_range_dict(self, src):
        out = {}

        for key, idxs in src.items():
            out[key] = {
                "n": len(idxs),
                "start": min(idxs),
                "end": max(idxs)
            }

        return out
    
    def __len__(self):
        return self.len
    
    @abstractmethod
    def __getitem__(self, index):
        pass

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

class INaturalist(AbstractINaturalist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        return self.iNaturalist[index]

class INaturalistPath(AbstractINaturalist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        _, target = self.iNaturalist[index]
        path = self.imgs[index]

        return path, target
