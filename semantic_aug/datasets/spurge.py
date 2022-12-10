from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.semantic_augmentation import (
    SemanticAugmentation,
    Identity
)
from typing import Any, Tuple
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
import glob
import numpy as np


class SpurgeDataset(FewShotDataset):

    def __init__(self, split: str = "train", seed: int = 0, 
                 examples_per_class: int = None, 
                 synthetic_aug: SemanticAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 *args, **kwargs):

        super(SpurgeDataset, self).__init__(
            examples_per_class=examples_per_class,
            synthetic_aug=synthetic_aug,
            synthetic_probability=synthetic_probability, 
            *args, **kwargs)

        data_dir = os.path.join(
            os.path.abspath(os.path.dirname(
            os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))), 'data')

        absent = list(glob.glob(os.path.join(data_dir, "spurge/absent/*.png")))
        apparent = list(glob.glob(os.path.join(data_dir, "spurge/apparent/*.png")))

        rng = np.random.default_rng(seed)

        absent_ids = rng.permutation(len(absent))
        apparent_ids = rng.permutation(len(apparent))

        absent_ids_train, absent_ids_val = np.array_split(absent_ids, 2)
        apparent_ids_train, apparent_ids_val = np.array_split(apparent_ids, 2)

        absent_ids = {"train": absent_ids_train, "val": absent_ids_val}[split]
        apparent_ids = {"train": apparent_ids_train, "val": apparent_ids_val}[split]

        if examples_per_class is not None:
            absent_ids = absent_ids[:examples_per_class]
            apparent_ids = apparent_ids[:examples_per_class]

        self.absent = [absent[i] for i in absent_ids]
        self.apparent = [apparent[i] for i in apparent_ids]

        self.all_images = self.absent + self.apparent
        self.all_labels = [0] * len(self.absent) + [1] * len(self.apparent)

        train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
        ])

        self.transform = {"train": train_transform, "val": val_transform}[split]

    def __len__(self):

        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> torch.Tensor:
        
        return Image.open(self.all_images[idx])
    
    def get_metadata_by_idx(self, idx: int) -> Any:

        return dict(label=self.all_labels[idx], token_name=(
            "<leafy_spurge>" if self.all_labels[idx] == 1 else "<no_spurge>"
        ))