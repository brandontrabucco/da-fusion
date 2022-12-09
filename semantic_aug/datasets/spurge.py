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

    def __init__(self, split: str, examples_per_class: int, seed: int = 0, 
                 transform: SemanticAugmentation = Identity, *args, **kwargs):

        super(SpurgeDataset, self).__init__(examples_per_class, transform=transform, *args, **kwargs)

        data_dir = os.path.join(
            os.path.abspath(os.path.dirname(
            os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))), 'data')

        absent = list(glob.glob(os.path.join(data_dir, "spurge/absent/*.png")))
        apparent = list(glob.glob(os.path.join(data_dir, "spurge/apparent/*.png")))

        rng = np.random.default_rng(seed)

        absent_ids = rng.permutation(len(absent))
        apparent_ids = rng.permutation(len(apparent))

        absent_ids_train, absent_ids_val, absent_ids_test = np.array_split(absent_ids, 3)
        apparent_ids_train, apparent_ids_val, apparent_ids_test = np.array_split(apparent_ids, 3)

        absent_ids = {"train": absent_ids_train, "val": absent_ids_val, "test": absent_ids_test}[split]
        apparent_ids = {"train": apparent_ids_train, "val": apparent_ids_val, "test": apparent_ids_test}[split]

        train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = {
            "train": train_transform, 
            "val": val_test_transform, 
            "test": val_test_transform
        }[split]

        self.absent = [absent[i] for i in absent_ids[:examples_per_class]]
        self.apparent = [apparent[i] for i in apparent_ids[:examples_per_class]]

        self.all_images = self.absent + self.apparent
        self.all_classes = [0] * examples_per_class + [1] * examples_per_class

    def __len__(self):

        return 2 * self.examples_per_class

    def filter_by_class(self, class_idx: int) -> torch.Tensor:

        images = [self.absent, self.apparent][class_idx]
        images = [self.transform(Image.open(x)) for x in images]
        return torch.stack(images, dim=0)
    
    def get_image_by_idx(self, idx: int) -> torch.Tensor:
        
        return self.transform(Image.open(self.all_images[idx]))
    
    def get_metadata_by_idx(self, idx: int) -> Any:

        return dict(label=self.all_classes[idx], token_name=(
            "<leafy_spurge>" if self.all_classes[idx] == 1 else "<no_spurge>"
        ))