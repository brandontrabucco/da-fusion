from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple
from torch.utils.data import Dataset
from PIL import Image

import os
import glob
import numpy as np
import torchvision.transforms as transforms
import torch


DEFAULT_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), 'data/spurge')


class SpurgeDataset(FewShotDataset):

    num_classes: int = 2
    class_names = ["absent", "present"] #simplifying class names and aligning with variables throughout

    def __init__(self, *args, data_dir: str = DEFAULT_DATA_DIR, 
                 split: str = "train", seed: int = 0, 
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256), **kwargs):

        super(SpurgeDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug, **kwargs)

        absent = list(glob.glob(os.path.join(data_dir, "absent/*.tif")))
        present = list(glob.glob(os.path.join(data_dir, "present/*.tif")))

        rng = np.random.default_rng(seed)

        clusters = [0, 1, 2, 4, 5, 6, 7, 8] #clusters available for training, 3 and 9 are test sets
        selected_cluster = rng.choice(clusters)
        clusters.remove(selected_cluster)
        if selected_cluster in [6, 7]: #spatial overlap between 6 and 7, avoid data leakage here
            clusters.remove(6 if selected_cluster == 7 else 7)
        
        def filter_files(files, cluster):
            return [f for f in files if f.endswith(f"_{cluster}.tif")]

        absent_train = sum((filter_files(absent, c) for c in clusters), [])
        present_train = sum((filter_files(present, c) for c in clusters), [])

        absent_val = filter_files(absent, selected_cluster)
        present_val = filter_files(present, selected_cluster)

        rng.shuffle(absent_train)
        rng.shuffle(present_train)
        rng.shuffle(absent_val)
        rng.shuffle(present_val)

        if examples_per_class is not None:
            absent_train = absent_train[:examples_per_class]
            present_train = present_train[:examples_per_class]

            absent_val = absent_val[:examples_per_class]
            present_val = present_val[:examples_per_class]

        self.absent = absent_train if split == "train" else absent_val
        self.present = present_train if split == "train" else present_val

        self.all_images = self.absent + self.present
        self.all_labels = [0] * len(self.absent) + [1] * len(self.present)

        if use_randaugment: 
            train_transform = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.expand(3, *image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                      std=[0.5, 0.5, 0.5])
            ])
        else: 
            train_transform = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                      std=[0.5, 0.5, 0.5])
            ])

        val_transform = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        self.transform = {"train": train_transform, "val": val_transform}[split]

    def __len__(self):
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> torch.Tensor:        
        return Image.open(self.all_images[idx]).convert("RGB") #converting 4-band tifs to 3-channel

    def get_label_by_idx(self, idx: int) -> torch.Tensor:        
        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> Any:
        return dict(name=self.class_names[self.all_labels[idx]])
