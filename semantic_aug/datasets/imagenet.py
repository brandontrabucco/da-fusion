from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from collections import defaultdict
import numpy as np


ILSVRC_DIR = "/projects/rsalakhugroup/datasets/imagenet/ILSVRC"

DEFAULT_IMAGE_SET = os.path.join(
    ILSVRC_DIR, "ImageSets/CLS-LOC/train_cls.txt")
DEFAULT_IMAGE_DIR = os.path.join(
    ILSVRC_DIR, "Data/CLS-LOC/train")


class ImageNetDataset(FewShotDataset):

    num_classes: int = 1000

    def __init__(self, *args, image_dir: str = DEFAULT_IMAGE_DIR, 
                 image_set: str = DEFAULT_IMAGE_SET, 
                 split: str = "train", seed: int = 0, 
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 val_fraction: float = 0.1, **kwargs):

        super(ImageNetDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)

        self.class_names = []
        class_to_images = defaultdict(list)

        with open(image_set, "r") as f:
            image_set_lines = f.readlines()

        for training_example in image_set_lines:

            path, idx = training_example.split(" ")
            class_name = path.split("/")[0]

            if class_name not in self.class_names:
                self.class_names.append(class_name)

            class_to_images[class_name].append(
                os.path.join(image_dir, path + ".JPEG"))

        rng = np.random.default_rng(seed)
        class_to_permutation = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}

        class_to_ids = {key: (
            {split_key: split_ids for split_key, split_ids in zip(
                ["train", "val"], np.array_split(
                    ids, [int(ids.size * (1 - val_fraction))]))}[split]
        ) for key, ids in class_to_permutation.items()}

        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.all_images = sum([self.class_to_images[key] 
                               for key in self.class_names], [])

        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, 256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, 256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
        ])

        self.transform = {"train": train_transform, "val": val_transform}[split]

    def __len__(self):
        
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> torch.Tensor:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> torch.Tensor:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> Dict:

        return dict(class_name=self.class_names[self.all_labels[idx]])