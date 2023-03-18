from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from collections import defaultdict
import numpy as np


ILSVRC_DIR = "/projects/rsalakhugroup/datasets/imagenet"

LABEL_SYNSET = os.path.join(
    ILSVRC_DIR, "LOC_synset_mapping.txt")

TRAIN_IMAGE_SET = os.path.join(
    ILSVRC_DIR, "ILSVRC/ImageSets/CLS-LOC/train_cls.txt")
TRAIN_IMAGE_DIR = os.path.join(
    ILSVRC_DIR, "ILSVRC/Data/CLS-LOC/train")

VAL_IMAGE_SET = "/projects/rsalakhugroup/spurge/val_cls.txt"
VAL_IMAGE_DIR = os.path.join(
    ILSVRC_DIR, "ILSVRC/Data/CLS-LOC/val")


class ImageNetDataset(FewShotDataset):

    def __init__(self, *args, split: str = "train", seed: int = 0,
                 train_image_dir: str = TRAIN_IMAGE_DIR, 
                 val_image_dir: str = VAL_IMAGE_DIR, 
                 train_image_set: str = TRAIN_IMAGE_SET, 
                 val_image_set: str = VAL_IMAGE_SET, 
                 label_synset: str = LABEL_SYNSET,
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 max_classes: int = 100,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256), **kwargs):

        super(ImageNetDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)

        image_dir = {"train": train_image_dir, "val": val_image_dir}[split]
        image_set = {"train": train_image_set, "val": val_image_set}[split]

        with open(label_synset, "r") as f:
            label_synset_lines = f.readlines()

        self.class_names = []
        self.dir_to_class_names = dict()

        for synset in label_synset_lines:

            dir_name, synset = synset.split(" ", maxsplit=1)
            class_name = synset.split(",")[0].strip()

            self.class_names.append(class_name)
            self.dir_to_class_names[dir_name] = class_name

        class_to_images = defaultdict(list)

        with open(image_set, "r") as f:
            image_set_lines = f.readlines()

        for training_example in image_set_lines:

            path, idx = training_example.split(" ")
            class_name = self.dir_to_class_names[path.split("/")[0]]

            class_to_images[class_name].append(
                os.path.join(image_dir, path + ".JPEG"))

        rng = np.random.default_rng(seed)
        self.class_names = [self.class_names[i] for i in (
            rng.permutation(len(self.class_names))[:max_classes])]

        self.num_classes = len(self.class_names)
        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}

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

        if use_randaugment: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        else: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15.0),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        self.transform = {"train": train_transform, "val": val_transform}[split]

    def __len__(self):
        
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> torch.Tensor:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> torch.Tensor:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> Dict:

        return dict(name=self.class_names[self.all_labels[idx]])