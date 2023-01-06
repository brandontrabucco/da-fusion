from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torch
import os

from PIL import Image
from collections import defaultdict


PASCAL_DIR = "/projects/rsalakhugroup/datasets/pascal"

CLASS_NAMES = os.path.join(PASCAL_DIR, "class_names.txt")

TRAIN_IMAGE_SET = os.path.join(
    PASCAL_DIR, "ImageSets/Segmentation/train.txt")
VAL_IMAGE_SET = os.path.join(
    PASCAL_DIR, "ImageSets/Segmentation/val.txt")

DEFAULT_IMAGE_DIR = os.path.join(PASCAL_DIR, "JPEGImages")
DEFAULT_LABEL_DIR = os.path.join(PASCAL_DIR, "SegmentationClass")
DEFAULT_INSTANCE_DIR = os.path.join(PASCAL_DIR, "SegmentationObject")

class PASCALDataset(FewShotDataset):

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 train_image_set: str = TRAIN_IMAGE_SET, 
                 val_image_set: str = VAL_IMAGE_SET, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                 label_dir: str = DEFAULT_LABEL_DIR, 
                 instance_dir: str = DEFAULT_INSTANCE_DIR,
                 class_names: str = CLASS_NAMES, 
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5, **kwargs):

        super(PASCALDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)

        image_set = {"train": train_image_set, "val": val_image_set}[split]

        with open(class_names, "r") as f:
            self.class_names = [x.strip() for x in f.readlines()]

        self.num_classes = len(self.class_names)

        with open(image_set, "r") as f:
            image_set_lines = [x.strip() for x in f.readlines()]

        class_to_images = defaultdict(list)
        class_to_annotations = defaultdict(list)

        for image_id in image_set_lines:

            labels = os.path.join(label_dir, image_id + ".png")
            instances = os.path.join(instance_dir, image_id + ".png")

            labels = np.asarray(Image.open(labels))
            instances = np.asarray(Image.open(instances))

            instance_ids, pixel_loc, counts = np.unique(
                instances, return_index=True, return_counts=True)

            counts[0] = counts[-1] = 0  # remove background

            argmax_index = counts.argmax()

            mask = np.equal(instances, instance_ids[argmax_index])
            class_name = self.class_names[labels.flat[pixel_loc[argmax_index]]]

            class_to_images[class_name].append(
                os.path.join(image_dir, image_id + ".jpg"))
            class_to_annotations[class_name].append(dict(mask=mask))

        rng = np.random.default_rng(seed)
        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}

        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.class_to_annotations = {
            key: [class_to_annotations[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])

        self.all_annotations = sum([
            self.class_to_annotations[key] 
            for key in self.class_names], [])

        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(p=0.5),
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

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.class_names[self.all_labels[idx]], 
                    **self.all_annotations[idx])