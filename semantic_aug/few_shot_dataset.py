from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple
from torch.utils.data import Dataset
from collections import defaultdict
import torchvision.transforms as transforms
import torch
import numpy as np
import abc
import random
from itertools import product
from tqdm import tqdm


class FewShotDataset(Dataset):

    num_classes: int = None

    def __init__(self, examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 *args, **kwargs):

        self.examples_per_class = examples_per_class
        self.generative_aug = generative_aug
        self.synthetic_probability = synthetic_probability

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5]),
        ])

        self.baked_examples = defaultdict(list)
    
    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> torch.Tensor:

        return NotImplemented
    
    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> Any:

        return NotImplemented
    
    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> Any:

        return NotImplemented

    def generate_augmentations(self, num_repeats: int):

        self.baked_examples.clear()
        options = product(range(len(self)), range(num_repeats))

        for idx, num in tqdm(list(
                options), desc="Generating Augmentations"):

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

            metadata = self.get_metadata_by_idx(idx)

            self.baked_examples[idx].append(
                self.generative_aug(image, label, metadata))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if len(self.baked_examples[idx]) > 0 and \
                np.random.uniform() < self.synthetic_probability:

            image, label = random.choice(self.baked_examples[idx])

        else:

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

        return self.transform(image), label