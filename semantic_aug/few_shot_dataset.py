from semantic_aug.semantic_augmentation import (
    SemanticAugmentation,
    Identity
)
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


UNLABELLED_ERROR_MESSAGE = "unlabelled image datasets are not \
currently supported, please specify the 'label' key"


class FewShotDataset(Dataset):

    def __init__(self, examples_per_class: int, 
                 synthetic_aug: SemanticAugmentation = Identity(), 
                 synthetic_probability: float = 0.5,
                 *args, **kwargs):

        self.examples_per_class = examples_per_class
        self.synthetic_aug = synthetic_aug
        self.synthetic_probability = synthetic_probability

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.baked_examples = defaultdict(list)

    @abc.abstractmethod
    def filter_by_class(self, class_idx: int) -> torch.Tensor:

        return NotImplemented
    
    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> torch.Tensor:

        return NotImplemented
    
    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> Any:

        return NotImplemented

    def bake_synthetic_data(self, num_repeats):

        self.baked_examples.clear()

        options = product(range(len(self)), range(num_repeats))

        for idx, num in tqdm(list(options),desc="Baking Synthetic Data"):

            image = self.get_image_by_idx(idx)
            metadata = self.get_metadata_by_idx(idx)
            image, metadata = self.synthetic_aug(image, metadata)

            self.baked_examples[idx].append((image, metadata))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if len(self.baked_examples[idx]) > 0 and \
                np.random.uniform() < self.synthetic_probability:

            image, metadata = random.choice(self.baked_examples[idx])

        else:

            image = self.get_image_by_idx(idx)
            metadata = self.get_metadata_by_idx(idx)

        assert "label" in metadata, UNLABELLED_ERROR_MESSAGE
        return self.transform(image), metadata["label"]