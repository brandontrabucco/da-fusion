from semantic_aug.semantic_augmentation import (
    SemanticAugmentation,
    Identity
)
from typing import Any, Tuple
from torch.utils.data import Dataset
import torch
import abc


UNLABELLED_ERROR = "unlabelled image datasets are not \
currently supported, please specify the class key"


class FewShotDataset(Dataset):

    def __init__(self, examples_per_class: int, 
                 transform: SemanticAugmentation = Identity, *args, **kwargs):

        super(FewShotDataset, self).__init__(*args, **kwargs)

        self.examples_per_class = examples_per_class
        self.transform = transform

    @abc.abstractmethod
    def filter_by_class(self, class_idx: int) -> torch.Tensor:

        return NotImplemented
    
    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> torch.Tensor:

        return NotImplemented
    
    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> Any:

        return NotImplemented
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        image = self.get_image_by_idx(idx)
        metadata = self.get_metadata_by_idx(idx)

        image, metadata = self.transform(image, metadata)
        assert "class" in metadata, UNLABELLED_ERROR

        return image, metadata["class"]