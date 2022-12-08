from torch.utils.data import Dataset
from typing import Any, Tuple
import torch.nn as nn
import torch
import abc


class SemanticAugmentation(nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self, image: torch.Tensor, 
                metadata: Any) -> Tuple[torch.Tensor, Any]:

        return NotImplemented


class Identity(SemanticAugmentation):

    def forward(self, image: torch.Tensor, 
                metadata: Any) -> Tuple[torch.Tensor, Any]:
                
        return image, metadata


class Compose(SemanticAugmentation):
        
    def __init__(self, *augmentations: SemanticAugmentation):

        super(Compose, self).__init__()

        self.augmentations

    def forward(self, image: torch.Tensor, 
                metadata: Any) -> Tuple[torch.Tensor, Any]:

        for aug in self.augmentations:
            image, metadata = aug(image, metadata)
                
        return image, metadata