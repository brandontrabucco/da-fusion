from torch.utils.data import Dataset
from typing import Any, Tuple
import torch.nn as nn
import torch
import abc


class GenerativeAugmentation(nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self, image: torch.Tensor, 
                label: torch.Tensor, 
                metadata: Any) -> Tuple[torch.Tensor, torch.Tensor]:

        return NotImplemented


class Identity(GenerativeAugmentation):

    def forward(self, image: torch.Tensor, 
                label: torch.Tensor, 
                metadata: Any) -> Tuple[torch.Tensor, torch.Tensor]:
                
        return image, label