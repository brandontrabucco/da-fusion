from torch.utils.data import Dataset
from typing import Any, Tuple
from PIL import Image

import torch.nn as nn
import torch
import abc


class GenerativeAugmentation(nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self, image: Image.Image, label: int, 
                metadata: dict) -> Tuple[Image.Image, int]:

        return NotImplemented