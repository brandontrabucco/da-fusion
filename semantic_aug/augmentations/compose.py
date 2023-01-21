from semantic_aug.generative_augmentation import GenerativeAugmentation
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import logging
from PIL import Image

from typing import List, Union, Any, Tuple
from torch import autocast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComposeSequential(GenerativeAugmentation):

    def __init__(self, augs: List[GenerativeAugmentation], 
                 probs: List[float] = None):

        super(ComposeSequential, self).__init__()

        self.augs = augs
        self.probs = probs if probs is not None \
            else [1.0 for _ in augs]

    def forward(self, image: Image.Image, label: int, 
                metadata: dict) -> Tuple[Image.Image, int]:

        for aug, p in zip(self.augs, self.probs):

            if np.random.uniform() < p:
                image, label = aug(image, label, metadata)

        return image, label


class ComposeParallel(GenerativeAugmentation):

    def __init__(self, augs: List[GenerativeAugmentation], 
                 probs: List[float] = None):

        super(ComposeParallel, self).__init__()

        self.augs = augs
        self.probs = probs if probs is not None \
            else [1.0 / len(augs) for _ in augs]

    def forward(self, image: Image.Image, label: int, 
                metadata: dict) -> Tuple[Image.Image, int]:

        idx = np.random.choice(len(self.probs), p=self.probs)

        image, label = self.augs[idx](image, label, metadata)

        return image, label