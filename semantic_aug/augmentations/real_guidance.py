from semantic_aug.semantic_augmentation import SemanticAugmentation
from typing import Any, Tuple

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import logging

import torch
from PIL import Image


class RealGuidance(SemanticAugmentation):

    def __init__(self, strength: float = 0.5, 
                 guidance_scale: float = 7.5, 
                 augment_probability: float = 0.5, 
                 model_path: str = "CompVis/stable-diffusion-v1-4",
                 prompt: str = "a drone image of a brown field"):

        super(RealGuidance, self).__init__()

        logging.disable_progress_bar()

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path, use_auth_token=True,
            #revision="fp16", 
            #torch_dtype=torch.float16
        ).to('cuda')

        self.strength = strength
        self.guidance_scale = guidance_scale
        self.augment_probability = augment_probability

        self.prompt = prompt

    def forward(self, image: torch.Tensor, 
                metadata: Any) -> Tuple[torch.Tensor, Any]:

        if torch.rand(1) < self.augment_probability:

            canvas = image.resize((512, 512), Image.BILINEAR)

            canvas = self.pipe(
                image=canvas,
                prompt=[self.prompt], 
                strength=self.strength, 
                guidance_scale=self.guidance_scale
            ).images[0]

            image = canvas.resize(image.size, Image.BILINEAR)

        return image, metadata