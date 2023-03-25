from semantic_aug.generative_augmentation import GenerativeAugmentation
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import logging
from PIL import Image, ImageOps

from typing import Any, Tuple, Callable
from torch import autocast
from scipy.ndimage import maximum_filter

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RealGuidance(GenerativeAugmentation):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(self, model_path: str = "CompVis/stable-diffusion-v1-4",
                 prompt: str = "a photo of a {name}",
                 strength: float = 0.5, 
                 guidance_scale: float = 7.5,
                 mask: bool = False,
                 inverted: bool = False,
                 mask_grow_radius: int = 16,
                 erasure_ckpt_path: str = None,
                 disable_safety_checker: bool = True,
                 **kwargs):

        super(RealGuidance, self).__init__()

        if RealGuidance.pipe is None:

            PipelineClass = (StableDiffusionInpaintPipeline 
                             if mask else 
                             StableDiffusionImg2ImgPipeline)

            self.pipe = PipelineClass.from_pretrained(
                model_path, use_auth_token=True,
                revision="fp16", 
                torch_dtype=torch.float16
            ).to('cuda')

            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker: 
                self.pipe.safety_checker = None

        self.prompt = prompt
        self.strength = strength
        self.guidance_scale = guidance_scale

        self.mask = mask
        self.inverted = inverted
        self.mask_grow_radius = mask_grow_radius

        self.erasure_ckpt_path = erasure_ckpt_path
        self.erasure_word_name = None

    def forward(self, image: Image.Image, label: int, 
                metadata: dict) -> Tuple[Image.Image, int]:

        canvas = image.resize((512, 512), Image.BILINEAR)
        prompt = self.prompt.format(name=metadata.get("name", ""))

        if self.mask: assert "mask" in metadata, \
            "mask=True but no mask present in metadata"
        
        word_name = metadata.get("name", "").replace(" ", "")

        if self.erasure_ckpt_path is not None and (
                self.erasure_word_name is None 
                or self.erasure_word_name != word_name):

            self.erasure_word_name = word_name
            ckpt_name = "method_full-sg_3-ng_1-iter_1000-lr_1e-05"

            ckpt_path = os.path.join(
                self.erasure_ckpt_path, 
                f"compvis-word_{word_name}-{ckpt_name}",
                f"diffusers-word_{word_name}-{ckpt_name}.pt")
    
            self.pipe.unet.load_state_dict(torch.load(
                ckpt_path, map_location='cuda'))

        kwargs = dict(
            image=canvas,
            prompt=[prompt], 
            strength=self.strength, 
            guidance_scale=self.guidance_scale
        )

        if self.mask:  # use focal object mask

            mask_image = Image.fromarray((
                np.where(metadata["mask"], 255, 0)
            ).astype(np.uint8)).resize((512, 512), Image.NEAREST)

            mask_image = Image.fromarray(
                maximum_filter(np.array(mask_image), 
                               size=self.mask_grow_radius))

            if self.inverted:

                mask_image = ImageOps.invert(
                    mask_image.convert('L')).convert('1')

            kwargs["mask_image"] = mask_image

        has_nsfw_concept = True
        while has_nsfw_concept:
            with autocast("cuda"):
                outputs = self.pipe(**kwargs)

            has_nsfw_concept = (
                self.pipe.safety_checker is not None 
                and outputs.nsfw_content_detected[0]
            )

        canvas = outputs.images[0].resize(
            image.size, Image.BILINEAR)

        return canvas, label