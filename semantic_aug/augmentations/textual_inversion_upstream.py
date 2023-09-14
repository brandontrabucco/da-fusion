from semantic_aug.generative_augmentation import GenerativeAugmentation
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from transformers import (
    CLIPFeatureExtractor, 
    CLIPTextModel, 
    CLIPTokenizer
)
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

from glob import glob

ERROR_MESSAGE = "Tokenizer already contains the token {token}. \
Please pass a different `token` that is not already in the tokenizer."

def format_name(name, num_tokens: int = 1):

    special_token = f"<{name.replace(' ', '_')}>"

    return " ".join([
        special_token
        if token_idx == 0 else
        f"{special_token}_{token_idx}"
        for token_idx in range(num_tokens)
    ])

def center_crop(img, new_width, new_height):
    width, height = img.size

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    return img.crop((left, top, right, bottom))

def create_mosaic(img, in_sz=39, out_sz=512):
    # Calculate the number of copies needed to exceed 512x512 pixels
    copies_x = (out_sz // in_sz) + 1
    copies_y = (out_sz // in_sz) + 1

    # Create a new blank image with dimensions slightly larger than 512x512
    mosaic_width = copies_x * in_sz
    mosaic_height = copies_y * in_sz
    mosaic = Image.new('RGB', (mosaic_width, mosaic_height))

    # Loop over a grid and paste the image into the correct positions
    for x in range(copies_x):
        for y in range(copies_y):
            mosaic.paste(img, (x * in_sz, y * in_sz))

    # Crop the image to out_sz x out_sz
    mosaic = mosaic.crop((0, 0, out_sz, out_sz))

    return mosaic

class TextualInversion(GenerativeAugmentation):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(self, embed_path: str, 
                 model_path: str = "CompVis/stable-diffusion-v1-4",
                 prompt: str = "a photo of a {name}",
                 format_name: Callable = format_name,
                 strength: float = 0.5, 
                 guidance_scale: float = 7.5,
                 mask: bool = False,
                 inverted: bool = False,
                 mask_grow_radius: int = 16,
                 erasure_ckpt_path: str = None,
                 disable_safety_checker: bool = True,
                 tokens_per_class: int = 1,
                 **kwargs):

        super(TextualInversion, self).__init__()

        if TextualInversion.pipe is None:

            PipelineClass = (StableDiffusionInpaintPipeline 
                             if mask else 
                             StableDiffusionImg2ImgPipeline)

            TextualInversion.pipe = PipelineClass.from_pretrained(
                model_path, use_auth_token=True,
                revision="fp16", 
                torch_dtype=torch.float16
            ).to('cuda')
            
            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker:
                self.pipe.safety_checker = None
        
            embeds_list = glob(embed_path + '/**/learned_embeds.bin')
            
            for e in embeds_list:
                self.pipe.load_textual_inversion(e)
        
        self.prompt = prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.format_name = format_name
        self.tokens_per_class = tokens_per_class

        self.mask = mask
        self.inverted = inverted
        self.mask_grow_radius = mask_grow_radius

        self.erasure_ckpt_path = erasure_ckpt_path
        self.erasure_word_name = None

    def forward(self, image: Image.Image, label: int, 
                metadata: dict) -> Tuple[Image.Image, int]:

        canvas = create_mosaic(image)
        name = self.format_name(
            metadata.get("name", ""),
            num_tokens=self.tokens_per_class)
        prompt = self.prompt.format(name=name)

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
        
        mosaic_crop = center_crop(outputs.images[0], 39, 39)

        canvas = mosaic_crop.resize(
            image.size, Image.BILINEAR)

        return canvas, label
