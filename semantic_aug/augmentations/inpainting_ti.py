from semantic_aug.generative_augmentation import GenerativeAugmentation
from diffusers import StableDiffusionInpaintPipeline
from transformers import (
    CLIPFeatureExtractor, 
    CLIPTextModel, 
    CLIPTokenizer
)
from diffusers.utils import logging
from PIL import Image
from typing import Any, Tuple, Callable

from torch import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


ERROR_MESSAGE = "The tokenizer already contains the token {token}. \
Please pass a different `token` that is not already in the tokenizer."


def load_embeddings(embed_path: str,
                    model_path: str = "CompVis/stable-diffusion-v1-4"):

    tokenizer = CLIPTokenizer.from_pretrained(
        model_path, use_auth_token=True,
        subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        model_path, use_auth_token=True, 
        subfolder="text_encoder")

    for token, token_embedding in torch.load(
            embed_path, map_location="cpu").items():

        # add the token in tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        assert num_added_tokens > 0, ERROR_MESSAGE.format(token=token)
    
        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
        added_token_id = tokenizer.convert_tokens_to_ids(token)

        # get the old word embeddings
        embeddings = text_encoder.get_input_embeddings()

        # get the id for the token and assign new embeds
        embeddings.weight.data[added_token_id] = \
            token_embedding.to(embeddings.weight.dtype)

    return tokenizer, text_encoder.to('cuda')


def format_name(name):
    return f"<{name.replace(' ', '_')}>"


class TextualInversionInpainting(GenerativeAugmentation):

    def __init__(self, fine_tuned_embeddings: str, 
                 model_path: str = "CompVis/stable-diffusion-v1-4",
                 prompt: str = "a photo of a {name}",
                 strength: float = 0.5, 
                 guidance_scale: float = 7.5,
                 format_name: Callable = format_name):

        super(TextualInversionInpainting, self).__init__()

        tokenizer, text_encoder = load_embeddings(
            fine_tuned_embeddings, model_path=model_path)

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_path, use_auth_token=True,
            revision="fp16", 
            torch_dtype=torch.float16
        ).to('cuda')

        self.pipe.tokenizer = tokenizer
        self.pipe.text_encoder = text_encoder

        logging.disable_progress_bar()
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.safety_checker = None

        self.prompt = prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.format_name = format_name

    def forward(self, image: Image.Image, label: int, 
                metadata: dict) -> Tuple[Image.Image, int]:

        canvas = image.resize((512, 512), Image.BILINEAR)
        mask = Image.fromarray(metadata["mask"].astype(np.uint8)*255)
        mask = mask.resize((512,512), Image.BILINEAR)
        name = self.format_name(metadata["name"])

        with autocast('cuda'):

            canvas = self.pipe(
                image=canvas,
                prompt=[self.prompt.format(name=name)], 
                strength=self.strength, 
                guidance_scale=self.guidance_scale,
                mask_image = mask
            ).images[0]

        image = canvas.resize(image.size, Image.BILINEAR)

        return image, label