from semantic_aug.generative_augmentation import GenerativeAugmentation
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import (
    CLIPFeatureExtractor, 
    CLIPTextModel, 
    CLIPTokenizer
)
from diffusers.utils import logging
from PIL import Image
from typing import Any, Tuple

from torch import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_embeddings(embed_path: str,
                    model_path: str = "CompVis/stable-diffusion-v1-4"):

    tokenizer = CLIPTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        use_auth_token=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_path, 
        subfolder="text_encoder", 
        use_auth_token=True
    )

    loaded_learned_embeds = torch.load(embed_path, map_location="cpu")
  
    # separate token and the embeds
    token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {token}. \
Please pass a different `token` that is not already in the tokenizer.")
  
    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
  
    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    return tokenizer, text_encoder.to('cuda')


class TextualInversion(GenerativeAugmentation):

    def __init__(self, *fine_tuned_embeddings: str, 
                 model_path: str = "CompVis/stable-diffusion-v1-4",
                 prompt: str = "a photo of a {name}",
                 strength: float = 0.5, 
                 guidance_scale: float = 7.5):

        super(TextualInversion, self).__init__()

        self.embeddings = [
            load_embeddings(embed_path, model_path=model_path)
            for embed_path in fine_tuned_embeddings
        ]

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path, use_auth_token=True,
            revision="fp16", 
            torch_dtype=torch.float16
        ).to('cuda')

        logging.disable_progress_bar()
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.safety_checker = None

        self.prompt = prompt
        self.strength = strength
        self.guidance_scale = guidance_scale

    def forward(self, image: Image.Image, label: int, 
                metadata: dict) -> Tuple[Image.Image, int]:

        canvas = image.resize((512, 512), Image.BILINEAR)

        tokenizer, text_encoder = self.embeddings[label]
        self.pipe.tokenizer = tokenizer
        self.pipe.text_encoder = text_encoder

        with autocast('cuda'):

            canvas = self.pipe(
                image=canvas,
                prompt=[self.prompt.format(name=metadata.get("name", ""))], 
                strength=self.strength, 
                guidance_scale=self.guidance_scale
            ).images[0]

        image = canvas.resize(image.size, Image.BILINEAR)

        return image, label