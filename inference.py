from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline
from torch import autocast
from PIL import Image


import os
import torch
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Inference script")

    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--embed-path", type=str, default="coco-step=500-tokens/coco-0-1.pt")

    parser.add_argument("--prompt", type=str, default="a photo of a <tennis_racket>")
    parser.add_argument("--out", type=str, default="inference_test.png")

    parser.add_argument("--guidance-scale", type=float, default=7.5)

    args = parser.parse_args()
    
    aug = TextualInversion(
        args.embed_path, prompt=args.prompt, 
        model_path=args.model_path, 
        guidance_scale=args.guidance_scale)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path, 
        use_auth_token=True,
        revision="fp16", 
        torch_dtype=torch.float16
    ).to('cuda')

    pipe.tokenizer = aug.pipe.tokenizer
    pipe.text_encoder = aug.pipe.text_encoder

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    with autocast('cuda'):

        image = pipe(
            prompt=args.prompt, 
            guidance_scale=args.guidance_scale,
        ).images[0]

    image.save(args.out)