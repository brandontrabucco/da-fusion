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
import numpy as np
import random


DATASETS = {
    "spurge": SpurgeDataset, 
    "coco": COCODataset, 
    "pascal": PASCALDataset,
    "imagenet": ImageNetDataset
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Inference script")

    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--embed-path", type=str, default="pascal-tokens/pascal-0-1.pt")
    
    parser.add_argument("--dataset", type=str, default="pascal")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--prompt", type=str, default="a photo of a {name}")
    parser.add_argument("--out", type=str, default="inference/")

    parser.add_argument("--guidance-scale", type=float, default=7.5)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
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

    for i, name in enumerate(DATASETS[args.dataset].class_names):

        initializer_ids = pipe.tokenizer.encode(
            name, add_special_tokens=False)

        fine_tuned_tokens = []

        for idx in initializer_ids:

            token = pipe.tokenizer._convert_id_to_token(idx)
            token = token.replace("</w>", "")

            fine_tuned_tokens.append(f"<{token}>")

        name = " ".join(fine_tuned_tokens)

        with autocast('cuda'):

            image = pipe(
                prompt=args.prompt.format(name=name), 
                guidance_scale=args.guidance_scale,
            ).images[0]

        image.save(os.path.join(args.out, (
            "-".join(fine_tuned_tokens) + ".png")))