from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline
from itertools import product
from torch import autocast
from PIL import Image

from tqdm import trange
import os
import torch
import argparse
import numpy as np
import random


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Stable Diffusion inference script")

    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--embed-path", type=str, default=(
        "fine-tuned/pascal-0-1/airplane/learned_embeds-steps-2000.bin"))
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-generate", type=int, default=10)

    parser.add_argument("--prompt", type=str, default="a photo of a <airplane>")
    parser.add_argument("--out", type=str, default="fine-tuned/pascal-0-1/airplane/")

    parser.add_argument("--guidance-scale", type=float, default=7.5)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path, use_auth_token=True,
        revision="fp16", 
        torch_dtype=torch.float16
    ).to('cuda')

    aug = TextualInversion(args.embed_path, model_path=args.model_path)
    pipe.tokenizer = aug.pipe.tokenizer
    pipe.text_encoder = aug.pipe.text_encoder

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    for idx in trange(args.num_generate, 
                      desc="Generating Images"):

        with autocast('cuda'):

            image = pipe(
                args.prompt, 
                guidance_scale=args.guidance_scale
            ).images[0]

        image.save(os.path.join(args.out, f"{idx}.png"))