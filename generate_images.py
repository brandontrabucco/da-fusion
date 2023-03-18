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


DEFAULT_ERASURE_CKPT = (
    "/projects/rsalakhugroup/btrabucc/esd-models/" + 
    "compvis-word_airplane-method_full-sg_3-ng_1-iter_1000-lr_1e-05/" + 
    "diffusers-word_airplane-method_full-sg_3-ng_1-iter_1000-lr_1e-05.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Stable Diffusion inference script")

    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--embed-path", type=str, default=(
        "erasure-tokens/fine-tuned/pascal-0-8/airplane/learned_embeds.bin"))
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-generate", type=int, default=10)

    parser.add_argument("--prompt", type=str, default="a photo of a <airplane>")
    parser.add_argument("--out", type=str, default="erasure-tokens/fine-tuned/pascal-0-8/airplane/")

    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--erasure-ckpt-name", type=str, default=DEFAULT_ERASURE_CKPT)

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

    if args.erasure_ckpt_name is not None:
        pipe.unet.load_state_dict(torch.load(
            args.erasure_ckpt_name, map_location='cuda'))

    for idx in trange(args.num_generate, 
                      desc="Generating Images"):

        with autocast('cuda'):

            image = pipe(
                args.prompt, 
                guidance_scale=args.guidance_scale
            ).images[0]

        image.save(os.path.join(args.out, f"{idx}.png"))