from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.augmentations.compose import ComposeParallel
from semantic_aug.augmentations.compose import ComposeSequential
from semantic_aug.augmentations.real_guidance import RealGuidance
from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline
from itertools import product
from torch import autocast
from PIL import Image

from tqdm import tqdm
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

COMPOSE = {
    "parallel": ComposeParallel,
    "sequential": ComposeSequential
}

AUGMENT = {
    "real-guidance": RealGuidance,
    "textual-inversion": TextualInversion
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Inference script")
    
    parser.add_argument("--out", type=str, default="real-guidance/")

    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--embed-path", type=str, default="erasure-tokens/pascal-tokens/pascal-0-8.pt")
    
    parser.add_argument("--dataset", type=str, default="pascal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-per-class", type=int, default=1)
    parser.add_argument("--num-synthetic", type=int, default=10)

    parser.add_argument("--prompt", type=str, default="a photo of a {name}")
    
    parser.add_argument("--aug", nargs="+", type=str, default=["real-guidance"], 
                        choices=["real-guidance", "textual-inversion"])

    parser.add_argument("--guidance-scale", nargs="+", type=float, default=[7.5])
    parser.add_argument("--strength", nargs="+", type=float, default=[0.5])

    parser.add_argument("--mask", nargs="+", type=int, default=[0], choices=[0, 1])
    parser.add_argument("--inverted", nargs="+", type=int, default=[0], choices=[0, 1])
    
    parser.add_argument("--probs", nargs="+", type=float, default=None)
    
    parser.add_argument("--compose", type=str, default="parallel", 
                        choices=["parallel", "sequential"])

    parser.add_argument("--class-name", type=str, default=None)
    
    parser.add_argument("--erasure-ckpt-path", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    aug = COMPOSE[args.compose]([
        
        AUGMENT[aug](
            embed_path=args.embed_path, 
            model_path=args.model_path, 
            prompt=args.prompt, 
            strength=strength, 
            guidance_scale=guidance_scale,
            mask=mask, 
            inverted=inverted,
            erasure_ckpt_path=args.erasure_ckpt_path
        )

        for (aug, guidance_scale, 
             strength, mask, inverted) in zip(
            args.aug, args.guidance_scale, 
            args.strength, args.mask, args.inverted
        )

    ], probs=args.probs)

    train_dataset = DATASETS[
        args.dataset](split="train", seed=args.seed, 
                      examples_per_class=args.examples_per_class)

    options = product(range(len(train_dataset)), range(args.num_synthetic))

    for idx, num in tqdm(list(
            options), desc="Generating Augmentations"):

        image = train_dataset.get_image_by_idx(idx)
        label = train_dataset.get_label_by_idx(idx)

        metadata = train_dataset.get_metadata_by_idx(idx)

        if args.class_name is not None: 
            if metadata["name"] != args.class_name: continue

        image, label = aug(
            image, label, metadata)

        name = metadata['name'].replace(" ", "_")

        pil_image, image = image, os.path.join(
            args.out, f"{name}-{idx}-{num}.png")

        pil_image.save(image)