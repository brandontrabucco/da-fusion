from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.augmentations.real_guidance import RealGuidance
from semantic_aug.augmentations.textual_inversion import TextualInversion
from semantic_aug.textual_inversion_dataset import TextualInversionDataset
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from itertools import product
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed

import argparse
import pandas as pd
import numpy as np
import random
import os
import itertools


DEFAULT_MODEL_PATH = "CompVis/stable-diffusion-v1-4"

DEFAULT_EMBED_PATH = "{dataset}-tokens/{dataset}-{seed}-{examples_per_class}.pt"

DATASETS = {"spurge": SpurgeDataset, 
            "coco": COCODataset, 
            "pascal": PASCALDataset,
            "imagenet": ImageNetDataset}


def run_experiment(dataset: str = "spurge", seed: int = 0, 
                   examples_per_class: int = 0, 
                   iterations_per_epoch: int = 200, 
                   num_epochs: int = 50, batch_size: int = 32,
                   model_path: str = DEFAULT_MODEL_PATH):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = DATASETS[dataset](
        split="train", examples_per_class=examples_per_class, seed=seed)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path, use_auth_token=True,
        revision="fp16", 
        torch_dtype=torch.float16
    ).to('cuda')

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    added_tokens = []

    for name in train_dataset.class_names:

        placeholder_token = f"<{name.replace(' ', '_')}>" 
        initializer_token = "object"

        added_tokens.append(placeholder_token)

        # Add the placeholder token in tokenizer
        num_added_tokens = pipe.tokenizer.add_tokens(placeholder_token)

        token_ids = pipe.tokenizer.encode(
            initializer_token, add_special_tokens=False)

        initializer_token_id = token_ids[0]
        placeholder_token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token)

        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

        token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    num_added_tokens = train_dataset.num_classes

    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    freeze_params(pipe.vae.parameters())
    freeze_params(pipe.unet.parameters())

    params_to_freeze = itertools.chain(
        pipe.text_encoder.text_model.encoder.parameters(),
        pipe.text_encoder.text_model.final_layer_norm.parameters(),
        pipe.text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    pipe.text_encoder.gradient_checkpointing_enable()
    pipe.unet.enable_gradient_checkpointing()

    noise_scheduler = DDPMScheduler.from_pretrained(
        model_path, subfolder="scheduler")
        
    train_dataset = TextualInversionDataset(
        train_dataset, pipe.tokenizer)

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=train_sampler, num_workers=4)

    optim = torch.optim.Adam(pipe.text_encoder.get_input_embeddings().parameters(), lr=0.0001)
    
    global_step = 0

    for epoch in trange(num_epochs):

        for step, (image, prompt) in enumerate(train_dataloader):

            # Convert images to latent space
            latents = pipe.vae.encode(image.to('cuda', dtype=torch.float16)).latent_dist.sample().detach()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = pipe.text_encoder(prompt.to('cuda'))[0]

            # Predict the noise residual
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states.to(torch.float16)).sample

            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            loss.backward()

            grads = pipe.text_encoder.get_input_embeddings().weight.grad
            grads.data[:-num_added_tokens] = grads.data[:-num_added_tokens].fill_(0)

            optim.step()
            optim.zero_grad()
        
    embeds = pipe.text_encoder.get_input_embeddings()
    embeds = embeds.weight.detach().cpu()[-num_added_tokens:]
        
    return {placeholder_token: embeds[i]
            for i, placeholder_token in enumerate(added_tokens)}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Textual Inversion Experiment")

    parser.add_argument("--checkpoint", type=str, default="CompVis/stable-diffusion-v1-4")

    parser.add_argument("--iterations-per-epoch", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[1, 2, 4, 8, 16])

    parser.add_argument("--embed-path", type=str, default=DEFAULT_EMBED_PATH)
    
    parser.add_argument("--dataset", type=str, default="coco", 
                        choices=["spurge", "imagenet", "coco", "pascal"])
    
    args = parser.parse_args()

    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        rank, world_size = 0, 1

    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    print(f'Initialized process {rank} / {world_size}')

    options = product(range(args.num_trials), args.examples_per_class)
    options = np.array(list(options))
    options = np.array_split(options, world_size)[rank]

    for seed, examples_per_class in options.tolist():

        hyperparameters = dict(
            dataset=args.dataset, seed=seed, 
            examples_per_class=examples_per_class,
            num_epochs=args.num_epochs,
            iterations_per_epoch=args.iterations_per_epoch, 
            batch_size=args.batch_size,
            model_path=args.checkpoint)

        path = args.embed_path.format(**hyperparameters)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        fine_tuned_dict = run_experiment(**hyperparameters)
        torch.save(fine_tuned_dict, path)

        print(f"[rank {rank}] n={examples_per_class} saved to: {path}")