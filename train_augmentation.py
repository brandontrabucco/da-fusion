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
DEFAULT_PROMPT = "a drone image of a brown field"

DEFAULT_SYNTHETIC_DIR = "/projects/rsalakhugroup/\
btrabucc/aug/{dataset}-{aug}-{seed}-{examples_per_class}"

DATASETS = {"spurge": SpurgeDataset, 
            "coco": COCODataset, 
            "pascal": PASCALDataset,
            "imagenet": ImageNetDataset}


def run_experiment(examples_per_class=0, seed=0, 
                   dataset="spurge", aug="real-guidance", 
                   num_synthetic=100, iterations_per_epoch=200, 
                   num_epochs=50, batch_size=32,
                   strength: float = 0.5, 
                   guidance_scale: float = 7.5, 
                   synthetic_probability: float = 0.5, 
                   synthetic_dir: str = DEFAULT_SYNTHETIC_DIR, 
                   model_path: str = DEFAULT_MODEL_PATH,
                   prompt: str = DEFAULT_PROMPT):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = DATASETS[dataset](
        split="train", examples_per_class=examples_per_class, 
        synthetic_probability=synthetic_probability, 
        synthetic_dir=synthetic_dir,
        generative_aug=aug, seed=seed)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path, use_auth_token=True,
        revision="fp16", 
        torch_dtype=torch.float16
    ).to('cuda')

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    for name in train_dataset.class_names:

        placeholder_token = f"<{name.replace(' ', '-')}>" 
        initializer_token = "object"

        # Add the placeholder token in tokenizer
        num_added_tokens = pipe.tokenizer.add_tokens(placeholder_token)

        token_ids = pipe.tokenizer.encode(initializer_token, add_special_tokens=False)
        
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

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

                # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
            loss.backward()

            grads = pipe.text_encoder.get_input_embeddings().weight.grad
            grads.data[:-num_added_tokens] = grads.data[:-num_added_tokens].fill_(0)

            optim.step()
            optim.zero_grad()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Few-Shot Baseline")

    parser.add_argument("--logdir", type=str, default="few_shot_combined")
    parser.add_argument("--checkpoint", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt", type=str, default="a photo of a {name}")

    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--synthetic-probability", type=float, default=0.5)
    parser.add_argument("--synthetic-dir", type=str, default=DEFAULT_SYNTHETIC_DIR)

    parser.add_argument("--iterations-per-epoch", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--num-synthetic", type=int, default=15)
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[1, 2, 4, 8, 16])
    
    parser.add_argument("--aug", type=str, default="real-guidance", 
                        choices=["real-guidance", "textual-inversion", "none"])
    
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
    os.makedirs(args.logdir, exist_ok=True)

    all_trials = []

    options = product(range(args.num_trials), args.examples_per_class)
    options = np.array(list(options))
    options = np.array_split(options, world_size)[rank]

    for seed, examples_per_class in options.tolist():

        hyperparameters = dict(
            seed=seed, examples_per_class=examples_per_class,
            dataset=args.dataset, aug=args.aug,
            num_epochs=args.num_epochs,
            iterations_per_epoch=args.iterations_per_epoch, 
            batch_size=args.batch_size,
            model_path=args.checkpoint,
            synthetic_probability=args.synthetic_probability, 
            num_synthetic=args.num_synthetic, 
            prompt=args.prompt,
            strength=args.strength, 
            guidance_scale=args.guidance_scale)

        synthetic_dir = args.synthetic_dir.format(**hyperparameters)

        all_trials.extend(run_experiment(
            synthetic_dir=synthetic_dir, **hyperparameters))

        path = f"results_{seed}_{examples_per_class}.csv"
        path = os.path.join(args.logdir, path)

        pd.DataFrame.from_records(all_trials).to_csv(path)
        print(f"[rank {rank}] n={examples_per_class} saved to: {path}")