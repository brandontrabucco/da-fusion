from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.datasets.caltech101 import CalTech101Dataset
from semantic_aug.datasets.flowers102 import Flowers102Dataset
from semantic_aug.augmentations.compose import ComposeParallel
from semantic_aug.augmentations.compose import ComposeSequential
from semantic_aug.augmentations.real_guidance import RealGuidance
from semantic_aug.augmentations.textual_inversion import TextualInversion
from semantic_aug.augmentations.textual_inversion_upstream \
    import TextualInversion as MultiTokenTextualInversion
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoImageProcessor, DeiTModel
from itertools import product
from tqdm import trange
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed

import argparse
import pandas as pd
import numpy as np
import random
import os

try: 
    from cutmix.cutmix import CutMix
    IS_CUTMIX_INSTALLED = True
except:
    IS_CUTMIX_INSTALLED = False


DEFAULT_MODEL_PATH = "CompVis/stable-diffusion-v1-4"
DEFAULT_PROMPT = "a photo of a {name}"

DEFAULT_SYNTHETIC_DIR = "/projects/rsalakhugroup/\
btrabucc/aug/{dataset}-{aug}-{seed}-{examples_per_class}"

DEFAULT_EMBED_PATH = "{dataset}-tokens/{dataset}-{seed}-{examples_per_class}.pt"

DATASETS = {
    "spurge": SpurgeDataset, 
    "coco": COCODataset, 
    "pascal": PASCALDataset,
    "imagenet": ImageNetDataset,
    "caltech": CalTech101Dataset,
    "flowers": Flowers102Dataset
}

COMPOSERS = {
    "parallel": ComposeParallel,
    "sequential": ComposeSequential
}

AUGMENTATIONS = {
    "real-guidance": RealGuidance,
    "textual-inversion": TextualInversion,
    "multi-token-inversion": MultiTokenTextualInversion
}


def run_experiment(examples_per_class: int = 0, 
                   seed: int = 0, 
                   dataset: str = "spurge", 
                   num_synthetic: int = 100, 
                   iterations_per_epoch: int = 200, 
                   num_epochs: int = 50, 
                   batch_size: int = 32, 
                   aug: List[str] = None,
                   strength: List[float] = None, 
                   guidance_scale: List[float] = None,
                   mask: List[bool] = None,
                   inverted: List[bool] = None, 
                   probs: List[float] = None,
                   compose: str = "parallel",
                   synthetic_probability: float = 0.5, 
                   synthetic_dir: str = DEFAULT_SYNTHETIC_DIR, 
                   embed_path: str = DEFAULT_EMBED_PATH,
                   model_path: str = DEFAULT_MODEL_PATH,
                   prompt: str = DEFAULT_PROMPT,
                   tokens_per_class: int = 4,
                   use_randaugment: bool = False,
                   use_cutmix: bool = False,
                   erasure_ckpt_path: str = None,
                   image_size: int = 256,
                   classifier_backbone: str = "resnet50"):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if aug is not None:

        aug = COMPOSERS[compose]([
            
            AUGMENTATIONS[aug](
                embed_path=embed_path, 
                model_path=model_path, 
                prompt=prompt, 
                strength=strength, 
                guidance_scale=guidance_scale,
                mask=mask, 
                inverted=inverted,
                erasure_ckpt_path=erasure_ckpt_path,
                tokens_per_class=tokens_per_class
            )

            for (aug, guidance_scale, 
                 strength, mask, inverted) in zip(
                aug, guidance_scale, 
                strength, mask, inverted
            )

        ], probs=probs)

    train_dataset = DATASETS[dataset](
        split="train", examples_per_class=examples_per_class, 
        synthetic_probability=synthetic_probability, 
        synthetic_dir=synthetic_dir,
        use_randaugment=use_randaugment,
        generative_aug=aug, seed=seed,
        image_size=(image_size, image_size))

    if num_synthetic > 0 and aug is not None:
        train_dataset.generate_augmentations(num_synthetic)

    cutmix_dataset = None
    if use_cutmix and IS_CUTMIX_INSTALLED:
        cutmix_dataset = CutMix(
            train_dataset, beta=1.0, prob=0.5, num_mix=2, 
            num_class=train_dataset.num_classes)

    train_sampler = torch.utils.data.RandomSampler(
        cutmix_dataset if cutmix_dataset is not None else 
        train_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(
        cutmix_dataset if cutmix_dataset is not None else 
        train_dataset, batch_size=batch_size, 
        sampler=train_sampler, num_workers=4)

    val_dataset = DATASETS[dataset](
        split="val", seed=seed,
        image_size=(image_size, image_size))

    val_sampler = torch.utils.data.RandomSampler(
        val_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, 
        sampler=val_sampler, num_workers=4)

    model = ClassificationModel(
        train_dataset.num_classes, 
        backbone=classifier_backbone
    ).cuda()

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    records = []

    for epoch in trange(num_epochs, desc="Training Classifier"):

        model.train()

        epoch_loss = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')
        epoch_accuracy = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')
        epoch_size = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')

        for image, label in train_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            if len(label.shape) > 1: label = label.argmax(dim=1)

            accuracy = (prediction == label).float()

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

            with torch.no_grad():
            
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        training_loss = epoch_loss / epoch_size.clamp(min=1)
        training_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        training_loss = training_loss.cpu().numpy()
        training_accuracy = training_accuracy.cpu().numpy()

        model.eval()

        epoch_loss = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')
        epoch_accuracy = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')
        epoch_size = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')

        for image, label in val_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            accuracy = (prediction == label).float()

            with torch.no_grad():
            
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        validation_loss = epoch_loss / epoch_size.clamp(min=1)
        validation_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        validation_loss = validation_loss.cpu().numpy()
        validation_accuracy = validation_accuracy.cpu().numpy()

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=training_loss.mean(), 
            metric="Loss", 
            split="Training"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=validation_loss.mean(), 
            metric="Loss", 
            split="Validation"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=training_accuracy.mean(), 
            metric="Accuracy", 
            split="Training"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=validation_accuracy.mean(), 
            metric="Accuracy", 
            split="Validation"
        ))

        for i, name in enumerate(train_dataset.class_names):

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=training_loss[i], 
                metric=f"Loss {name.title()}", 
                split="Training"
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=validation_loss[i], 
                metric=f"Loss {name.title()}", 
                split="Validation"
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=training_accuracy[i], 
                metric=f"Accuracy {name.title()}", 
                split="Training"
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=validation_accuracy[i], 
                metric=f"Accuracy {name.title()}", 
                split="Validation"
            ))
            
    return records


class ClassificationModel(nn.Module):
    
    def __init__(self, num_classes: int, backbone: str = "resnet50"):
        
        super(ClassificationModel, self).__init__()

        self.backbone = backbone
        self.image_processor  = None

        if backbone == "resnet50":
        
            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.out = nn.Linear(2048, num_classes)

        elif backbone == "deit":

            self.base_model = DeiTModel.from_pretrained(
                "facebook/deit-base-distilled-patch16-224")
            self.out = nn.Linear(768, num_classes)
        
    def forward(self, image):
        
        x = image

        if self.backbone == "resnet50":
            
            with torch.no_grad():

                x = self.base_model.conv1(x)
                x = self.base_model.bn1(x)
                x = self.base_model.relu(x)
                x = self.base_model.maxpool(x)

                x = self.base_model.layer1(x)
                x = self.base_model.layer2(x)
                x = self.base_model.layer3(x)
                x = self.base_model.layer4(x)

                x = self.base_model.avgpool(x)
                x = torch.flatten(x, 1)

        elif self.backbone == "deit":
            
            with torch.no_grad():

                x = self.base_model(x)[0][:, 0, :]
            
        return self.out(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Few-Shot Baseline")

    parser.add_argument("--logdir", type=str, default="few_shot_combined")
    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")

    parser.add_argument("--prompt", type=str, default="a photo of a {name}")

    parser.add_argument("--synthetic-probability", type=float, default=0.5)
    parser.add_argument("--synthetic-dir", type=str, default=DEFAULT_SYNTHETIC_DIR)
    
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--classifier-backbone", type=str, 
                        default="resnet50", choices=["resnet50", "deit"])

    parser.add_argument("--iterations-per-epoch", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--num-synthetic", type=int, default=15)
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[1, 2, 4, 8, 16])
    
    parser.add_argument("--embed-path", type=str, default=DEFAULT_EMBED_PATH)
    
    parser.add_argument("--dataset", type=str, default="pascal", 
                        choices=["spurge", "imagenet", "coco", "pascal", "flowers", "caltech"])
    
    parser.add_argument("--aug", nargs="+", type=str, default=None, 
                        choices=["real-guidance", "textual-inversion",
                                 "multi-token-inversion"])

    parser.add_argument("--strength", nargs="+", type=float, default=None)
    parser.add_argument("--guidance-scale", nargs="+", type=float, default=None)

    parser.add_argument("--mask", nargs="+", type=int, default=None, choices=[0, 1])
    parser.add_argument("--inverted", nargs="+", type=int, default=None, choices=[0, 1])
    
    parser.add_argument("--probs", nargs="+", type=float, default=None)
    
    parser.add_argument("--compose", type=str, default="parallel", 
                        choices=["parallel", "sequential"])
    
    parser.add_argument("--erasure-ckpt-path", type=str, default=None)

    parser.add_argument("--use-randaugment", action="store_true")
    parser.add_argument("--use-cutmix", action="store_true")

    parser.add_argument("--tokens-per-class", type=int, default=4)
    
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
            examples_per_class=examples_per_class,
            seed=seed, 
            dataset=args.dataset,
            num_epochs=args.num_epochs,
            iterations_per_epoch=args.iterations_per_epoch, 
            batch_size=args.batch_size,
            model_path=args.model_path,
            synthetic_probability=args.synthetic_probability, 
            num_synthetic=args.num_synthetic, 
            prompt=args.prompt, 
            tokens_per_class=args.tokens_per_class,
            aug=args.aug,
            strength=args.strength, 
            guidance_scale=args.guidance_scale,
            mask=args.mask, 
            inverted=args.inverted,
            probs=args.probs,
            compose=args.compose,
            use_randaugment=args.use_randaugment,
            use_cutmix=args.use_cutmix,
            erasure_ckpt_path=args.erasure_ckpt_path,
            image_size=args.image_size,
            classifier_backbone=args.classifier_backbone)

        synthetic_dir = args.synthetic_dir.format(**hyperparameters)
        embed_path = args.embed_path.format(**hyperparameters)

        all_trials.extend(run_experiment(
            synthetic_dir=synthetic_dir, 
            embed_path=embed_path, **hyperparameters))

        path = f"results_{seed}_{examples_per_class}.csv"
        path = os.path.join(args.logdir, path)

        pd.DataFrame.from_records(all_trials).to_csv(path)
        print(f"[rank {rank}] n={examples_per_class} saved to: {path}")