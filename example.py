from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.augmentations.real_guidance import RealGuidance
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from itertools import product
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed

import pandas as pd
import numpy as np
import random
import os


def run_experiment(model, examples_per_class, 
                   seed=0, iterations_per_epoch=100, 
                   num_epochs=100, batch_size=32,
                   strength: float = 0.5, 
                   guidance_scale: float = 7.5, 
                   augment_probability: float = 0.5, 
                   model_path: str = "CompVis/stable-diffusion-v1-4",
                   prompt: str = "a drone image of a brown field"):

    aug = RealGuidance(
        strength=strength,
        guidance_scale=guidance_scale,
        augment_probability=augment_probability,
        model_path=model_path,
        prompt=prompt,
    ).cuda()

    train_dataset = SpurgeDataset(
        "train", examples_per_class, aug=aug, seed=seed)

    val_dataset = SpurgeDataset(
        "val", examples_per_class, aug=aug, seed=seed)

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    val_sampler = torch.utils.data.RandomSampler(
        val_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=train_sampler, num_workers=4)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, 
        sampler=val_sampler, num_workers=4)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    records = []

    for epoch in range(num_epochs):

        model.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for image, label in train_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, label).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.detach().cpu().numpy() * logits.shape[0]
            epoch_accuracy += (prediction == label).float().sum().cpu().numpy()

        training_loss = epoch_loss / len(train_dataloader.dataset)
        training_accuracy = epoch_accuracy / len(train_dataloader.dataset)

        model.eval()

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for image, label in val_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, label).mean()

            epoch_loss += loss.detach().cpu().numpy() * logits.shape[0]
            epoch_accuracy += (prediction == label).float().sum().cpu().numpy()

        validation_loss = epoch_loss / len(val_dataloader.dataset)
        validation_accuracy = epoch_accuracy / len(val_dataloader.dataset)

        records.append(dict(
            epoch=epoch, 
            value=training_loss, 
            metric="Loss", 
            split="Training"
        ))

        records.append(dict(
            epoch=epoch, 
            value=validation_loss, 
            metric="Loss", 
            split="Validation"
        ))

        records.append(dict(
            epoch=epoch, 
            value=training_accuracy, 
            metric="Accuracy", 
            split="Training"
        ))

        records.append(dict(
            epoch=epoch, 
            value=validation_accuracy, 
            metric="Accuracy", 
            split="Validation"
        ))
            
    return records


class ClassificationModel(nn.Module):
    
    def __init__(self):
        
        super(ClassificationModel, self).__init__()
        
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.out = nn.Linear(2048, 3)
        
    def forward(self, image):
        
        x = image
        
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
        
        return self.out(x)


if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser("Few-Shot Baseline")

    parser.add_argument("--logdir", type=str, default="few_shot_combined")
    parser.add_argument("--checkpoint", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt", type=str, default="a woodland seen from a drone")
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--augment-probability", type=float, default=0.5)
    parser.add_argument("--iterations-per-epoch", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=10)

    args = parser.parse_args()

    try:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        rank, world_size = 0, 1
    else:
        distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    device_id = rank % torch.cuda.device_count()
    print(f'Initialized process {rank} / {world_size}')

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    os.makedirs(args.logdir, exist_ok=True)

    all_trials = []

    options = list(product([1, 5, 10, 15, 20, 25], range(5)))
    options = np.array_split(np.array(options), world_size)[rank].tolist()

    for examples_per_class, seed in options:

        model = ClassificationModel().cuda()

        records = run_experiment(
            model, examples_per_class, seed=seed, 
            iterations_per_epoch=args.iterations_per_epoch,
            num_epochs=args.num_epochs,
            strength=args.strength, 
            guidance_scale=args.guidance_scale, 
            augment_probability=args.augment_probability, 
            model_path=args.checkpoint,
            prompt=args.prompt
        )

        all_trials.extend([dict(
            **x, examples_per_class=examples_per_class,
        ) for x in records])

        print(f"[rank {rank}] n={examples_per_class} finished")

        all_trials_final = [x for x in all_trials if x["epoch"] > 75]
        
        df = pd.DataFrame.from_records(all_trials_final)
        df.to_csv(os.path.join(args.logdir, f"results-{rank}.csv"))