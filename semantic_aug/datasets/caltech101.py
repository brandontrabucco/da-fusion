from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
import glob
import os

from PIL import Image
from collections import defaultdict


DEFAULT_IMAGE_DIR = "/projects/rsalakhugroup/datasets/caltech101/caltech101/101_ObjectCategories"


class CalTech101Dataset(FewShotDataset):

    class_names = ['accordion', 'airplanes', 'anchor', 'ant', 
        'background google', 'barrel', 'bass', 'beaver', 'binocular', 
        'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 
        'cannon', 'car side', 'ceiling fan', 'cellphone', 'chair', 
        'chandelier', 'cougar body', 'cougar face', 'crab', 'crayfish', 
        'crocodile', 'crocodile head', 'cup', 'dalmatian', 'dollar bill', 
        'dolphin', 'dragonfly', 'electric guitar', 'elephant', 'emu', 
        'euphonium', 'ewer', 'faces', 'faces easy', 'ferry', 'flamingo', 
        'flamingo head', 'garfield', 'gerenuk', 'gramophone', 'grand piano', 
        'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 
        'inline skate', 'joshua tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 
        'leopards', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 
        'menorah', 'metronome', 'minaret', 'motorbikes', 'nautilus', 
        'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 
        'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 
        'scissors', 'scorpion', 'sea horse', 'snoopy', 'soccer ball', 
        'stapler', 'starfish', 'stegosaurus', 'stop sign', 'strawberry', 
        'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water lilly', 
        'wheelchair', 'wild cat', 'windsor chair', 'wrench', 'yin yang']

    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256), **kwargs):

        super(CalTech101Dataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)

        class_to_images = defaultdict(list)

        for image_path in glob.glob(os.path.join(image_dir, "*/*.jpg")):
            class_name = image_path.split("/")[-2].lower().replace("_", " ")
            class_to_images[class_name].append(image_path)

        rng = np.random.default_rng(seed)

        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}
        
        class_to_ids = {key: np.array_split(class_to_ids[key], 2)[0 if split == "train" else 1] for key in self.class_names}

        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])

        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        if use_randaugment: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        else: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15.0),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        self.transform = {"train": train_transform, "val": val_transform}[split]

    def __len__(self):
        
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.class_names[self.all_labels[idx]])