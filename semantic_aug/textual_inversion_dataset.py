from transformers import CLIPTokenizer
from semantic_aug.few_shot_dataset import FewShotDataset
from torch.utils.data import Dataset
from typing import Tuple, List, Callable
import torchvision.transforms.functional as F

import torch
import random


DEFAULT_PROMPT_TEMPLATES = [
    "a photo of a {name}",
    "a rendering of a {name}",
    "a cropped photo of the {name}",
    "the photo of a {name}",
    "a photo of a clean {name}",
    "a photo of a dirty {name}",
    "a dark photo of the {name}",
    "a photo of my {name}",
    "a photo of the cool {name}",
    "a close-up photo of a {name}",
    "a bright photo of the {name}",
    "a cropped photo of a {name}",
    "a photo of the {name}",
    "a good photo of the {name}",
    "a photo of one {name}",
    "a close-up photo of the {name}",
    "a rendition of the {name}",
    "a photo of the clean {name}",
    "a rendition of a {name}",
    "a photo of a nice {name}",
    "a good photo of a {name}",
    "a photo of the nice {name}",
    "a photo of the small {name}",
    "a photo of the weird {name}",
    "a photo of the large {name}",
    "a photo of a cool {name}",
    "a photo of a small {name}"
]


def format_name(name):
    return f"<{name.replace(' ', '_')}>"


class TextualInversionDataset(Dataset):

    def __init__(self, dataset: FewShotDataset, 
                 tokenizer: CLIPTokenizer, 
                 format_name: Callable = format_name,
                 prompt_templates: List[str] = DEFAULT_PROMPT_TEMPLATES,
                 image_size: Tuple[int] = (512, 512)):

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.format_name = format_name
        self.prompt_templates = prompt_templates
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image, label = self.dataset[idx]
        metadata = self.dataset.get_metadata_by_idx(idx)

        metadata["name"] = self.format_name(metadata["name"])
        prompt = random.choice(self.prompt_templates).format(**metadata)

        image = F.resize(image, self.image_size)
        
        return image, self.tokenizer(
            prompt, padding="max_length",
            truncation=True, return_tensors="pt",
            max_length=self.tokenizer.model_max_length
        ).input_ids[0]