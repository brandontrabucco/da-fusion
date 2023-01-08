from transformers import CLIPTokenizer
from semantic_aug.few_shot_dataset import FewShotDataset
from torch.utils.data import Dataset
from typing import Tuple, List, Callable
import torchvision.transforms.functional as F

import torch
import random


DEFAULT_PROMPTS = [
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
    "a photo of a small {name}"]


class TextualInversionDataset(Dataset):

    def __init__(self, dataset: FewShotDataset, 
                 tokenizer: CLIPTokenizer, 
                 prompts: List[str] = DEFAULT_PROMPTS):

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.prompts = prompts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image, label = self.dataset[idx]
        metadata = self.dataset.get_metadata_by_idx(idx)

        initializer_ids = self.tokenizer.encode(
            metadata["name"], add_special_tokens=False)

        fine_tuned_tokens = []

        for idx in initializer_ids:

            token = self.tokenizer._convert_id_to_token(idx)
            token = token.replace("</w>", "")

            fine_tuned_tokens.append(f"<{token}>")

        metadata["name"] = " ".join(fine_tuned_tokens)
        prompt = random.choice(self.prompts).format(**metadata)

        return image, self.tokenizer(
            prompt, padding="max_length",
            truncation=True, return_tensors="pt",
            max_length=self.tokenizer.model_max_length
        ).input_ids[0]