from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.datasets.caltech101 import CalTech101Dataset
from semantic_aug.datasets.flowers102 import Flowers102Dataset
import numpy as np
import os


SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=erase
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-2-13,matrix-1-8
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=russ_reserved
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32g
 
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ldm
cd ~/spurge/semantic-aug/stable-diffusion

for CLASS_NAME in {}; do

python train-scripts/train-esd.py --prompt "$CLASS_NAME" --train_method 'full' --devices '0,0'

done
"""


PART_SIZE = 5


script_dir = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":

    for class_names, dataset_name in [
            (COCODataset().class_names, "coco"), 
            (PASCALDataset().class_names, "pascal"), 
            (SpurgeDataset().class_names, "spurge"), 
            (CalTech101Dataset().class_names, "caltech101"), 
            (Flowers102Dataset().class_names, "flowers102"),
            (ImageNetDataset().class_names, "imagenet")]:

        num_parts = int(np.ceil(len(class_names) / PART_SIZE))

        for i in range(num_parts):

            part_names = class_names[i*PART_SIZE:(i + 1)*PART_SIZE]

            with open(os.path.join(
                script_dir, f"erase_{dataset_name}_part{i}.sh"), "w") as f:

                f.write(SCRIPT_TEMPLATE.format(
                    " ".join([f"'{x}'" for x in part_names])))