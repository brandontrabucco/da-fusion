#!/bin/bash
#SBATCH --job-name=focal_image_spurge
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-2-13,matrix-1-8,matrix-1-10
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=russ_reserved
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32g
#SBATCH --array=5,6,7
 
source ~/anaconda3/etc/profile.d/conda.sh
conda activate semantic-aug
cd ~/spurge/semantic-aug

RANK=$SLURM_ARRAY_TASK_ID WORLD_SIZE=40 python train_classifier.py \
--logdir /home/kdoherty/spurge/semantic-aug/results/textual-inversion-0.5 \
--dataset spurge --aug textual-inversion --prompt "a drone image of {name}" \
--checkpoint /home/kdoherty/spurge/models/stable-diffusion-v1-4 \
--embed-path /home/kdoherty/spurge/semantic-aug/data/all_embeddings.bin \
--synthetic-dir /home/kdoherty/spurge/semantic-aug/results/synthetic_images/{dataset}-{aug}-{seed}-{examples_per_class} \
--strength 0.5 --num-synthetic 50 \
--synthetic-probability 0.5 --num-trials 8 \
--examples-per-class 1 2 4 8 16 
