#!/bin/bash
#SBATCH --job-name=TIGS3
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-2-13,matrix-1-8,matrix-1-10
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=russ_reserved
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32g
#SBATCH --array=0-39
 
source ~/anaconda3/etc/profile.d/conda.sh
conda activate semantic-aug
cd ~/spurge/semantic-aug

RANK=$SLURM_ARRAY_TASK_ID WORLD_SIZE=40 python train_classifier.py \
--logdir /home/kdoherty/spurge/semantic-aug/results/250_qtr/textual-inversion-1.0-3.0 \
--dataset spurge --aug textual-inversion --prompt "a drone image of {name}" \
--checkpoint /home/kdoherty/spurge/models/stable-diffusion-v1-4 \
--embed-path /home/kdoherty/spurge/semantic-aug/data/all_embeddings_250_qtr.bin \
--synthetic-dir /home/kdoherty/spurge/semantic-aug/results/250_qtr/synthetic_images/{dataset}-{aug}-{strength}-{guidance_scale}-{seed}-{examples_per_class} \
--strength 1 --num-synthetic 50 \
--guidance-scale 3.0 \
--synthetic-probability 0.5 --num-trials 8 \
--examples-per-class 1 2 4 8 16 
