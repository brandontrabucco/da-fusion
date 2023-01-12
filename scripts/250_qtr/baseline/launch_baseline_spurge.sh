#!/bin/bash
#SBATCH --job-name=base250
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
--logdir /home/kdoherty/spurge/semantic-aug/results/250_qtr/baseline \
--dataset spurge --aug none \
--checkpoint /home/kdoherty/spurge/models/stable-diffusion-v1-4 \
--strength 0.0 --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8 \
--examples-per-class 1 2 4 8 16
