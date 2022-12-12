#!/bin/bash
#SBATCH --job-name=spurge
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-16,matrix-1-4,matrix-2-13,matrix-1-8
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=russ_reserved
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128g
 
source ~/anaconda3/etc/profile.d/conda.sh
conda activate semantic-aug
cd ~/spurge/semantic-aug

torchrun --standalone --nnodes 1 --nproc_per_node 4 \
train_classifier.py --logdir ./baselines/baseline --aug none \
--strength 0.0 --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8