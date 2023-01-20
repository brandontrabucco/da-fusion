#!/bin/bash
#SBATCH --job-name=inpainting
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-2-13,matrix-1-8,matrix-0-34
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=russ_reserved
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32g
#SBATCH --array=0-39
 
source ~/anaconda3/etc/profile.d/conda.sh
conda activate semantic-aug-1
cd ~/semantic-aug

RANK=$SLURM_ARRAY_TASK_ID WORLD_SIZE=40 python train_classifier.py \
--logdir ./pascal-baselines/inpainting \
--dataset pascal --aug inpainting --prompt "a photo" \
--strength 0.7 --num-synthetic 10 \
--synthetic-probability 0.5 --num-trials 8 \
--examples-per-class 1 2 4 8 16