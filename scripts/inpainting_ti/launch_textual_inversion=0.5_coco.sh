#!/bin/bash
#SBATCH --job-name=spurge
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-2-13,matrix-1-8
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
--logdir ./coco-baselines/inpainting-ti \
--dataset coco --aug inpainting-ti --prompt "a photo of a {name}" \
--strength 0.5 --num-synthetic 10 \
--synthetic-probability 0.5 --num-trials 8 \
--examples-per-class 1 2 4 8 16