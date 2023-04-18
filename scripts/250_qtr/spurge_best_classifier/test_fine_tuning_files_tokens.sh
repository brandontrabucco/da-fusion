#!/bin/bash
#SBATCH --job-name=poolfix
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-2-13,matrix-1-8,matrix-1-10
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=russ_reserved
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32g
#SBATCH --array=0
 
source ~/anaconda3/etc/profile.d/conda.sh
conda activate semantic-aug
cd ~/spurge/semantic-aug

RANK=$SLURM_ARRAY_TASK_ID python ./semantic_aug/test_image_copy_token.py no_spurge
RANK=$SLURM_ARRAY_TASK_ID python ./semantic_aug/test_image_copy_token.py leafy_spurge

# RANK=$SLURM_ARRAY_TASK_ID python spurge_classifier_five_fold.py \
# --logdir ./results/250_qtr/spurge_five_fold_pooled_fixed \
# --dataset spurge --aug integrated-stacking --prompt "a drone image of {name}" \
# --checkpoint /home/kdoherty/spurge/models/stable-diffusion-v1-4 \
# --embed-path ./data/spurge/pooled_embeddings/seed_$SLURM_ARRAY_TASK_ID/both_embeddings.bin \
# --synthetic-dir ./results/250_qtr/synthetic_images/spurge_five_fold_pooled_fixed \
# --num-synthetic 10 \
# --num-epochs 500 \
# --synthetic-probability 0.5
