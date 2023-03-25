#!/bin/bash
#SBATCH --job-name=pascal
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-2-13,matrix-1-8
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

RANK=$SLURM_ARRAY_TASK_ID WORLD_SIZE=$SLURM_ARRAY_TASK_COUNT \
python train_classifier.py --logdir num-synthetic-pascal-baselines/real-guidance-0.5-num-synthetic-5 \
--synthetic-dir "/projects/rsalakhugroup/btrabucc/aug/\
real-guidance-0.5-num-synthetic-5/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo" \
--aug real-guidance --guidance-scale 7.5 \
--strength 0.5 --mask 0 --inverted 0 \
--num-synthetic 5 --synthetic-probability 0.5 \
--num-trials 8 --examples-per-class 1 2 4 8 16