#!/bin/bash
#SBATCH --job-name=coco
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
python train_classifier.py --logdir coco-baselines/textual-inversion-mask-0.5-0 \
--synthetic-dir "/projects/rsalakhugroup/btrabucc/aug/\
textual-inversion-mask-0.5-0/{dataset}-{seed}-{examples_per_class}" \
--dataset coco --prompt "a photo of a {name}" \
--aug textual-inversion \
--guidance-scale 7.5 \
--strength 0.5 \
--mask 1 \
--inverted 0 \
--probs 1 \
--compose sequential --num-synthetic 10 --synthetic-probability 0.5 \
--num-trials 8 --examples-per-class 1 2 4 8 16