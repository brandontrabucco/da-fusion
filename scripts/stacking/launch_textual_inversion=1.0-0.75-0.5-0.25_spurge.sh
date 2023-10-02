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
conda activate semantic-aug
cd ~/spurge/semantic-aug

RANK=$SLURM_ARRAY_TASK_ID WORLD_SIZE=$SLURM_ARRAY_TASK_COUNT \
python train_classifier.py --logdir spurge-baselines_v2_256/textual-inversion-1.0-0.75-0.5-0.25 \
--synthetic-dir "synthetic_images/textual_inversion/{dataset}-{seed}-{examples_per_class}" \
--dataset spurge --prompt "a photo of a {name}" \
--aug textual-inversion textual-inversion textual-inversion textual-inversion \
--guidance-scale 7.5 7.5 7.5 7.5 \
--strength 1.0 0.75 0.5 0.25 \
--mask 0 0 0 0 \
--inverted 0 0 0 0 \
--probs 0.25 0.25 0.25 0.25 \
--compose parallel --num-synthetic 50 --synthetic-probability 0.5 \
--num-trials 8 --examples-per-class 1 2 4 8 16
