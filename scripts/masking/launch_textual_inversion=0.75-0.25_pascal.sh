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
python train_classifier.py --logdir pascal-baselines/textual-inversion-mask-0.75-0.25 \
--synthetic-dir "/projects/rsalakhugroup/btrabucc/aug/\
textual-inversion-mask-0.75-0.25/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo of a {name}" \
--aug textual-inversion textual-inversion \
--guidance-scale 7.5 7.5 \
--strength 0.75 0.25 \
--mask 1 1 \
--inverted 0 1 \
--probs 1 1 \
--compose sequential --num-synthetic 10 --synthetic-probability 0.5 \
--num-trials 8 --examples-per-class 1 2 4 8 16