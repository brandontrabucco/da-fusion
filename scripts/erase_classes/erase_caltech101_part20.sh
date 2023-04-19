#!/bin/bash
#SBATCH --job-name=erase
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-2-13,matrix-1-8
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=russ_reserved
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32g
 
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ldm
cd ~/spurge/semantic-aug/stable-diffusion

for CLASS_NAME in 'wrench' 'yin yang'; do

python train-scripts/train-esd.py --prompt "$CLASS_NAME" --train_method 'full' --devices '0,0'

done
