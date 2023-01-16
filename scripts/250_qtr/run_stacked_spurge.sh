#!/bin/bash

cd ~/spurge/semantic-aug
rm -r data/spurge
mkdir data/spurge
unzip data/stable_diffusion_donors.zip -d data/spurge

cd ~/spurge/semantic-aug/scripts/250_qtr/stack_augs
sbatch stack_augs_spurge.sh
