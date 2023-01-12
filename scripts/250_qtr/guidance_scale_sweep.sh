#!/bin/bash

cd ~/spurge/semantic-aug
rm -r data/spurge
mkdir data/spurge
unzip data/stable_diffusion_donors.zip -d data/spurge

cd ~/spurge/semantic-aug/scripts/250_qtr/textual_inversion
sbatch launch_textual_inversion=1.0_3.0_spurge.sh
sbatch launch_textual_inversion=1.0_6.0_spurge.sh
sbatch launch_textual_inversion=1.0_9.0_spurge.sh
sbatch launch_textual_inversion=1.0_12.0_spurge.sh
