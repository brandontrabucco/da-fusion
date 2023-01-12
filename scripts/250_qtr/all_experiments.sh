#!/bin/bash

cd ~/spurge/semantic-aug
rm -r data/spurge
mkdir data/spurge
unzip data/stable_diffusion_donors.zip -d data/spurge

cd ~/spurge/semantic-aug/scripts/250_qtr/baseline
sbatch launch_baseline_spurge.sh
cd ~/spurge/semantic-aug/scripts/250_qtr/real_guidance
sbatch launch_real_guidance=0.5_spurge.sh
cd ~/spurge/semantic-aug/scripts/250_qtr/textual_inversion
sbatch launch_textual_inversion=0.5_spurge.sh
sbatch launch_textual_inversion=1.0_spurge.sh
sbatch launch_textual_inversion=0.5_3.25_spurge.sh
