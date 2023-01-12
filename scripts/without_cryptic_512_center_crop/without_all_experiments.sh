#!/bin/bash

cd ~/spurge/semantic-aug
rm -r data/spurge
mkdir data/spurge
unzip data/sd_final_512.zip -d data/spurge

cd ~/spurge/semantic-aug/scripts/without_cryptic_512_center_crop/baseline
sbatch launch_baseline_spurge.sh
cd ~/spurge/semantic-aug/scripts/without_cryptic_512_center_crop/real_guidance
sbatch launch_real_guidance=0.5_spurge.sh
cd ~/spurge/semantic-aug/scripts/without_cryptic_512_center_crop/textual_inversion
sbatch launch_textual_inversion=0.5_spurge.sh
