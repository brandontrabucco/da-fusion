#!/bin/bash

cd /home/kdoherty/spurge/semantic-aug/scripts/without_cryptic/baseline
sbatch launch_baseline_spurge.sh
cd /home/kdoherty/spurge/semantic-aug/scripts/without_cryptic/real_guidance
sbatch launch_real_guidance=0.5_spurge.sh
cd /home/kdoherty/spurge/semantic-aug/scripts/without_cryptic/textual_inversion
sbatch launch_textual_inversion=0.5_spurge.sh
