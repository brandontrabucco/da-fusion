#!/bin/bash

cd ~/spurge/semantic-aug
rm -r data/spurge
mkdir data/spurge
unzip data/stable_diffusion_donors.zip -d data/spurge

cd ~/spurge/semantic-aug/scripts/250_qtr/integrated_stacking

#for p in 0.2 0.4 0.6 0.8;
for p in 0.1 0.3 0.5 0.7 0.9;
do
 sbatch integrated_stacking_accepts_prob.sh $p
done
