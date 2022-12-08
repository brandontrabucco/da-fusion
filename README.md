# Semantic Controls For Data Augmentation

We provide a set of tools for adding semantic controls to your data augmentation pipeline.

```
conda create -n semantic-aug python=3.7 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate semantic-aug
pip install diffusers["torch"]

git clone https://github.com/brandontrabucco/semantic-aug
pip install semantic-aug
```