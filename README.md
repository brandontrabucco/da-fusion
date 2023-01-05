# Prompt-Based Data Augmentation

Existing data augmentations like rotations and re-colorizations provide diversity but preserve semantics. We explore how prompt-based generative models complement existing data augmentations by controlling image semantics via prompts. Our generative data augmentations build on Stable Diffusion and improve visual few-shot learning.

## Installation

To install the package, first create a `conda` environment.

```bash
conda create -n semantic-aug python=3.7 pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
conda activate semantic-aug
pip install diffusers["torch"] transformers
```

Then download and install the source code.

```bash
git clone git@github.com:brandontrabucco/semantic-aug.git
pip install -e semantic-aug
```

## Spurge Experiments

We assume a default location for the spurge dataset at `data/spurge/absent/*.png` for images of the absent class and `data/spurge/apparent/*.png` for images of the apparent class. This default may be changed via the `data_dir` argument to the `semantic_aug.datasets.spurge.SpurgeDataset` class `__init__` method.

You may evaluate the baseline on our spurge dataset with the following script.

```bash
python train_classifier.py \
--logdir ./baselines/baseline \
--dataset spurge --aug none \
--strength 0.0 --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8
```

Real Guidance may be evaluated on our spurge dataset using the following arguments.

```bash
python train_classifier.py \
--logdir ./baselines/real-guidance-0.5 \
--dataset spurge --aug real-guidance \
--prompt "a woodland seen from a drone" \
--strength 0.5 --num-synthetic 20 \
--synthetic-probability 0.5 --num-trials 8
```

Textual inversion may be evaluated on our spurge dataset using the following arguments.

```bash
python train_classifier.py \
--logdir ./baselines/textual-inversion-0.5 \
--dataset spurge --aug textual-inversion \
--prompt "a drone image of {name}" \
--strength 0.5 --num-synthetic 20 \
--synthetic-probability 0.5 --num-trials 8
```

## ImageNet Experiments

You may evaluate the baseline on ImageNet with the following script.

```bash
python train_classifier.py \
--logdir ./imagenet-baselines/baseline \
--dataset imagenet --aug none \
--strength 0.0 --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8
```

Real Guidance may be evaluated on ImageNet using the following arguments.

```bash
python train_classifier.py \
--logdir ./imagenet-baselines/real-guidance-0.5 \
--dataset imagenet --aug real-guidance \
--prompt "a photo of a {name}" \
--strength 0.5 --num-synthetic 1 \
--synthetic-probability 0.5 --num-trials 8
```

See `scripts/launch_baseline_imagenet.sh` for a parallelized version for slurm.
