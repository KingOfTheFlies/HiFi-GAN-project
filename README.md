<h1 align="center">HiFi-GAN</h1>

## About

The model based on [HiFi-GAN](https://arxiv.org/abs/2010.05646).

See [wandb report](https://wandb.ai/dungeon_as_fate/HiFi%20GAN).

## Installation

0. Create new conda environment:
```bash
conda create -n hifi_env python=3.10

conda activate hifi_env
``` 

1. Install all requirements.
```bash
pip install -r requirements.txt
```

## Train
   Training script for DPRNN:
   ```bash
   python train.py
   ```

## There is no Inference Inference due to a vacuum cleaner (more details in the report).