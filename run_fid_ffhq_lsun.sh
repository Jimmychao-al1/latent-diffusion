#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/jimmy/anaconda3/envs/ldm/bin/python

$PYTHON scripts/sample_diffusion.py \
  --resume models/ldm/ffhq256/model.ckpt \
  --dataset_tag ffhq256 \
  --n_samples 50000 \
  --custom_steps 200 \
  --batch_size 16 \
  -e 1 \
  --no_npz \
  --eval_fid \
  --real_image_dir ffhq-dataset/images1024x1024 \
  --num_fid_samples 50000 \
  --fid_batch_size 32 \
  --img_size 256 \
  --seed 0
