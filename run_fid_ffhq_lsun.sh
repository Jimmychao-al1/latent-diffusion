#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/jimmy/anaconda3/envs/ldm/bin/python

$PYTHON scripts/sample_diffusion.py \
  --resume models/ldm/ffhq256/model.ckpt \
  --dataset_tag ffhq256 \
  --n_samples 50000 \
  --custom_steps 200 \
  --batch_size 16 \
  -e 0 \
  --no_npz \
  --eval_fid \
  --real_lmdb datasets/ffhq256_lmdb \
  --lmdb_resolution 256 \
  --lmdb_zfill 5 \
  --num_fid_samples 50000 \
  --fid_batch_size 32 \
  --img_size 256 \
  seed=0

$PYTHON scripts/sample_diffusion.py \
  --resume models/ldm/lsun_bedrooms/model.ckpt \
  --dataset_tag lsun_bedroom256 \
  --n_samples 50000 \
  --custom_steps 200 \
  --batch_size 16 \
  -e 0 \
  --no_npz \
  --eval_fid \
  --real_lmdb datasets/bedroom256_lmdb \
  --lmdb_resolution 256 \
  --lmdb_zfill 7 \
  --num_fid_samples 50000 \
  --fid_batch_size 32 \
  --img_size 256 \
  seed=0
