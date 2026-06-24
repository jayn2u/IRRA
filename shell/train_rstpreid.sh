#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d /mnt/data/lab_datasets ]; then
  DATASET_ROOT=/mnt/data/lab_datasets
elif [ -d /data/jayn2u/lab_datasets ]; then
  DATASET_ROOT=/data/jayn2u/lab_datasets
else
  echo "lab_datasets not found" >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name rstpreid \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name RSTPReid \
--root_dir "$DATASET_ROOT" \
--loss_names 'sdm+mlm+id' \
--num_epoch 60
