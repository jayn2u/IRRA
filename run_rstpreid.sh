#!/bin/bash
set -euo pipefail

DATASET_NAME="RSTPReid"
# Same storage, different mount depending on the machine.
if [ -d /mnt/data/lab_datasets ]; then
  DATASET_ROOT="/mnt/data/lab_datasets"
else
  DATASET_ROOT="/data/jayn2u/lab_datasets"
fi

CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name rstpreid \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--root_dir $DATASET_ROOT \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--wandb
