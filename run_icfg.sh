#!/bin/bash
DATASET_NAME="ICFG-PEDES"

CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name icfg \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--wandb
