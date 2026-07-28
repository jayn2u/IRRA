#!/bin/bash
DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name rstpreid \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--wandb
