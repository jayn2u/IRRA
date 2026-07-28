#!/bin/bash
DATASET_NAME="CUHK-PEDES"
# Dataset root comes from DATASET_ROOT_DIR in env/.env (gitignored, per-machine).

CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name irra \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--wandb
