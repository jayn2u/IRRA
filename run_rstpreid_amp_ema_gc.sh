#!/bin/bash
# AMP + gradient checkpointing + EMA control run for RSTPReid.
DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name rstpreid_amp_ema_gc \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--amp \
--gradient_checkpointing \
--ema \
--ema_decay 0.999 \
--wandb
