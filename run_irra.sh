#!/bin/bash
DATASET_NAME="CUHK-PEDES"
DATASET_ROOT="/data/jayn2u/lab_datasets"

CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name irra \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--root_dir $DATASET_ROOT \
--loss_names 'sdm+mlm+id' \
--num_epoch 60
