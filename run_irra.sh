#!/bin/bash
# Load environment variables from .env if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

DATASET_NAME=${DATASET_NAME}

CUDA_VISIBLE_DEVICES=0 \
uv run train.py \
--name irra \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60
