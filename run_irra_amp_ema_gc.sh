#!/bin/bash
# Control run for the peer-review ablation: does plugging AMP + gradient
# checkpointing + EMA into the plain IRRA baseline reproduce the efficiency
# gains attributed to the simpler loss? Compare against run_irra.sh.
DATASET_NAME="CUHK-PEDES"
# Dataset root comes from DATASET_ROOT_DIR in env/.env (gitignored, per-machine).

CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name irra_amp_ema_gc \
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
