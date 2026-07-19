# ICFG-PEDES Training Launcher Design

## Goal

Add a convenient shell launcher for training IRRA on ICFG-PEDES while matching the existing `run_irra.sh` and `run_rstpreid.sh` scripts.

## Design

Create an executable `run_icfg.sh` in the project root. It will use the same command structure and training options as the existing launchers, with only the dataset-specific values changed:

- Set `DATASET_NAME` to `ICFG-PEDES`.
- Set `DATASET_ROOT` to `/mnt/data/lab_datasets`.
- Use `icfg` as the experiment name.
- Select GPU 0 with `CUDA_VISIBLE_DEVICES=0`.
- Run `train.py` through `uv run python`.
- Preserve batch size 64, image augmentation, MLM, `sdm+mlm+id` losses, and 60 epochs.

The launcher will not add dataset validation or change the Python training pipeline. This intentionally matches the behavior of the other launchers. Because the local ICFG-PEDES dataset is incomplete, training can fail when it encounters a missing image.

## Verification

Validate the script without starting a training run:

1. Check shell syntax with `bash -n run_icfg.sh`.
2. Compare its structure and options with the two existing launchers.
3. Confirm the file is executable.
