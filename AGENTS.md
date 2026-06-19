# IRRA Agent Notes

## Dataset location

Training and evaluation datasets live on this machine at:

```
/mnt/data/lab_datasets
```

Pass this path to IRRA via `--root_dir`. The default in `utils/options.py` is `./data`, which does **not** point at the lab datasets unless you symlink or copy data there.

Expected layout under the root:

```
/mnt/data/lab_datasets/
├── CUHK-PEDES/
│   ├── imgs/
│   └── reid_raw.json
├── ICFG-PEDES/
│   ├── imgs/
│   └── ICFG-PEDES.json
└── RSTPReid/
    ├── imgs/
    └── data_captions.json
```

Code resolves paths as `{root_dir}/{dataset_name}/...` (see `datasets/build.py`, `datasets/cuhkpedes.py`, etc.).

## Dataset readiness (verified on this machine)

| Dataset       | `--dataset_name` | Status | Notes |
|---------------|------------------|--------|-------|
| CUHK-PEDES    | `CUHK-PEDES`     | Ready  | All splits; 0 missing images. Default in `run_irra.sh`. |
| RSTPReid      | `RSTPReid`       | Ready  | JSON uses `img_path`; matches `datasets/rstpreid.py`. |
| ICFG-PEDES    | `ICFG-PEDES`     | Partial | `imgs/train/` missing; ~33% of train annotations reference missing files. Not safe for full training until fixed. |

## Training scripts

`run_irra.sh` does not set `--root_dir`. For lab datasets, use:

```bash
--root_dir /mnt/data/lab_datasets
```

Example:

```bash
python train.py \
  --name irra \
  --img_aug \
  --batch_size 64 \
  --MLM \
  --dataset_name CUHK-PEDES \
  --root_dir /mnt/data/lab_datasets \
  --loss_names 'sdm+mlm+id' \
  --num_epoch 60
```

Run from the project root (`/mnt/data/IRRA`) so relative paths such as `./data` and `./logs` resolve correctly.

## CLIP BPE vocab (MLM)

When `--MLM` is enabled, `SimpleTokenizer` loads:

```
/mnt/data/IRRA/data/bpe_simple_vocab_16e6.txt.gz
```

This file is separate from `--root_dir` and is already present under the project `data/` directory.
