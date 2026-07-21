# IRRA Agent Notes

Use `uv run python` to execute Python code.

## Dataset location

Training and evaluation datasets are stored at one of:

- `/mnt/data/lab_datasets`
- `/data/jayn2u/lab_datasets`

These paths refer to the same storage. Use whichever exists on the current machine.

Pass the chosen path to IRRA via `--root_dir`. The default in `utils/options.py` is `./data`, which does **not** point at the lab datasets unless you symlink or copy data there.

Expected layout under the root:

```
{root_dir}/
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
| CUHK-PEDES    | `CUHK-PEDES`     | Ready  | 40206 anns across train/val/test; 0 missing images. Default in `run_irra.sh`. |
| RSTPReid      | `RSTPReid`       | Ready  | 20505 anns; JSON uses `img_path`, matching `datasets/rstpreid.py`. |
| ICFG-PEDES    | `ICFG-PEDES`     | Ready  | 54522 anns (train 34674 / test 19848); 0 missing images. `imgs/train/` has since been populated — the earlier "partial" note is obsolete. |

Re-verify with a JSON-vs-filesystem check after any dataset move; the counts above
were confirmed on 2026-07-21.

## Training scripts

`run_irra.sh`, `run_icfg.sh`, and `run_rstpreid.sh` pick `--root_dir` automatically:
`/mnt/data/lab_datasets` when that mount exists, otherwise `/data/jayn2u/lab_datasets`.
When invoking `train.py` by hand, pass whichever exists:

```bash
--root_dir /data/jayn2u/lab_datasets
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
  --num_epoch 60 \
  --wandb
```

Run from the project root (`/mnt/data/IRRA`) so relative paths such as `./data` and `./logs` resolve correctly.

## Weights & Biases logging

`--wandb` turns on per-epoch logging to W&B (`utils/wandb_tracking.py`, modelled on
lab_clip's `src/wandb_tracking.py`). Without the flag every logging call is a no-op,
so training behaves exactly as before.

Credentials come from `env/.env` (gitignored; `env/.env.example` is the template).
Process environment variables win over the file. Override the path with
`--wandb_env_file`.

```
WANDB_API_KEY=...
WANDB_ENTITY=tonychoi179-jayn2u
WANDB_PROJECT=irra
```

Logged every epoch:

| Key | Meaning |
|-----|---------|
| `val/t2i_error@{1,5,10}` | validation error, `100 - R@k` (text→image; the primary curve) |
| `val/i2t_error@{1,5,10}` | validation error (image→text) |
| `val/t2i_R{1,5,10}`, `val/t2i_mAP`, `val/t2i_mINP` | raw retrieval metrics |
| `val/i2t_*` | same for image→text |
| `train/loss`, `train/sdm_loss`, `train/mlm_loss`, `train/id_loss`, `train/*_acc` | epoch averages from the meters |
| `train/lr`, `train/temperature` | scheduler LR and learned temperature |

Everything is stepped by `epoch` via `define_metric`, so W&B plots against the
epoch axis. Validation runs on `--eval_period` epochs (default 1) over
`--val_dataset` (default `test`). Run summary carries `val/best_t2i_R1`,
`val/best_t2i_error@1`, and `val/best_epoch`. Rank 0 owns the run under DDP.

Validation now always computes i2t metrics as well (`Evaluator.eval(..., i2t_metric=True)`),
which is why the i2t curves exist; `Evaluator.eval` still returns t2i R1 unless
`return_metrics=True` is passed, so `test.py` is unaffected.

Other flags: `--wandb_project`, `--wandb_entity`, `--wandb_run_name` (default: the
timestamped output dir name), `--wandb_group` (default: dataset name), `--wandb_tags`,
`--wandb_notes`. The run id is written to `{output_dir}/wandb_meta.json` and
`{output_dir}/wandb_run_id` for downstream jobs to attach to.

## CLIP BPE vocab (MLM)

When `--MLM` is enabled, `SimpleTokenizer` loads:

```
/mnt/data/IRRA/data/bpe_simple_vocab_16e6.txt.gz
```

This file is separate from `--root_dir` and is already present under the project `data/` directory.
