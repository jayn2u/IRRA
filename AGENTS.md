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
| CUHK-PEDES    | `CUHK-PEDES`     | Ready  | All splits; 0 missing images. Train via `shell/train_cuhk_pedes.sh`. |
| RSTPReid      | `RSTPReid`       | Ready  | JSON uses `img_path`; matches `datasets/rstpreid.py`. |
| ICFG-PEDES    | `ICFG-PEDES`     | Partial | `imgs/train/` missing; ~33% of train annotations reference missing files. Not safe for full training until fixed. |

## Config injection and dataset naming

Lab pedestrian evaluation wrappers (for example, `sugarcrepe-pedes.py` and future YAML-based retrieval scripts) must **not** silently default to a dataset. Every run must inject the YAML config path through an environment variable. If the variable is missing or empty, the script must raise an error.

| Script | Environment variable | Config prefix |
|--------|---------------------|---------------|
| `sugarcrepe-pedes.py` | `SUGARCREPE_CONFIG` | `sugarcrepe` |
| `text-to-image-retrieval.py` (when added) | `RETRIEVAL_CONFIG` | `text_to_image_retrieval` |

Per-dataset config files use the pattern `configs/{task}_{dataset_slug}.yaml`, where `dataset_slug` is one of:

- `cuhk_pedes` — CUHK-PEDES (`dataset: cuhk-pedes`)
- `icfg_pedes` — ICFG-PEDES (`dataset: icfg-pedes`)
- `rstpreid` — RSTPReid (`dataset: rstpreid`)

Examples:

- `configs/sugarcrepe_cuhk_pedes.yaml`
- `configs/sugarcrepe_icfg_pedes.yaml`
- `configs/sugarcrepe_rstpreid.yaml`
- `configs/text_to_image_retrieval_cuhk_pedes.yaml` (future)

Each YAML must set `dataset` explicitly. Do not rely on code defaults for dataset selection. Shell scripts under `shell/` should export the matching config path before calling the Python entry point.

Example:

```bash
export SUGARCREPE_CONFIG=configs/sugarcrepe_icfg_pedes.yaml
uv run python sugarcrepe-pedes.py
```

Legacy `train.py` / `test.py` use `--dataset_name` (`CUHK-PEDES`, `ICFG-PEDES`, `RSTPReid`) and `--root_dir`. That CLI pattern is for original IRRA training and checkpoint evaluation, **not** for new lab YAML-based pedestrian probes. New wrappers should use environment config injection instead of `--dataset_name`.

Supported lab pedestrian datasets:

| Dataset slug | Annotation directory | Annotation file | Image path field |
|--------------|---------------------|-----------------|------------------|
| `cuhk_pedes` | `CUHK-PEDES/` | `reid_raw.json` | `file_path` |
| `icfg_pedes` | `ICFG-PEDES/` | `ICFG-PEDES.json` | `file_path` |
| `rstpreid` | `RSTPReid/` | `data_captions.json` | `img_path` |

## Training scripts

Per-dataset training wrappers live under `shell/`:

- `shell/train_cuhk_pedes.sh` (`--dataset_name CUHK-PEDES`)
- `shell/train_icfg_pedes.sh` (`--dataset_name ICFG-PEDES`)
- `shell/train_rstpreid.sh` (`--dataset_name RSTPReid`)

Each script resolves `DATASET_ROOT` from `/mnt/data/lab_datasets` or `/data/jayn2u/lab_datasets` and runs from the project root so relative paths such as `./data` and `./logs` resolve correctly.

Example:

```bash
sh shell/train_cuhk_pedes.sh
```

## CLIP BPE vocab (MLM)

When `--MLM` is enabled, `SimpleTokenizer` loads:

```
/mnt/data/IRRA/data/bpe_simple_vocab_16e6.txt.gz
```

This file is separate from `--root_dir` and is already present under the project `data/` directory.
