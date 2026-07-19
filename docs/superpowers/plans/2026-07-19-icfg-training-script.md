# ICFG-PEDES Training Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an executable `run_icfg.sh` launcher that trains IRRA on ICFG-PEDES with the same settings and structure as the existing dataset launchers.

**Architecture:** Add one project-root shell script that delegates directly to the existing `train.py` entry point through `uv`. Keep dataset-specific values in the two variables already used by `run_irra.sh` and `run_rstpreid.sh`; do not change Python code or add dataset validation.

**Tech Stack:** Bash, `uv`, Python, IRRA `train.py`

## Global Constraints

- Work on branch `codex/icfg-training-script`.
- Set `DATASET_NAME` to `ICFG-PEDES`.
- Set `DATASET_ROOT` to `/mnt/data/lab_datasets`.
- Use experiment name `icfg` and GPU 0.
- Preserve image augmentation, batch size 64, MLM, `sdm+mlm+id`, and 60 epochs.
- Match the existing launchers and do not add dataset validation.
- Keep the human Git identity as primary author and append `Co-authored-by: Codex <codex@openai.com>` to commits.

## File Structure

- Create `run_icfg.sh`: select the ICFG-PEDES dataset and invoke the existing training entry point.

---

### Task 1: Add the ICFG-PEDES launcher

**Files:**
- Create: `run_icfg.sh`
- Reference: `run_irra.sh`
- Reference: `run_rstpreid.sh`

**Interfaces:**
- Consumes: the existing `train.py` CLI options `--name`, `--img_aug`, `--batch_size`, `--MLM`, `--dataset_name`, `--root_dir`, `--loss_names`, and `--num_epoch`.
- Produces: executable command `./run_icfg.sh` that starts ICFG-PEDES training on GPU 0.

- [x] **Step 1: Run the launcher acceptance check before implementation**

Run:

```bash
test -x run_icfg.sh
```

Expected: FAIL with exit status 1 because `run_icfg.sh` does not exist.

- [x] **Step 2: Create the minimal launcher**

Create `run_icfg.sh` with exactly:

```bash
#!/bin/bash
DATASET_NAME="ICFG-PEDES"
DATASET_ROOT="/mnt/data/lab_datasets"

CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name icfg \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--root_dir $DATASET_ROOT \
--loss_names 'sdm+mlm+id' \
--num_epoch 60
```

Then make it executable:

```bash
chmod +x run_icfg.sh
```

- [x] **Step 3: Verify syntax, executability, and exact launcher content**

Run:

```bash
bash -n run_icfg.sh
test -x run_icfg.sh
rg -n '^DATASET_NAME="ICFG-PEDES"$|^DATASET_ROOT="/mnt/data/lab_datasets"$|^--name icfg \\$|^--dataset_name \$DATASET_NAME \\$|^--root_dir \$DATASET_ROOT \\$' run_icfg.sh
git diff --check
```

Expected: `bash -n`, `test -x`, and `git diff --check` exit 0; `rg` prints five matching lines.

- [x] **Step 4: Review the scoped diff**

Run:

```bash
git diff -- run_icfg.sh
git status --short
```

Expected: the diff contains only the new launcher implementation, while status also lists this implementation plan until both are committed.

- [x] **Step 5: Commit the implementation and plan**

Run:

```bash
git add run_icfg.sh docs/superpowers/plans/2026-07-19-icfg-training-script.md
git commit -m "feat: add ICFG-PEDES training launcher" -m "Co-authored-by: Codex <codex@openai.com>"
```

Expected: one commit containing the executable launcher and this implementation plan, with the configured human Git identity as author and Codex as co-author.
