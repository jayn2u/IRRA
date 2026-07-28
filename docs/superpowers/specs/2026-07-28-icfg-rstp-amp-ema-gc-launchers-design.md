# ICFG-PEDES and RSTPReid AMP/EMA/GC Launchers Design

## Goal

Add dedicated launchers for running the existing ICFG-PEDES and RSTPReid
training configurations with automatic mixed precision, gradient
checkpointing, and exponential moving average weights enabled.

## Approach

Create `run_icfg_amp_ema_gc.sh` and `run_rstpreid_amp_ema_gc.sh` beside the
existing launchers. Each new script keeps its dataset's current training
arguments and adds the same optimization flags and EMA decay used by
`run_irra_amp_ema_gc.sh`.

The launchers remain independent rather than introducing a shared wrapper.
This follows the repository's current structure, keeps each experiment
directly runnable, and avoids changing the behavior of the existing baseline
scripts.

## Launcher Contracts

`run_icfg_amp_ema_gc.sh` invokes:

- `uv run python train.py`
- experiment name `icfg_amp_ema_gc`
- dataset `ICFG-PEDES`
- the existing ICFG-PEDES augmentation, batch, MLM, loss, epoch, and W&B
  settings
- `--amp`, `--gradient_checkpointing`, `--ema`, and `--ema_decay 0.999`

`run_rstpreid_amp_ema_gc.sh` invokes the same command shape with experiment
name `rstpreid_amp_ema_gc` and dataset `RSTPReid`.

Both launchers expose GPU 0 to the training command, matching the current
scripts.

## Repository Configuration

Track the existing `.codex/config.toml` in the same pull request, preserving
its `sandbox_mode = "danger-full-access"` setting without modification.

## Validation

An automated test executes each launcher with a temporary fake `uv`
executable. It asserts the complete command-line argument vector and the
`CUDA_VISIBLE_DEVICES=0` environment visible to the command. This verifies
the launchers' observable behavior without starting model training.

Run the launcher test first while the files are absent to prove it detects
the missing feature. After implementation, run that test, shell syntax
checks, and the complete Python unit test suite.

## Pull Request

Commit only the two launchers, their test, the approved design and
implementation plan, and `.codex/config.toml`. Push
`codex/icfg-rstp-amp-ema-gc-scripts` and open a draft pull request targeting
`main`.
