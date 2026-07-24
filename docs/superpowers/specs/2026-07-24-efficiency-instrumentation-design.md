# IRRA Efficiency Instrumentation Design

## Goal

Measure the original IRRA implementation with the same process-level VRAM and
training-time definitions used by `lab_clip`, without changing model,
objective, optimizer, scheduler, data, or checkpoint-selection behavior.

## Metrics

Each training epoch records:

- `train/epoch_seconds`: CUDA-synchronized elapsed time for the training loop
  only.
- `train/examples_per_second`: processed training examples divided by
  `train/epoch_seconds`.
- `train/cumulative_gpu_hours`: sum of training-loop epoch durations.
- `train/peak_vram_allocated_mb`: PyTorch process peak allocated memory during
  the training loop.
- `train/peak_vram_reserved_mb`: PyTorch process peak reserved memory during
  the training loop.

Each validation epoch separately records:

- `val/epoch_seconds`: CUDA-synchronized elapsed validation time.
- `val/peak_vram_allocated_mb`: PyTorch process peak allocated memory during
  validation.
- `val/peak_vram_reserved_mb`: PyTorch process peak reserved memory during
  validation.

W&B `_runtime` remains the end-to-end wall-clock source. It is not relabeled as
training time.

## Architecture

`utils/efficiency.py` owns CUDA synchronization, monotonic timing, peak-memory
reset, peak-memory reads, and metric construction. `processor/processor.py`
places measurement boundaries immediately around the train and validation
loops. `utils/wandb_tracking.py` maps the resulting metrics to stable W&B keys
and declares peak summaries.

The train peak is read before validation begins. Validation resets peak-memory
statistics before evaluation, so train and validation peaks cannot contaminate
each other.

## Compatibility

CPU execution returns empty VRAM metrics. Disabled W&B remains a no-op.
Existing training and validation metric names remain unchanged. The added CUDA
synchronizations occur only at timing and peak-reading boundaries and do not
change gradients or model state.

## Testing

Unit tests verify:

- CPU and mocked-CUDA peak-memory behavior.
- CUDA synchronization around monotonic timing.
- Training and validation W&B payload keys.
- Disabled W&B no-op behavior.

The existing training entry point is compiled after the unit suite to catch
signature or import errors.
