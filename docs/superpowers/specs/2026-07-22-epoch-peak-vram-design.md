# Epoch Peak VRAM W&B Logging Design

## Goal

Record the peak GPU memory required by each complete training epoch in W&B,
in GiB. The metrics must expose both memory actively allocated to tensors and
memory reserved by PyTorch's CUDA caching allocator so runs can be compared and
GPU capacity can be planned.

## Metric Semantics

Each epoch emits these history metrics on the existing `epoch` step axis:

- `epoch/peak_vram_allocated_gib`: the largest
  `torch.cuda.max_memory_allocated()` value observed on any training rank during
  the epoch, divided by `1024 ** 3`.
- `epoch/peak_vram_reserved_gib`: the largest
  `torch.cuda.max_memory_reserved()` value observed on any training rank during
  the epoch, divided by `1024 ** 3`.

The values cover the whole epoch: its training loop and, on evaluation epochs,
validation. They measure memory managed by PyTorch's CUDA allocator. They do not
claim to include CUDA contexts or memory allocated by unrelated processes;
W&B's sampled system metrics remain available for that device-level view.

For distributed training, each rank measures its local CUDA device and the
reported value is the maximum across ranks. This represents the minimum per-GPU
capacity needed by the most memory-intensive rank, rather than a sum across
devices.

## Training Flow

At the start of every epoch, reset the local CUDA peak-memory statistics. Run
training and optional validation as usual. At the end of the epoch, read the
local allocated and reserved peaks. In distributed execution, reduce both
values with `MAX` across ranks. Rank 0 then logs the resulting GiB values to the
existing W&B session.

All ranks must participate in the reduction after rank-0-only validation. This
also creates an explicit epoch boundary before the next distributed training
epoch begins.

## Components

Keep CUDA measurement and distributed aggregation in a small helper in the
training processor. Extend the W&B epoch logging helper with optional peak VRAM
arguments so metric naming and scalar conversion remain owned by
`utils/wandb_tracking.py`.

The existing training metrics may still be logged after the training loop. The
VRAM metrics are logged at the end of the complete epoch as a separate W&B
payload using the same explicit `epoch` value. This avoids delaying existing
training metrics while ensuring validation allocations are included.

## Error and Compatibility Behavior

When CUDA is unavailable, skip peak-stat reset and omit the VRAM metrics. W&B
disabled mode remains a no-op. Single-GPU training does not initialize or call
distributed collectives. No new command-line flag is needed because measurement
is lightweight and the new metrics only appear when CUDA and W&B logging are
active.

## Testing

Unit tests will cover:

- conversion from bytes to GiB and the exact W&B metric names;
- omission of the metrics when no values are supplied;
- CUDA peak reset and reads through controllable test doubles;
- maximum aggregation semantics for distributed ranks where practical without
  requiring physical GPUs.

The repository's relevant test suite will run with `uv run python` as required
by the project instructions. Static compilation/import checks will supplement
unit tests without requiring a training dataset or GPU.
