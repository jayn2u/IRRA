# Epoch Peak VRAM W&B Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log the maximum per-GPU PyTorch allocated and reserved VRAM for each complete epoch to W&B in GiB.

**Architecture:** The W&B helper owns byte-to-GiB conversion and metric names. The training processor resets CUDA peak statistics at each epoch boundary, reads the local peaks after optional validation, reduces both with `MAX` across DDP ranks, and lets rank 0 log them.

**Tech Stack:** Python 3.11, PyTorch 2.9, W&B, standard-library `unittest` and `unittest.mock`.

## Global Constraints

- Log `epoch/peak_vram_allocated_gib` and `epoch/peak_vram_reserved_gib` using `1024 ** 3` bytes per GiB.
- Cover the whole epoch, including validation when validation runs in that epoch.
- In DDP, log the maximum individual-rank peak, not the sum across GPUs.
- When CUDA is unavailable, omit both VRAM metrics.
- Keep W&B-disabled behavior a no-op and add no command-line flag.
- Run Python through `uv run python`.

---

### Task 1: W&B GiB Metric Logging

**Files:**
- Create: `tests/test_wandb_tracking.py`
- Modify: `utils/wandb_tracking.py`

**Interfaces:**
- Consumes: `WandbSession.log(metrics)`.
- Produces: `log_peak_vram_metrics(session, epoch, allocated_bytes=None, reserved_bytes=None)`.

- [ ] **Step 1: Write the failing W&B metric tests**

```python
import unittest

from utils.wandb_tracking import WandbSession, log_peak_vram_metrics


class FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))


class PeakVramLoggingTest(unittest.TestCase):
    def test_logs_peak_bytes_as_gib(self):
        run = FakeRun()
        session = WandbSession(run)

        log_peak_vram_metrics(
            session,
            epoch=3,
            allocated_bytes=8 * 1024 ** 3,
            reserved_bytes=10 * 1024 ** 3,
        )

        self.assertEqual(len(run.logged), 1)
        self.assertEqual(run.logged[0][1], None)
        self.assertEqual(run.logged[0][0], {
            "epoch": 3,
            "epoch/peak_vram_allocated_gib": 8.0,
            "epoch/peak_vram_reserved_gib": 10.0,
        })

    def test_omits_vram_log_when_measurements_are_unavailable(self):
        run = FakeRun()

        log_peak_vram_metrics(WandbSession(run), epoch=1)

        self.assertEqual(run.logged, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the missing helper fails**

Run: `uv run python -m unittest tests.test_wandb_tracking -v`

Expected: FAIL because `log_peak_vram_metrics` cannot be imported.

- [ ] **Step 3: Implement the minimal metric helper and epoch namespace**

Add this metric definition beside the existing train/validation definitions in
`start_train_run`:

```python
run.define_metric("epoch/*", step_metric="epoch")
```

Add this helper after `log_train_epoch_metrics`:

```python
def log_peak_vram_metrics(session, epoch, allocated_bytes=None,
                          reserved_bytes=None):
    """Log complete-epoch CUDA allocator peaks in GiB."""
    if not session.enabled:
        return
    if allocated_bytes is None or reserved_bytes is None:
        return
    bytes_per_gib = 1024 ** 3
    session.log({
        "epoch": epoch,
        "epoch/peak_vram_allocated_gib": _scalar(allocated_bytes) / bytes_per_gib,
        "epoch/peak_vram_reserved_gib": _scalar(reserved_bytes) / bytes_per_gib,
    })
```

- [ ] **Step 4: Run the focused tests**

Run: `uv run python -m unittest tests.test_wandb_tracking -v`

Expected: 2 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add tests/test_wandb_tracking.py utils/wandb_tracking.py
git commit -m "feat: log epoch peak VRAM in GiB"
```

---

### Task 2: Complete-Epoch CUDA Peak Measurement

**Files:**
- Create: `tests/test_peak_vram.py`
- Modify: `processor/processor.py`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: `log_peak_vram_metrics(session, epoch, allocated_bytes, reserved_bytes)` from Task 1.
- Produces: `_reset_peak_vram_stats()` and `_peak_vram_bytes(distributed=False)` in `processor/processor.py`.

- [ ] **Step 1: Write failing CUDA peak helper tests**

```python
import unittest
from unittest.mock import patch

from processor.processor import _peak_vram_bytes, _reset_peak_vram_stats


class FakePeakTensor:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class PeakVramMeasurementTest(unittest.TestCase):
    @patch("processor.processor.torch.cuda.reset_peak_memory_stats")
    @patch("processor.processor.torch.cuda.is_available", return_value=True)
    def test_resets_cuda_peak_stats(self, _is_available, reset):
        self.assertTrue(_reset_peak_vram_stats())
        reset.assert_called_once_with()

    @patch("processor.processor.torch.cuda.reset_peak_memory_stats")
    @patch("processor.processor.torch.cuda.is_available", return_value=False)
    def test_skips_reset_without_cuda(self, _is_available, reset):
        self.assertFalse(_reset_peak_vram_stats())
        reset.assert_not_called()

    @patch("processor.processor.torch.tensor")
    @patch("processor.processor.torch.cuda.max_memory_reserved", return_value=20)
    @patch("processor.processor.torch.cuda.max_memory_allocated", return_value=12)
    @patch("processor.processor.torch.cuda.is_available", return_value=True)
    def test_reads_local_peak_bytes(self, _available, _allocated, _reserved,
                                    tensor):
        tensor.return_value = FakePeakTensor([12, 20])

        self.assertEqual(_peak_vram_bytes(), (12, 20))

    @patch("processor.processor.torch.distributed.all_reduce")
    @patch("processor.processor.torch.tensor")
    @patch("processor.processor.torch.cuda.max_memory_reserved", return_value=20)
    @patch("processor.processor.torch.cuda.max_memory_allocated", return_value=12)
    @patch("processor.processor.torch.cuda.is_available", return_value=True)
    def test_reduces_distributed_peaks_with_max(self, _available, _allocated,
                                                _reserved, tensor, all_reduce):
        peaks = FakePeakTensor([12, 20])
        tensor.return_value = peaks

        def replace_with_global_max(value, op):
            self.assertIs(op, __import__("torch").distributed.ReduceOp.MAX)
            value.values = [18, 24]

        all_reduce.side_effect = replace_with_global_max

        self.assertEqual(_peak_vram_bytes(distributed=True), (18, 24))

    @patch("processor.processor.torch.cuda.is_available", return_value=False)
    def test_omits_peaks_without_cuda(self, _available):
        self.assertIsNone(_peak_vram_bytes())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the missing helpers fail**

Run: `uv run python -m unittest tests.test_peak_vram -v`

Expected: FAIL because `_peak_vram_bytes` and `_reset_peak_vram_stats` cannot be imported.

- [ ] **Step 3: Implement CUDA reset and max-reduction helpers**

Add these functions above `do_train` in `processor/processor.py`:

```python
def _reset_peak_vram_stats():
    if not torch.cuda.is_available():
        return False
    torch.cuda.reset_peak_memory_stats()
    return True


def _peak_vram_bytes(distributed=False):
    if not torch.cuda.is_available():
        return None
    peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(),
         torch.cuda.max_memory_reserved()],
        dtype=torch.int64,
        device="cuda",
    )
    if distributed:
        torch.distributed.all_reduce(peaks, op=torch.distributed.ReduceOp.MAX)
    allocated_bytes, reserved_bytes = peaks.tolist()
    return int(allocated_bytes), int(reserved_bytes)
```

Import `log_peak_vram_metrics` from `utils.wandb_tracking`.

- [ ] **Step 4: Connect measurement to complete epoch boundaries**

Immediately after entering each epoch, before training work, call:

```python
_reset_peak_vram_stats()
```

After the optional validation block has fully completed, but still inside the
epoch loop, call the helper on every rank and log only on rank 0:

```python
peak_vram = _peak_vram_bytes(distributed=args.distributed)
if get_rank() == 0 and peak_vram is not None:
    allocated_bytes, reserved_bytes = peak_vram
    log_peak_vram_metrics(
        wandb_session,
        epoch=epoch,
        allocated_bytes=allocated_bytes,
        reserved_bytes=reserved_bytes,
    )
```

This code must remain after validation and outside the rank-0-only validation
branch so every DDP rank participates in `all_reduce`.

- [ ] **Step 5: Document the new W&B keys**

Add these rows to the logged-every-epoch table in `AGENTS.md`:

```markdown
| `epoch/peak_vram_allocated_gib` | maximum PyTorch tensor allocation during the complete epoch, in GiB; DDP logs the largest rank |
| `epoch/peak_vram_reserved_gib` | maximum PyTorch CUDA allocator reservation during the complete epoch, in GiB; DDP logs the largest rank |
```

- [ ] **Step 6: Run focused and combined tests**

Run: `uv run python -m unittest tests.test_peak_vram tests.test_wandb_tracking -v`

Expected: 7 tests pass.

Run: `uv run python -m compileall processor utils tests`

Expected: command exits 0.

- [ ] **Step 7: Commit Task 2**

```bash
git add AGENTS.md processor/processor.py tests/test_peak_vram.py
git commit -m "feat: measure complete-epoch peak VRAM"
```
