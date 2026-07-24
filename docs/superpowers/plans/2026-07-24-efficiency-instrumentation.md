# IRRA Efficiency Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add train/validation process-level VRAM and CUDA-synchronized timing metrics to original IRRA training without changing its learning behavior.

**Architecture:** A focused `utils/efficiency.py` module owns measurement primitives. The processor applies separate train and validation measurement boundaries, while `utils/wandb_tracking.py` owns stable metric names and W&B summaries.

**Tech Stack:** Python 3.12, PyTorch 2.9, unittest, Weights & Biases

## Global Constraints

- Execute Python with `uv run python`.
- Preserve the original IRRA model, losses, optimizer, scheduler, data loader, and checkpoint-selection behavior.
- Record PyTorch process peak reserved VRAM as the canonical memory metric.
- Keep train and validation VRAM scopes separate.
- Define training time as the CUDA-synchronized training loop excluding validation and checkpoint I/O.

---

### Task 1: Measurement Primitives and W&B Payloads

**Files:**
- Create: `utils/efficiency.py`
- Create: `tests/test_efficiency_metrics.py`
- Modify: `utils/wandb_tracking.py`

**Interfaces:**
- Produces: `start_cuda_timer(device) -> float`
- Produces: `finish_cuda_timer(device, started_at) -> float`
- Produces: `reset_peak_vram_stats(device) -> None`
- Produces: `get_peak_vram_metrics(device) -> dict[str, float]`
- Produces: extended `log_train_epoch_metrics(..., efficiency_metrics, vram_metrics)`
- Produces: extended `log_val_metrics(..., efficiency_metrics, vram_metrics)`

- [ ] **Step 1: Write failing efficiency and W&B payload tests**

Create tests that require CUDA synchronization, MiB conversion, distinct
`train/*` and `val/*` keys, and disabled-session no-op behavior.

- [ ] **Step 2: Run tests and confirm expected import/signature failures**

Run: `uv run python -m unittest tests.test_efficiency_metrics -v`

Expected: failure because `utils.efficiency` and the extended W&B signatures do
not exist.

- [ ] **Step 3: Implement minimal measurement helpers and payload mapping**

Use `time.perf_counter`, boundary `torch.cuda.synchronize`, and PyTorch
allocated/reserved peak APIs. Define W&B summaries for process peaks.

- [ ] **Step 4: Run the focused tests**

Run: `uv run python -m unittest tests.test_efficiency_metrics -v`

Expected: all focused tests pass.

### Task 2: Processor Integration

**Files:**
- Modify: `processor/processor.py`
- Modify: `tests/test_efficiency_metrics.py`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: Task 1 measurement helpers and extended W&B logging functions.
- Produces: separate train and validation measurement boundaries in `do_train`.

- [ ] **Step 1: Add a failing processor integration test**

Require the processor to reset/read peaks independently around train and
validation and to pass epoch timing, throughput, cumulative GPU hours, and
process peaks to W&B helpers.

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `uv run python -m unittest tests.test_efficiency_metrics -v`

Expected: failure because `do_train` does not provide efficiency metrics.

- [ ] **Step 3: Integrate train and validation measurement scopes**

End the train timer and read train peaks before scheduler, logging, validation,
and checkpoint work. Reset validation peaks immediately before evaluator
execution and read them immediately afterward.

- [ ] **Step 4: Document metric definitions**

Add the stable W&B keys and their inclusion boundaries to `AGENTS.md`.

- [ ] **Step 5: Run focused and repository verification**

Run:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall processor utils train.py
git diff --check
```

Expected: tests pass, compilation succeeds, and `git diff --check` emits no
errors.

- [ ] **Step 6: Commit implementation**

Commit the intended files with a Korean message and the required Codex
co-author trailer.
