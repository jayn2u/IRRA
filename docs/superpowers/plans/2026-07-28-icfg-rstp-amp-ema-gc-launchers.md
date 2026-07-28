# ICFG-PEDES and RSTPReid AMP/EMA/GC Launchers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add directly runnable ICFG-PEDES and RSTPReid launchers that enable AMP, gradient checkpointing, and EMA with the established IRRA control-run settings.

**Architecture:** Keep one launcher per dataset and preserve the current shell-script interface. Verify each launcher by substituting a temporary fake `uv` executable and asserting the command and environment it receives, so tests exercise shell behavior without launching training.

**Tech Stack:** Bash, Python 3 `unittest`, `uv`, GitHub

## Global Constraints

- Create `run_icfg_amp_ema_gc.sh` and `run_rstpreid_amp_ema_gc.sh`.
- Use `--amp`, `--gradient_checkpointing`, `--ema`, and `--ema_decay 0.999`.
- Preserve the ordinary `run_icfg.sh` and `run_rstpreid.sh` launchers unchanged.
- Invoke training through `uv run python train.py`.
- Track `.codex/config.toml` with `sandbox_mode = "danger-full-access"` unchanged.
- Work on `codex/icfg-rstp-amp-ema-gc-scripts` and target `main` with a draft pull request.

---

### Task 1: Launcher Behavior Test

**Files:**
- Create: `tests/test_amp_ema_gc_scripts.py`
- Test: `tests/test_amp_ema_gc_scripts.py`

**Interfaces:**
- Consumes: Bash launchers invoked by repository-relative path.
- Produces: A regression test that captures the argument vector and CUDA environment passed to `uv`.

- [ ] **Step 1: Write the failing test**

```python
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_launcher(script_name):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        args_path = temp_path / "args"
        cuda_path = temp_path / "cuda"
        fake_uv = temp_path / "uv"
        fake_uv.write_text(
            "#!/bin/sh\n"
            'printf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
            'printf "%s\\n" "$CUDA_VISIBLE_DEVICES" > "$CAPTURE_CUDA"\n',
            encoding="utf-8",
        )
        fake_uv.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{temp_path}{os.pathsep}{env['PATH']}"
        env["CAPTURE_ARGS"] = str(args_path)
        env["CAPTURE_CUDA"] = str(cuda_path)
        subprocess.run(
            ["bash", str(REPO_ROOT / script_name)],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )

        return args_path.read_text(encoding="utf-8").splitlines(), cuda_path.read_text(
            encoding="utf-8"
        ).strip()


class AmpEmaGcLauncherTest(unittest.TestCase):
    def test_dataset_launchers_pass_expected_training_configuration(self):
        common_tail = [
            "--loss_names",
            "sdm+mlm+id",
            "--num_epoch",
            "60",
            "--amp",
            "--gradient_checkpointing",
            "--ema",
            "--ema_decay",
            "0.999",
            "--wandb",
        ]
        cases = [
            (
                "run_icfg_amp_ema_gc.sh",
                "icfg_amp_ema_gc",
                "ICFG-PEDES",
            ),
            (
                "run_rstpreid_amp_ema_gc.sh",
                "rstpreid_amp_ema_gc",
                "RSTPReid",
            ),
        ]

        for script_name, run_name, dataset_name in cases:
            with self.subTest(script=script_name):
                args, cuda_visible_devices = run_launcher(script_name)
                self.assertEqual(
                    args,
                    [
                        "run",
                        "python",
                        "train.py",
                        "--name",
                        run_name,
                        "--img_aug",
                        "--batch_size",
                        "64",
                        "--MLM",
                        "--dataset_name",
                        dataset_name,
                        *common_tail,
                    ],
                )
                self.assertEqual(cuda_visible_devices, "0")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_amp_ema_gc_scripts -v
```

Expected: `ERROR` for both subtests because
`run_icfg_amp_ema_gc.sh` and `run_rstpreid_amp_ema_gc.sh` do not exist.

### Task 2: Dataset Launchers

**Files:**
- Create: `run_icfg_amp_ema_gc.sh`
- Create: `run_rstpreid_amp_ema_gc.sh`
- Test: `tests/test_amp_ema_gc_scripts.py`

**Interfaces:**
- Consumes: `train.py` command-line flags already implemented in the repository.
- Produces: Two Bash entry points that invoke the configured dataset training command.

- [ ] **Step 1: Implement the ICFG-PEDES launcher**

```bash
#!/bin/bash
# AMP + gradient checkpointing + EMA control run for ICFG-PEDES.
DATASET_NAME="ICFG-PEDES"

CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name icfg_amp_ema_gc \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--amp \
--gradient_checkpointing \
--ema \
--ema_decay 0.999 \
--wandb
```

- [ ] **Step 2: Implement the RSTPReid launcher**

```bash
#!/bin/bash
# AMP + gradient checkpointing + EMA control run for RSTPReid.
DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
uv run python train.py \
--name rstpreid_amp_ema_gc \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--amp \
--gradient_checkpointing \
--ema \
--ema_decay 0.999 \
--wandb
```

- [ ] **Step 3: Make both launchers executable**

Run:

```bash
chmod +x run_icfg_amp_ema_gc.sh run_rstpreid_amp_ema_gc.sh
```

- [ ] **Step 4: Run the launcher test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_amp_ema_gc_scripts -v
```

Expected: `1` test passes with both dataset subtests successful.

- [ ] **Step 5: Check Bash syntax**

Run:

```bash
bash -n run_icfg_amp_ema_gc.sh run_rstpreid_amp_ema_gc.sh
```

Expected: exit status `0` and no output.

### Task 3: Repository Validation and Publication

**Files:**
- Track: `.codex/config.toml`
- Track: `docs/superpowers/specs/2026-07-28-icfg-rstp-amp-ema-gc-launchers-design.md`
- Track: `docs/superpowers/plans/2026-07-28-icfg-rstp-amp-ema-gc-launchers.md`
- Track: `tests/test_amp_ema_gc_scripts.py`
- Track: `run_icfg_amp_ema_gc.sh`
- Track: `run_rstpreid_amp_ema_gc.sh`

**Interfaces:**
- Consumes: The completed launcher implementation and unit tests.
- Produces: A reviewed commit and draft pull request against `main`.

- [ ] **Step 1: Run the complete Python unit test suite**

Run:

```bash
uv run python -m unittest discover -s tests -v
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 2: Confirm the scoped diff**

Run:

```bash
git status -sb
git diff --check
git diff --stat
```

Expected: only the six listed paths are new and `git diff --check` exits
with status `0`.

- [ ] **Step 3: Commit the scoped files**

```bash
git add .codex/config.toml \
  docs/superpowers/specs/2026-07-28-icfg-rstp-amp-ema-gc-launchers-design.md \
  docs/superpowers/plans/2026-07-28-icfg-rstp-amp-ema-gc-launchers.md \
  tests/test_amp_ema_gc_scripts.py \
  run_icfg_amp_ema_gc.sh \
  run_rstpreid_amp_ema_gc.sh
git commit -m "Add AMP EMA GC dataset launchers" \
  -m "Co-authored-by: Codex <codex@openai.com>"
```

- [ ] **Step 4: Push and create a draft pull request**

Run:

```bash
git push -u origin codex/icfg-rstp-amp-ema-gc-scripts
```

Create a draft pull request targeting `main` that explains the new dataset
launchers, preserved baseline behavior, tracked Codex configuration, and
validation commands.
