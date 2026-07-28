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
