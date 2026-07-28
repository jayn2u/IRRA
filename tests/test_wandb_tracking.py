import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import utils.wandb_tracking as wandb_tracking


class WandbGpuConfigTest(unittest.TestCase):
    def test_get_gpu_config_uses_current_device_and_world_size(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "NVIDIA RTX A6000"

        with (
            patch.object(wandb_tracking, "torch", torch, create=True),
            patch.dict(os.environ, {"WORLD_SIZE": "4"}),
        ):
            config = wandb_tracking.get_gpu_config()

        self.assertEqual(
            config,
            {
                "gpu_name": "NVIDIA RTX A6000",
                "gpu_count": 4,
            },
        )
        torch.cuda.get_device_name.assert_called_once_with()

    def test_get_gpu_config_returns_empty_config_without_cuda(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False

        with patch.object(wandb_tracking, "torch", torch, create=True):
            config = wandb_tracking.get_gpu_config()

        self.assertEqual(config, {})
        torch.cuda.get_device_name.assert_not_called()

    def test_start_train_run_includes_gpu_config(self):
        run = Mock()
        run.id = "test-run"
        run.project = "irra"
        run.entity = "test-entity"
        wandb = Mock()
        wandb.init.return_value = run

        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                wandb=True,
                wandb_env_file="unused",
                wandb_project="irra",
                wandb_entity="test-entity",
                wandb_run_name="future-run",
                wandb_group="",
                wandb_notes="",
                wandb_tags=[],
                output_dir=output_dir,
                dataset_name="CUHK-PEDES",
                loss_names="sdm+mlm+id",
            )
            with (
                patch.object(wandb_tracking, "wandb", wandb),
                patch.object(wandb_tracking, "read_env_value", return_value=None),
                patch.object(
                    wandb_tracking,
                    "get_gpu_config",
                    return_value={
                        "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
                        "gpu_count": 1,
                    },
                ),
            ):
                wandb_tracking.start_train_run(args)

        config = wandb.init.call_args.kwargs["config"]
        self.assertEqual(config["gpu_name"], "NVIDIA GeForce RTX 5070 Ti")
        self.assertEqual(config["gpu_count"], 1)


if __name__ == "__main__":
    unittest.main()
