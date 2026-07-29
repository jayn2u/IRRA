import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

import model.build as model_build
from model.clip_model import Transformer
from processor.processor import do_train
from utils.ema import ModelEMA


class ModelEMATest(unittest.TestCase):
    def test_update_applies_exponential_decay(self):
        model = nn.Linear(4, 4)
        ema = ModelEMA(model, decay=0.9)
        before = ema.module.weight.clone()

        with torch.no_grad():
            model.weight.add_(1.0)
        ema.update(model)

        expected = before * 0.9 + (before + 1.0) * 0.1
        self.assertTrue(torch.allclose(ema.module.weight, expected, atol=1e-6))

    def test_shadow_is_frozen_and_decoupled_from_live_model(self):
        model = nn.Linear(4, 4)
        ema = ModelEMA(model, decay=0.9)

        self.assertFalse(any(p.requires_grad for p in ema.module.parameters()))

        ema.update(model)
        snapshot = ema.module.weight.clone()
        with torch.no_grad():
            model.weight.add_(100.0)

        self.assertTrue(torch.equal(ema.module.weight, snapshot))


class GradientCheckpointingTransformerTest(unittest.TestCase):
    def test_matches_non_checkpointed_forward_and_backward(self):
        torch.manual_seed(0)
        plain = Transformer(width=8, layers=2, heads=2, use_grad_checkpointing=False)
        checkpointed = Transformer(width=8, layers=2, heads=2, use_grad_checkpointing=True)
        checkpointed.load_state_dict(plain.state_dict())
        plain.train()
        checkpointed.train()

        x_plain = torch.randn(5, 3, 8, requires_grad=True)
        x_ckpt = x_plain.detach().clone().requires_grad_(True)

        out_plain = plain(x_plain)
        out_ckpt = checkpointed(x_ckpt)
        self.assertTrue(torch.allclose(out_plain, out_ckpt, atol=1e-5))

        out_plain.sum().backward()
        out_ckpt.sum().backward()
        self.assertTrue(torch.allclose(x_plain.grad, x_ckpt.grad, atol=1e-5))

    def test_disabled_by_default_and_skipped_in_eval_mode(self):
        transformer = Transformer(width=8, layers=1, heads=2)
        self.assertFalse(transformer.use_grad_checkpointing)

        checkpointed = Transformer(width=8, layers=1, heads=2, use_grad_checkpointing=True)
        checkpointed.eval()
        # checkpoint.checkpoint requires grad-enabled tensors; eval-mode/no-grad
        # inference must take the plain nn.Sequential path instead.
        with torch.no_grad():
            checkpointed(torch.randn(4, 2, 8))


class BuildModelAmpWeightDtypeTest(unittest.TestCase):
    """GradScaler requires fp32 master weights/grads. convert_weights()
    permanently casts weights to fp16, so it must be skipped under --amp or
    scaler.step() raises "Attempting to unscale FP16 gradients." on the very
    first optimizer step.
    """

    def _build(self, amp):
        args = SimpleNamespace(
            loss_names="sdm",
            pretrain_choice="ViT-B/16",
            img_size=(384, 128),
            stride_size=16,
            gradient_checkpointing=False,
            temperature=0.02,
            amp=amp,
        )
        fake_base_model = nn.Linear(4, 4)
        with (
            patch.object(
                model_build,
                "build_CLIP_from_openai_pretrained",
                return_value=(fake_base_model, {"embed_dim": 4}),
            ),
            patch.object(
                model_build, "convert_weights", wraps=model_build.convert_weights
            ) as convert_mock,
        ):
            built = model_build.build_model(args, num_classes=10)
        return built, convert_mock

    def test_amp_enabled_skips_fp16_conversion(self):
        built, convert_mock = self._build(amp=True)
        convert_mock.assert_not_called()
        self.assertTrue(all(p.dtype == torch.float32 for p in built.parameters()))

    def test_amp_disabled_preserves_fp16_conversion(self):
        built, convert_mock = self._build(amp=False)
        convert_mock.assert_called_once_with(built)
        self.assertTrue(any(p.dtype == torch.float16 for p in built.parameters()))


class TinyReIDModel(nn.Module):
    """Minimal stand-in for IRRA: real nn.Module so autocast/GradScaler/EMA all engage."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, batch):
        out = self.fc(batch["images"])
        return {
            "sdm_loss": out.sum(),
            "temperature": torch.tensor(0.02),
        }


class BatchValue:
    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tensor.shape

    def to(self, device):
        return self.tensor


def _run_do_train(args_overrides):
    train_loader = [
        {
            "images": BatchValue(torch.randn(2, 4)),
            "caption_ids": BatchValue(torch.zeros(2, 8)),
        }
    ]
    evaluator = Mock()
    optimizer = torch.optim.SGD(TinyReIDModel().parameters(), lr=0.1)
    scheduler = Mock()
    scheduler.get_lr.return_value = [1e-5]
    checkpointer = Mock()
    model = TinyReIDModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    args_kwargs = dict(
        log_period=100,
        eval_period=1,
        num_epoch=1,
        output_dir=tempfile.mkdtemp(),
        distributed=False,
        amp=False,
        ema=False,
        ema_decay=0.9,
    )
    args_kwargs.update(args_overrides)
    args = SimpleNamespace(**args_kwargs)

    eval_model_holder = {}

    def fake_evaluate(evaluator, model, device):
        eval_model_holder["model"] = model
        return {"t2i_R1": 73.5}, {"epoch_seconds": 2.0}, {
            "peak_vram_allocated_mb": 1.0,
            "peak_vram_reserved_mb": 1.0,
        }

    with (
        patch("processor.processor.SummaryWriter"),
        patch("processor.processor.synchronize"),
        patch("processor.processor.get_rank", return_value=0),
        patch("processor.processor.get_world_size", return_value=1),
        patch("processor.processor.start_measurement", return_value=10.0),
        patch("processor.processor.finish_cuda_timer", return_value=5.0),
        patch(
            "processor.processor.get_peak_vram_metrics",
            return_value={"peak_vram_allocated_mb": 1.0, "peak_vram_reserved_mb": 1.0},
        ),
        patch("processor.processor.get_global_processed_examples", return_value=2),
        patch("processor.processor._evaluate_with_efficiency", side_effect=fake_evaluate),
        patch("processor.processor.log_train_epoch_metrics"),
        patch("processor.processor.log_val_metrics"),
        patch("processor.processor.torch.cuda.empty_cache"),
    ):
        do_train(
            start_epoch=1,
            args=args,
            model=model,
            train_loader=train_loader,
            evaluator=evaluator,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpointer=checkpointer,
        )

    return model, args, eval_model_holder


class DoTrainEmaWiringTest(unittest.TestCase):
    def test_without_ema_evaluates_the_live_model(self):
        model, args, eval_model_holder = _run_do_train({})
        self.assertIs(eval_model_holder["model"], model)
        self.assertFalse(os.path.exists(os.path.join(args.output_dir, "best_ema.pth")))

    def test_with_ema_evaluates_shadow_and_saves_best_ema_checkpoint(self):
        model, args, eval_model_holder = _run_do_train({"ema": True})
        self.assertIsNot(eval_model_holder["model"], model)
        ema_ckpt_path = os.path.join(args.output_dir, "best_ema.pth")
        self.assertTrue(os.path.exists(ema_ckpt_path))
        saved = torch.load(ema_ckpt_path, map_location="cpu")
        self.assertIn("model", saved)
        self.assertEqual(saved["epoch"], 1)

    def test_amp_flag_runs_without_error_on_cpu_fallback(self):
        # enabled=True with no CUDA device just warns and behaves like AMP off.
        model, args, eval_model_holder = _run_do_train({"amp": True})
        self.assertIs(eval_model_holder["model"], model)


if __name__ == "__main__":
    unittest.main()
