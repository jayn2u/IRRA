import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from utils.efficiency import (
    build_epoch_efficiency_metrics,
    finish_cuda_timer,
    format_peak_vram,
    get_global_processed_examples,
    get_peak_vram_metrics,
    reset_peak_vram_stats,
    start_measurement,
    start_cuda_timer,
)
from utils.wandb_tracking import (
    WandbSession,
    log_train_epoch_metrics,
    log_val_metrics,
)
from processor.processor import _evaluate_with_efficiency, do_train


class RecordingSession:
    enabled = True

    def __init__(self):
        self.payloads = []

    def log(self, payload, step=None):
        self.payloads.append((payload, step))


class BatchValue:
    def __init__(self, shape=()):
        self.shape = shape

    def to(self, device):
        return self


class OneEpochModel:
    def train(self):
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        return {
            "sdm_loss": torch.tensor(1.0, requires_grad=True),
            "temperature": torch.tensor(0.02),
        }


class EfficiencyMetricsTest(unittest.TestCase):
    def test_cpu_vram_metrics_are_empty(self):
        device = SimpleNamespace(type="cpu")

        reset_peak_vram_stats(device)

        self.assertEqual(get_peak_vram_metrics(device), {})

    @patch("utils.efficiency.torch.cuda.max_memory_reserved")
    @patch("utils.efficiency.torch.cuda.max_memory_allocated")
    @patch("utils.efficiency.torch.cuda.reset_peak_memory_stats")
    @patch("utils.efficiency.torch.cuda.synchronize")
    def test_cuda_vram_metrics_are_reported_in_mib(
        self,
        synchronize,
        reset_peak_memory_stats,
        max_memory_allocated,
        max_memory_reserved,
    ):
        device = SimpleNamespace(type="cuda")
        max_memory_allocated.return_value = 1536 * 1024 * 1024
        max_memory_reserved.return_value = 2048 * 1024 * 1024

        reset_peak_vram_stats(device)
        metrics = get_peak_vram_metrics(device)

        reset_peak_memory_stats.assert_called_once_with(device)
        synchronize.assert_called_once_with(device)
        max_memory_allocated.assert_called_once_with(device)
        max_memory_reserved.assert_called_once_with(device)
        self.assertEqual(
            metrics,
            {
                "peak_vram_allocated_mb": 1536.0,
                "peak_vram_reserved_mb": 2048.0,
            },
        )

    @patch("utils.efficiency.time.perf_counter", side_effect=[10.0, 22.5])
    @patch("utils.efficiency.torch.cuda.synchronize")
    def test_cuda_timer_synchronizes_both_boundaries(
        self,
        synchronize,
        perf_counter,
    ):
        device = SimpleNamespace(type="cuda")

        started_at = start_cuda_timer(device)
        elapsed = finish_cuda_timer(device, started_at)

        self.assertEqual(elapsed, 12.5)
        self.assertEqual(synchronize.call_args_list, [unittest.mock.call(device)] * 2)
        self.assertEqual(perf_counter.call_count, 2)

    @patch("utils.efficiency.time.perf_counter", return_value=10.0)
    @patch("utils.efficiency.torch.cuda.reset_peak_memory_stats")
    @patch("utils.efficiency.torch.cuda.synchronize")
    def test_measurement_start_synchronizes_before_resetting_peak(
        self,
        synchronize,
        reset_peak_memory_stats,
        perf_counter,
    ):
        device = SimpleNamespace(type="cuda")
        calls = []
        synchronize.side_effect = lambda _: calls.append("synchronize")
        reset_peak_memory_stats.side_effect = lambda _: calls.append("reset")
        perf_counter.side_effect = lambda: calls.append("timer") or 10.0

        started_at = start_measurement(device)

        self.assertEqual(started_at, 10.0)
        self.assertEqual(calls, ["synchronize", "reset", "timer"])

    def test_global_processed_examples_returns_local_count_without_ddp(self):
        device = SimpleNamespace(type="cuda")

        with patch(
            "utils.efficiency.torch.distributed.is_available",
            return_value=False,
        ):
            count = get_global_processed_examples(64, device)

        self.assertEqual(count, 64)

    def test_global_processed_examples_sums_all_ddp_ranks(self):
        device = SimpleNamespace(type="cuda")
        count_tensor = Mock()
        count_tensor.item.return_value = 128

        with (
            patch(
                "utils.efficiency.torch.distributed.is_available",
                return_value=True,
            ),
            patch(
                "utils.efficiency.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "utils.efficiency.torch.tensor",
                return_value=count_tensor,
            ) as tensor,
            patch(
                "utils.efficiency.torch.distributed.all_reduce",
            ) as all_reduce,
        ):
            count = get_global_processed_examples(64, device)

        tensor.assert_called_once_with(64, dtype=torch.long, device=device)
        all_reduce.assert_called_once_with(
            count_tensor,
            op=torch.distributed.ReduceOp.SUM,
        )
        self.assertEqual(count, 128)

    def test_peak_vram_formatting_is_safe_when_metrics_are_unavailable(self):
        self.assertEqual(format_peak_vram({}), "")
        self.assertEqual(
            format_peak_vram(
                {
                    "peak_vram_allocated_mb": 7000.0,
                    "peak_vram_reserved_mb": 7500.0,
                }
            ),
            " Peak allocated: 7000.00[MiB] Peak reserved: 7500.00[MiB]",
        )

    def test_epoch_efficiency_metrics_include_throughput_and_cumulative_hours(self):
        metrics = build_epoch_efficiency_metrics(
            epoch_seconds=12.5,
            processed_examples=4000,
            cumulative_seconds=25.0,
        )

        self.assertEqual(metrics["epoch_seconds"], 12.5)
        self.assertEqual(metrics["examples_per_second"], 320.0)
        self.assertEqual(metrics["cumulative_gpu_hours"], 25.0 / 3600.0)

    def test_train_wandb_payload_includes_efficiency_and_vram(self):
        session = RecordingSession()
        meter = SimpleNamespace(avg=1.25)

        log_train_epoch_metrics(
            session,
            epoch=3,
            meters={"loss": meter},
            lr=1e-5,
            temperature=0.02,
            efficiency_metrics={
                "epoch_seconds": 12.5,
                "examples_per_second": 320.0,
                "cumulative_gpu_hours": 0.01,
            },
            vram_metrics={
                "peak_vram_allocated_mb": 9000.0,
                "peak_vram_reserved_mb": 9500.0,
            },
        )

        payload, step = session.payloads[0]
        self.assertIsNone(step)
        self.assertEqual(payload["train/epoch_seconds"], 12.5)
        self.assertEqual(payload["train/examples_per_second"], 320.0)
        self.assertEqual(payload["train/cumulative_gpu_hours"], 0.01)
        self.assertEqual(payload["train/peak_vram_allocated_mb"], 9000.0)
        self.assertEqual(payload["train/peak_vram_reserved_mb"], 9500.0)

    def test_validation_wandb_payload_includes_time_and_vram(self):
        session = RecordingSession()

        log_val_metrics(
            session,
            epoch=4,
            metrics={"t2i_R1": 73.5},
            efficiency_metrics={"epoch_seconds": 8.25},
            vram_metrics={
                "peak_vram_allocated_mb": 7000.0,
                "peak_vram_reserved_mb": 7500.0,
            },
        )

        payload, step = session.payloads[0]
        self.assertIsNone(step)
        self.assertEqual(payload["val/epoch_seconds"], 8.25)
        self.assertEqual(payload["val/peak_vram_allocated_mb"], 7000.0)
        self.assertEqual(payload["val/peak_vram_reserved_mb"], 7500.0)

    def test_disabled_wandb_session_accepts_efficiency_metrics(self):
        session = WandbSession(None)

        log_train_epoch_metrics(
            session,
            epoch=1,
            meters={},
            lr=1e-5,
            efficiency_metrics={"epoch_seconds": 1.0},
            vram_metrics={"peak_vram_reserved_mb": 1.0},
        )
        log_val_metrics(
            session,
            epoch=1,
            metrics={},
            efficiency_metrics={"epoch_seconds": 1.0},
            vram_metrics={"peak_vram_reserved_mb": 1.0},
        )

    def test_validation_measurement_has_independent_time_and_vram_scope(
        self,
    ):
        calls = []
        evaluator = Mock()
        evaluator.eval.side_effect = (
            lambda *args, **kwargs:
            calls.append("evaluate") or {"t2i_R1": 73.5}
        )
        model = Mock()
        eval_model = Mock()
        model.eval.return_value = eval_model
        device = SimpleNamespace(type="cuda")

        with (
            patch(
                "processor.processor.start_measurement",
                side_effect=lambda _: calls.append("start") or 10.0,
            ) as start_measurement_mock,
            patch(
                "processor.processor.finish_cuda_timer",
                side_effect=lambda *_: calls.append("finish") or 8.25,
            ) as finish_cuda_timer_mock,
            patch(
                "processor.processor.get_peak_vram_metrics",
                side_effect=lambda _: calls.append("read") or {
                    "peak_vram_allocated_mb": 7000.0,
                    "peak_vram_reserved_mb": 7500.0,
                },
            ) as get_peak_vram_metrics_mock,
        ):
            metrics, efficiency, vram = _evaluate_with_efficiency(
                evaluator=evaluator,
                model=model,
                device=device,
            )

        start_measurement_mock.assert_called_once_with(device)
        evaluator.eval.assert_called_once_with(
            eval_model,
            i2t_metric=True,
            return_metrics=True,
        )
        finish_cuda_timer_mock.assert_called_once_with(device, 10.0)
        get_peak_vram_metrics_mock.assert_called_once_with(device)
        self.assertEqual(calls, ["start", "evaluate", "finish", "read"])
        self.assertEqual(metrics, {"t2i_R1": 73.5})
        self.assertEqual(efficiency, {"epoch_seconds": 8.25})
        self.assertEqual(vram["peak_vram_reserved_mb"], 7500.0)

    def test_one_epoch_training_logs_measured_efficiency(self):
        train_loader = [
            {
                "images": BatchValue(shape=(2, 3, 4, 4)),
                "caption_ids": BatchValue(shape=(2, 8)),
            }
        ]
        evaluator = Mock()
        optimizer = Mock()
        scheduler = Mock()
        scheduler.get_lr.return_value = [1e-5]
        checkpointer = Mock()
        train_log = Mock()
        val_log = Mock()
        args = SimpleNamespace(
            log_period=100,
            eval_period=1,
            num_epoch=1,
            output_dir=tempfile.gettempdir(),
            distributed=False,
        )

        with (
            patch("processor.processor.SummaryWriter"),
            patch("processor.processor.synchronize"),
            patch("processor.processor.get_rank", return_value=0),
            patch("processor.processor.get_world_size", return_value=1),
            patch(
                "processor.processor.start_measurement",
                return_value=10.0,
            ),
            patch(
                "processor.processor.finish_cuda_timer",
                return_value=5.0,
            ),
            patch(
                "processor.processor.get_peak_vram_metrics",
                return_value={
                    "peak_vram_allocated_mb": 7000.0,
                    "peak_vram_reserved_mb": 7500.0,
                },
            ),
            patch(
                "processor.processor.get_global_processed_examples",
                return_value=2,
            ) as global_examples,
            patch(
                "processor.processor._evaluate_with_efficiency",
                return_value=(
                    {"t2i_R1": 73.5},
                    {"epoch_seconds": 2.0},
                    {
                        "peak_vram_allocated_mb": 6000.0,
                        "peak_vram_reserved_mb": 6500.0,
                    },
                ),
            ),
            patch("processor.processor.log_train_epoch_metrics", train_log),
            patch("processor.processor.log_val_metrics", val_log),
            patch("processor.processor.torch.cuda.empty_cache"),
        ):
            best_top1, best_epoch = do_train(
                start_epoch=1,
                args=args,
                model=OneEpochModel(),
                train_loader=train_loader,
                evaluator=evaluator,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpointer=checkpointer,
                wandb_session=RecordingSession(),
            )

        global_examples.assert_called_once()
        train_efficiency = train_log.call_args.kwargs["efficiency_metrics"]
        self.assertEqual(train_efficiency["epoch_seconds"], 5.0)
        self.assertEqual(train_efficiency["examples_per_second"], 0.4)
        self.assertEqual(train_efficiency["cumulative_gpu_hours"], 5.0 / 3600.0)
        val_log.assert_called_once()
        checkpointer.save.assert_called_once_with("best", num_epoch=1, iteration=0, epoch=1)
        self.assertEqual((best_top1, best_epoch), (73.5, 1))


if __name__ == "__main__":
    unittest.main()
