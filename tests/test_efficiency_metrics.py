import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils.efficiency import (
    build_epoch_efficiency_metrics,
    finish_cuda_timer,
    get_peak_vram_metrics,
    reset_peak_vram_stats,
    start_cuda_timer,
)
from utils.wandb_tracking import (
    WandbSession,
    log_train_epoch_metrics,
    log_val_metrics,
)
from processor.processor import _evaluate_with_efficiency


class RecordingSession:
    enabled = True

    def __init__(self):
        self.payloads = []

    def log(self, payload, step=None):
        self.payloads.append((payload, step))


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

    @patch(
        "processor.processor.get_peak_vram_metrics",
        return_value={
            "peak_vram_allocated_mb": 7000.0,
            "peak_vram_reserved_mb": 7500.0,
        },
    )
    @patch("processor.processor.finish_cuda_timer", return_value=8.25)
    @patch("processor.processor.start_cuda_timer", return_value=10.0)
    @patch("processor.processor.reset_peak_vram_stats")
    def test_validation_measurement_has_independent_time_and_vram_scope(
        self,
        reset_peak_vram_stats_mock,
        start_cuda_timer_mock,
        finish_cuda_timer_mock,
        get_peak_vram_metrics_mock,
    ):
        evaluator = Mock()
        evaluator.eval.return_value = {"t2i_R1": 73.5}
        model = Mock()
        eval_model = Mock()
        model.eval.return_value = eval_model
        device = SimpleNamespace(type="cuda")

        metrics, efficiency, vram = _evaluate_with_efficiency(
            evaluator=evaluator,
            model=model,
            device=device,
        )

        reset_peak_vram_stats_mock.assert_called_once_with(device)
        start_cuda_timer_mock.assert_called_once_with(device)
        evaluator.eval.assert_called_once_with(
            eval_model,
            i2t_metric=True,
            return_metrics=True,
        )
        finish_cuda_timer_mock.assert_called_once_with(device, 10.0)
        get_peak_vram_metrics_mock.assert_called_once_with(device)
        self.assertEqual(metrics, {"t2i_R1": 73.5})
        self.assertEqual(efficiency, {"epoch_seconds": 8.25})
        self.assertEqual(vram["peak_vram_reserved_mb"], 7500.0)


if __name__ == "__main__":
    unittest.main()
