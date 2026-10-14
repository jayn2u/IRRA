import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import processor.processor as processor


class StubPoller:
    log_dirs = []
    stops = 0

    def __init__(self, log_dir, **kwargs):
        StubPoller.log_dirs.append(str(log_dir))

    def start(self):
        return self

    def peaks(self):
        return {"system_process_peak_gib": 5.0, "system_gpu_peak_gib": 8.0, "system_poll_samples": 3}

    def stop(self):
        StubPoller.stops += 1
        return self.peaks()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.stop()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, batch):
        return {"sdm_loss": self.linear(batch["images"]).pow(2).mean()}


class RecordingSession:
    def __init__(self):
        self.logged = []
        self.summary = {}

    def log(self, metrics, step=None):
        self.logged.append(metrics)

    def set_summary(self, metrics):
        self.summary.update(metrics)

    def save(self, path, base_path=None):
        pass


class MeasureVramTest(unittest.TestCase):
    def test_runs_exact_steps_and_writes_report(self):
        model = TinyModel()
        loader = [{"images": torch.ones(2, 2)} for _ in range(3)]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        calls = []
        cpu = torch.device("cpu")
        StubPoller.log_dirs.clear()
        with tempfile.TemporaryDirectory() as out, \
                patch.object(processor.torch, "device", lambda name: cpu), \
                patch.object(processor, "SystemMemoryPoller", StubPoller), \
                patch.object(torch.cuda, "synchronize", lambda device=None: None), \
                patch.object(torch.cuda, "reset_peak_memory_stats", lambda device=None: calls.append("reset")), \
                patch.object(torch.cuda, "max_memory_allocated", lambda device=None: 1024 ** 3), \
                patch.object(torch.cuda, "max_memory_reserved", lambda device=None: 2 * 1024 ** 3):
            args = SimpleNamespace(
                measure_vram_steps=150, amp=False, ema=True, ema_decay=0.999,
                distributed=False, log_period=100, name="irra", batch_size=2,
                gradient_checkpointing=False, output_dir=out,
            )
            session = RecordingSession()
            report = processor.do_measure_vram(args, model, loader, optimizer, session)
            with open(f"{out}/vram_measure.json") as f:
                saved = json.load(f)
        self.assertEqual(calls, ["reset"])
        self.assertEqual(StubPoller.log_dirs, [out])
        self.assertEqual(report["steps"], 150)
        self.assertEqual(sorted(saved["checkpoints"]), ["100", "150"])
        self.assertEqual(saved["system_process_peak_gib"], 5.0)
        self.assertEqual(saved["system_gpu_peak_gib"], 8.0)
        self.assertEqual(saved["max_memory_allocated_gib"], 1.0)
        self.assertEqual(saved["max_memory_reserved_gib"], 2.0)
        self.assertEqual(session.summary["vram_measure/steps"], 150)


    def test_exception_inside_measure_stops_poller(self):
        class FailingModel(TinyModel):
            def forward(self, batch):
                raise RuntimeError("oom")

        model = FailingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        cpu = torch.device("cpu")
        StubPoller.stops = 0
        with tempfile.TemporaryDirectory() as out, \
                patch.object(processor.torch, "device", lambda name: cpu), \
                patch.object(processor, "SystemMemoryPoller", StubPoller), \
                patch.object(torch.cuda, "synchronize", lambda device=None: None), \
                patch.object(torch.cuda, "reset_peak_memory_stats", lambda device=None: None):
            args = SimpleNamespace(
                measure_vram_steps=5, amp=False, ema=False, ema_decay=0.999,
                distributed=False, log_period=100, name="irra", batch_size=2,
                gradient_checkpointing=False, output_dir=out,
            )
            with self.assertRaises(RuntimeError):
                processor.do_measure_vram(args, model, [{"images": torch.ones(2, 2)}], optimizer, RecordingSession())
        self.assertEqual(StubPoller.stops, 1)


if __name__ == "__main__":
    unittest.main()
