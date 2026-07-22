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
