import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from utils.system_memory import SystemMemoryPoller, nvidia_smi_command


def emitter(process_rows, gpu_rows, ignore_term=False):
    def build(query):
        rows = process_rows if query.startswith("compute-apps") else gpu_rows
        script = (
            ("import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
            + "import sys, time\n"
            + f"sys.stdout.write({''.join(row + chr(10) for row in rows)!r})\n"
            + "sys.stdout.flush()\n"
            + "time.sleep(600)\n"
        )
        return [sys.executable, "-c", script]

    return build


def wait_for_rows(path, count, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and len(path.read_text().splitlines()) >= count:
            return
        time.sleep(0.02)
    raise AssertionError(f"{path} did not receive {count} rows")


class SystemMemoryPollerTest(unittest.TestCase):
    def test_tracks_process_and_gpu_peaks_from_files(self):
        with tempfile.TemporaryDirectory() as out:
            poller = SystemMemoryPoller(
                out,
                pid=42,
                gpu="0",
                command=emitter(["42, 1024", "7, 99999", "42, 3072", "42, 2048"], ["8192", "16384", "12288"]),
            ).start()
            wait_for_rows(poller.process_path, 4)
            wait_for_rows(poller.gpu_path, 3)
            self.assertEqual(poller.peaks()["system_process_peak_gib"], 3.0)
            processes = list(poller._processes)
            peaks = poller.stop()
            self.assertTrue(all(process.poll() is not None for process in processes))
            self.assertEqual(Path(out) / "system_memory_process.csv", poller.process_path)
        self.assertEqual(
            peaks,
            {"system_process_peak_gib": 3.0, "system_gpu_peak_gib": 16.0, "system_poll_samples": 3},
        )

    def test_handles_missing_process(self):
        with tempfile.TemporaryDirectory() as out:
            poller = SystemMemoryPoller(out, pid=1, gpu="0", command=emitter([], [])).start()
            self.assertIsNone(poller.stop()["system_process_peak_gib"])

    def test_stop_kills_process_ignoring_sigterm_within_bound(self):
        with tempfile.TemporaryDirectory() as out:
            poller = SystemMemoryPoller(
                out,
                pid=42,
                gpu="0",
                stop_timeout_seconds=0.5,
                command=emitter(["42, 512"], ["1024"], ignore_term=True),
            ).start()
            wait_for_rows(poller.process_path, 1)
            wait_for_rows(poller.gpu_path, 1)
            processes = list(poller._processes)
            started = time.monotonic()
            peaks = poller.stop()
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertEqual(peaks["system_process_peak_gib"], 0.5)

    def test_child_runs_in_new_session_without_inherited_pipes(self):
        with tempfile.TemporaryDirectory() as out:
            poller = SystemMemoryPoller(out, pid=42, gpu="0", command=emitter(["42, 1"], ["1"])).start()
            wait_for_rows(poller.process_path, 1)
            process = poller._processes[0]
            try:
                self.assertEqual(os.getsid(process.pid), process.pid)
                self.assertIsNone(process.stdout)
            finally:
                poller.stop()

    def test_missing_binary_does_not_raise(self):
        with tempfile.TemporaryDirectory() as out:
            poller = SystemMemoryPoller(
                out, pid=42, gpu="0", command=lambda query: ["/nonexistent/nvidia-smi"]
            ).start()
            self.assertEqual(poller.stop()["system_poll_samples"], 0)

    def test_nvidia_smi_command_streams_queries(self):
        self.assertEqual(
            nvidia_smi_command("1", 100)("gpu=memory.used"),
            ["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
        )


if __name__ == "__main__":
    unittest.main()
