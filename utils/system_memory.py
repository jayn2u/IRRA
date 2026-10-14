from __future__ import annotations

import atexit
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

MIB_PER_GIB = 1024.0
PROCESS_QUERY = "compute-apps=pid,used_memory"
GPU_QUERY = "gpu=memory.used"


def visible_gpu_identifier() -> str:
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return first or "0"


def nvidia_smi_command(gpu: str, interval_ms: int) -> Callable[[str], list[str]]:
    def build(query: str) -> list[str]:
        return [
            "nvidia-smi",
            "-i",
            gpu,
            f"--query-{query}",
            "--format=csv,noheader,nounits",
            "-lms",
            str(interval_ms),
        ]

    return build


def read_rows(path: Path) -> list[list[str]]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    return [
        [cell.strip() for cell in line.split(",")]
        for line in text.splitlines()
        if line.strip()
    ]


def stop_process(process: subprocess.Popen, timeout_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=timeout_seconds)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
        process.wait(timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired):
        pass


class SystemMemoryPoller:
    def __init__(
        self,
        log_dir: str | os.PathLike[str],
        *,
        name: str = "system_memory",
        pid: int | None = None,
        gpu: str | None = None,
        interval_ms: int = 100,
        stop_timeout_seconds: float = 5.0,
        command: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.pid = os.getpid() if pid is None else pid
        self.gpu = visible_gpu_identifier() if gpu is None else gpu
        self.stop_timeout_seconds = stop_timeout_seconds
        self._command = command or nvidia_smi_command(self.gpu, interval_ms)
        self.available = command is not None or shutil.which("nvidia-smi") is not None
        log_path = Path(log_dir)
        self.process_path = log_path / f"{name}_process.csv"
        self.gpu_path = log_path / f"{name}_gpu.csv"
        self._processes: list[subprocess.Popen] = []

    def _launch(self, query: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as output:
            try:
                process = subprocess.Popen(
                    self._command(query),
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                )
            except OSError:
                return
        self._processes.append(process)

    def start(self) -> SystemMemoryPoller:
        atexit.register(self.stop)
        if self.available:
            self._launch(PROCESS_QUERY, self.process_path)
            self._launch(GPU_QUERY, self.gpu_path)
        return self

    def peaks(self) -> dict[str, float | int | None]:
        process_mib = None
        for row in read_rows(self.process_path):
            if len(row) >= 2 and row[0] == str(self.pid) and row[1].isdigit():
                process_mib = max(process_mib or 0, int(row[1]))
        gpu_mib = None
        samples = 0
        for row in read_rows(self.gpu_path):
            if row and row[0].isdigit():
                samples += 1
                gpu_mib = max(gpu_mib or 0, int(row[0]))
        return {
            "system_process_peak_gib": None
            if process_mib is None
            else process_mib / MIB_PER_GIB,
            "system_gpu_peak_gib": None if gpu_mib is None else gpu_mib / MIB_PER_GIB,
            "system_poll_samples": samples,
        }

    def stop(self) -> dict[str, float | int | None]:
        atexit.unregister(self.stop)
        processes, self._processes = self._processes, []
        for process in processes:
            stop_process(process, self.stop_timeout_seconds)
        return self.peaks()

    def __enter__(self) -> SystemMemoryPoller:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
