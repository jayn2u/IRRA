from __future__ import annotations

import os
import shutil
import subprocess
import threading
from collections.abc import Callable

MIB_PER_GIB = 1024.0


def query_nvidia_smi(arguments: list[str]) -> list[list[str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [
        [cell.strip() for cell in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def visible_gpu_identifier() -> str:
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return first or "0"


class SystemMemoryPoller:
    def __init__(
        self,
        *,
        pid: int | None = None,
        gpu: str | None = None,
        interval_seconds: float = 0.1,
        query: Callable[[list[str]], list[list[str]]] | None = None,
    ) -> None:
        self.pid = os.getpid() if pid is None else pid
        self.gpu = visible_gpu_identifier() if gpu is None else gpu
        self.interval_seconds = interval_seconds
        self._query = query or query_nvidia_smi
        self.available = query is not None or shutil.which("nvidia-smi") is not None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process_peak_mib: int | None = None
        self._gpu_peak_mib: int | None = None
        self._samples = 0

    def sample(self) -> None:
        if not self.available:
            return
        process_mib = None
        for row in self._query(["--query-compute-apps=pid,used_memory"]):
            if len(row) >= 2 and row[0] == str(self.pid) and row[1].isdigit():
                process_mib = (process_mib or 0) + int(row[1])
        gpu_mib = None
        for row in self._query(["-i", self.gpu, "--query-gpu=memory.used"]):
            if row and row[0].isdigit():
                gpu_mib = int(row[0])
        with self._lock:
            self._samples += 1
            if process_mib is not None:
                self._process_peak_mib = max(self._process_peak_mib or 0, process_mib)
            if gpu_mib is not None:
                self._gpu_peak_mib = max(self._gpu_peak_mib or 0, gpu_mib)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def start(self) -> SystemMemoryPoller:
        self.sample()
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def peaks(self) -> dict[str, float | int | None]:
        with self._lock:
            process_mib = self._process_peak_mib
            gpu_mib = self._gpu_peak_mib
            samples = self._samples
        return {
            "system_process_peak_gib": None
            if process_mib is None
            else process_mib / MIB_PER_GIB,
            "system_gpu_peak_gib": None if gpu_mib is None else gpu_mib / MIB_PER_GIB,
            "system_poll_samples": samples,
        }

    def stop(self) -> dict[str, float | int | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.sample()
        return self.peaks()
