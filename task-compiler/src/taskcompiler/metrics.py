"""Record run metrics, resource use, token counts, and logs."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from dataclasses import dataclass
from pathlib import Path

import psutil

try:
    import torch
except ImportError:
    torch = None


@dataclass
class StageTiming:
    stage: str
    start: float
    end: float | None = None


class RunRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.log_path = run_dir / "log.txt"
        self.summary_path = run_dir / "summary.json"
        self._process = psutil.Process()
        self._teacher_tokens = {"input": 0, "output": 0}
        self._student_tokens = {"input": 0, "output": 0}
        self._start_time = time.monotonic()
        self._task_metrics: dict = {}
        self._model_identities: dict[str, dict] = {}
        self._terminal_available = True
        self._source_snapshot = _source_snapshot()

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)
        if self._terminal_available:
            encoding = getattr(sys.stdout, "encoding", None)
            terminal_line = (
                line.encode(encoding, errors="backslashreplace").decode(encoding)
                if encoding
                else line
            )
            try:
                print(terminal_line, end="", flush=True)
            except OSError:
                self._terminal_available = False

    def stage(self, name: str):
        return _StageContext(self, name)

    def _write_stage_metrics(self, name: str, elapsed: float) -> None:
        gpu_util, gpu_mem_mb = self._read_gpu()
        record = {
            "stage": name,
            "elapsed_seconds": elapsed,
            "cpu_percent": self._process.cpu_percent(interval=None),
            "ram_mb": self._process.memory_info().rss / (1024 * 1024),
            "gpu_util_percent": gpu_util,
            "gpu_mem_mb": gpu_mem_mb,
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _read_gpu(self) -> tuple[float | None, float | None]:
        if torch is None or not torch.cuda.is_available():
            return None, None
        mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        return None, mem_mb

    def add_teacher_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._teacher_tokens["input"] += input_tokens
        self._teacher_tokens["output"] += output_tokens

    def add_student_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._student_tokens["input"] += input_tokens
        self._student_tokens["output"] += output_tokens

    def set_task_metrics(self, metrics: dict) -> None:
        self._task_metrics = metrics

    def set_model_identity(self, role: str, identity: dict) -> None:
        self._model_identities[role] = identity

    def finalize(self) -> None:
        total_elapsed = time.monotonic() - self._start_time
        summary = {
            "total_wall_seconds": total_elapsed,
            "teacher_tokens": self._teacher_tokens,
            "student_tokens": self._student_tokens,
            "task_metrics": self._task_metrics,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "models": self._model_identities,
                "packages": {
                    name: _package_version(name)
                    for name in (
                        "anthropic",
                        "datasets",
                        "gepa",
                        "sentence-transformers",
                        "torch",
                        "transformers",
                    )
                },
                "source": self._source_snapshot,
            },
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


class _StageContext:
    def __init__(self, recorder: RunRecorder, name: str):
        self.recorder = recorder
        self.name = name
        self._start = None

    def __enter__(self):
        self._start = time.monotonic()
        self.recorder.log(f"stage started: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.monotonic() - self._start
        self.recorder._write_stage_metrics(self.name, elapsed)
        if exc_type is None:
            self.recorder.log(f"stage finished: {self.name} ({elapsed:.2f}s)")
        else:
            self.recorder.log(f"stage failed: {self.name}: {exc_type.__name__}: {exc_val}")
        return False


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _source_snapshot() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    paths = []
    for folder in ("src", "scripts", "configs", "frozen", "gui"):
        root = project_root / folder
        if root.exists():
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".yaml", ".yml", ".json"}
            )
    paths.extend(
        path
        for path in (project_root / "pyproject.toml",)
        if path.exists()
    )
    digest = hashlib.sha256()
    relative_paths = []
    file_hashes = []
    for path in sorted(paths):
        relative = path.relative_to(project_root).as_posix()
        relative_paths.append(relative)
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        file_hashes.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "snapshot_sha256": digest.hexdigest(),
        "file_count": len(relative_paths),
        "files": file_hashes,
    }
