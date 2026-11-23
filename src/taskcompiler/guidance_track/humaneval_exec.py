"""Run HumanEval candidates in an isolated Docker container."""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_IMAGE = "taskcompiler-humaneval:latest"
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024


@dataclass
class ExecutionResult:
    passed: bool
    error: str | None
    timed_out: bool = False
    container_exit_code: int | None = None
    image: str | None = None


def _docker_command(
    container_name: str,
    image: str,
    memory_mb: int,
    cpus: float,
    pids_limit: int,
) -> list[str]:
    return [
        "docker",
        "run",
        "--name",
        container_name,
        "--rm",
        "--interactive",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        str(pids_limit),
        "--memory",
        f"{memory_mb}m",
        "--memory-swap",
        f"{memory_mb}m",
        "--cpus",
        str(cpus),
        "--ulimit",
        "nofile=64:64",
        "--ulimit",
        f"nproc={pids_limit}:{pids_limit}",
        "--ipc",
        "none",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--user",
        "65532:65532",
        image,
    ]


def _remove_container(container_name: str) -> None:
    """Remove a container after a timeout or interruption."""
    subprocess.run(
        ["docker", "rm", "--force", container_name],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def inspect_docker_image(image: str) -> str:
    """Return the immutable ID of a local Docker image.

    Args:
        image: Docker image name or tag.
    """
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker CLI was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Docker image inspection timed out") from exc
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"Docker sandbox image {image!r} is unavailable: {message[:1000]}"
        )
    image_id = completed.stdout.strip()
    if not image_id.startswith("sha256:"):
        raise RuntimeError(f"Docker returned an invalid image ID: {image_id!r}")
    return image_id


def run_candidate(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    *,
    enabled: bool = False,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = 256,
    cpus: float = 0.5,
    pids_limit: int = 32,
) -> ExecutionResult:
    """Run one HumanEval completion in Docker.

    Args:
        prompt: Function prefix supplied by HumanEval.
        completion: Generated code to test.
        test_code: HumanEval checks to execute.
        entry_point: Function passed to the checks.
        enabled: Explicitly allow generated-code execution.
        image: Docker image used as the sandbox.
        timeout_seconds: Maximum execution time.
        memory_mb: Container memory limit.
        cpus: Container CPU limit.
        pids_limit: Maximum container processes.
    """
    if not enabled:
        raise RuntimeError(
            "generated-code execution is disabled; explicitly enable the "
            "Docker sandbox after building the configured image"
        )
    if not image.strip():
        raise ValueError("Docker sandbox image must be non-empty")
    if timeout_seconds <= 0:
        raise ValueError("Docker timeout must be positive")
    if memory_mb < 64:
        raise ValueError("Docker memory limit must be at least 64 MB")
    if cpus <= 0:
        raise ValueError("Docker CPU limit must be positive")
    if pids_limit < 1:
        raise ValueError("Docker PID limit must be positive")

    payload = json.dumps(
        {
            "prompt": prompt,
            "completion": completion,
            "test_code": test_code,
            "entry_point": entry_point,
            "timeout_seconds": timeout_seconds,
        }
    )
    if len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError("HumanEval execution payload exceeds the 2 MB limit")

    container_name = f"taskcompiler-humaneval-{uuid.uuid4().hex}"
    command = _docker_command(
        container_name,
        image,
        memory_mb,
        cpus,
        pids_limit,
    )
    host_timeout = timeout_seconds + 5.0
    try:
        completed = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=host_timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker CLI was not found; no local-execution fallback is permitted"
        ) from exc
    except subprocess.TimeoutExpired:
        _remove_container(container_name)
        return ExecutionResult(
            passed=False,
            error=f"container exceeded {timeout_seconds:.2f}s execution limit",
            timed_out=True,
            image=image,
        )

    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Docker sandbox failed before returning a test result: "
            f"{message[:2000] or f'exit code {completed.returncode}'}"
        )

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Docker sandbox returned malformed JSON: "
            f"{completed.stdout[:1000]!r}"
        ) from exc
    if not isinstance(response, dict) or not isinstance(response.get("passed"), bool):
        raise RuntimeError("Docker sandbox result does not match the expected schema")
    error = response.get("error")
    if error is not None and not isinstance(error, str):
        raise RuntimeError("Docker sandbox error field must be a string or null")
    return ExecutionResult(
        passed=response["passed"],
        error=error,
        timed_out=bool(response.get("timed_out", False)),
        container_exit_code=completed.returncode,
        image=image,
    )


def extract_code_block(text: str) -> str:
    """Remove Markdown fences while preserving Python indentation."""
    if "```" not in text:
        return text.strip("\r\n")
    parts = text.split("```")
    for part in parts:
        candidate = part.strip("\r\n")
        first_line, separator, remainder = candidate.partition("\n")
        if first_line.strip().casefold() == "python":
            candidate = remainder if separator else ""
        if candidate.strip():
            return candidate.strip("\r\n")
    return text.strip("\r\n")
