"""Run DSL programs with time and evaluation-step limits."""

from __future__ import annotations

from dataclasses import dataclass
import time

from taskcompiler.dsl.ast import Program
from taskcompiler.dsl.interpreter import ExecutionTrace, InterpreterError, run

DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_MAX_STEPS = 10_000


@dataclass
class SandboxResult:
    trace: ExecutionTrace | None
    error: str | None
    timed_out: bool


def run_sandboxed(
    program: Program,
    input_record: dict,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> SandboxResult:
    """Run a DSL program within fixed resource limits.

    Args:
        program: Program to execute.
        input_record: Input passed to the program.
        timeout_seconds: Maximum wall-clock duration.
        max_steps: Maximum number of interpreter steps.
    """
    # The step limit is deterministic; the clock is a secondary guard.
    start = time.monotonic()
    try:
        trace = run(program, input_record, max_steps=max_steps)
    except (InterpreterError, RecursionError) as exc:
        return SandboxResult(trace=None, error=str(exc), timed_out=False)
    elapsed = time.monotonic() - start
    if elapsed > timeout_seconds:
        return SandboxResult(trace=None, error="execution exceeded timeout", timed_out=True)
    return SandboxResult(trace=trace, error=None, timed_out=False)
