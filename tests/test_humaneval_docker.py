import subprocess

from taskcompiler.guidance_track import humaneval_exec
from taskcompiler.guidance_track.pipeline import _completion_text


def test_docker_command_has_required_isolation_and_no_host_mount(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"passed": true, "error": null, "timed_out": false}',
            stderr="",
        )

    monkeypatch.setattr(humaneval_exec.subprocess, "run", fake_run)
    result = humaneval_exec.run_candidate(
        "def add(a, b):\n",
        "    return a + b\n",
        "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        "add",
        enabled=True,
        image="sandbox:test",
    )

    command = captured["command"]
    assert result.passed
    assert command[:2] == ["docker", "run"]
    assert command[-1] == "sandbox:test"
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert (
        command[command.index("--security-opt") + 1]
        == "no-new-privileges:true"
    )
    assert command[command.index("--user") + 1] == "65532:65532"
    assert "--memory" in command
    assert "--cpus" in command
    assert "--pids-limit" in command
    assert "--tmpfs" in command
    assert "--volume" not in command
    assert "-v" not in command
    assert captured["kwargs"]["input"]


def test_client_timeout_forcibly_removes_named_container(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(command, timeout=1)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(humaneval_exec.subprocess, "run", fake_run)
    result = humaneval_exec.run_candidate(
        "def loop():\n",
        "    while True:\n        pass\n",
        "def check(candidate):\n    candidate()\n",
        "loop",
        enabled=True,
        timeout_seconds=0.1,
    )

    assert result.timed_out
    assert calls[1][:3] == ["docker", "rm", "--force"]
    assert calls[1][3] == calls[0][calls[0].index("--name") + 1]


def test_completion_indentation_is_restored_after_model_whitespace_stripping():
    prompt = 'def add(a, b):\n    """Add two values."""\n'
    assert _completion_text("return a + b", prompt) == "    return a + b"


def test_fenced_completion_preserves_python_body_indentation():
    prompt = 'def add(a, b):\n    """Add two values."""\n'
    output = "```python\n    return a + b\n```"
    assert _completion_text(output, prompt) == "    return a + b"
