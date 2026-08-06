import json
from types import SimpleNamespace

from taskcompiler.metrics import RunRecorder


def test_log_is_saved_and_echoed(capsys, tmp_path):
    recorder = RunRecorder(tmp_path)

    recorder.log("stage started: example")

    assert "stage started: example" in capsys.readouterr().out
    assert "stage started: example" in (tmp_path / "log.txt").read_text(
        encoding="utf-8"
    )


def test_log_survives_a_closed_terminal(monkeypatch, tmp_path):
    recorder = RunRecorder(tmp_path)
    calls = 0

    def closed_terminal(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr("builtins.print", closed_terminal)
    recorder.log("first")
    recorder.log("second")

    assert calls == 1
    assert "first" in recorder.log_path.read_text(encoding="utf-8")
    assert "second" in recorder.log_path.read_text(encoding="utf-8")


def test_log_escapes_characters_unsupported_by_terminal(
    monkeypatch, tmp_path
):
    recorder = RunRecorder(tmp_path)
    output = []
    monkeypatch.setattr(
        "taskcompiler.metrics.sys.stdout",
        SimpleNamespace(encoding="cp1252"),
    )
    monkeypatch.setattr(
        "builtins.print",
        lambda text, **_kwargs: output.append(text),
    )

    recorder.log("a \u2260 b")

    assert "\\u2260" in output[0]
    assert "a \u2260 b" in recorder.log_path.read_text(encoding="utf-8")


def test_summary_records_model_identity(tmp_path):
    recorder = RunRecorder(tmp_path)
    recorder.set_model_identity(
        "student",
        {"backend": "ollama", "model": "example:1b", "digest": "sha256:test"},
    )

    recorder.finalize()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["environment"]["models"]["student"]["digest"] == "sha256:test"
    source = summary["environment"]["source"]
    assert len(source["snapshot_sha256"]) == 64
    assert source["file_count"] > 0
    assert len(source["files"]) == source["file_count"]


def test_source_snapshot_is_captured_when_the_run_starts(monkeypatch, tmp_path):
    calls = 0

    def source_snapshot():
        nonlocal calls
        calls += 1
        return {
            "snapshot_sha256": "a" * 64,
            "file_count": 1,
            "files": [{"path": "example.py", "sha256": "b" * 64}],
        }

    monkeypatch.setattr("taskcompiler.metrics._source_snapshot", source_snapshot)
    recorder = RunRecorder(tmp_path)
    recorder.finalize()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert calls == 1
    assert summary["environment"]["source"]["snapshot_sha256"] == "a" * 64
