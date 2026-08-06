from types import SimpleNamespace

import pytest

from taskcompiler.config import OLLAMA_BACKEND, RunConfig
from taskcompiler.guidance_track import gepa_baseline
from taskcompiler.metrics import RunRecorder


class _Student:
    def __init__(self, model_name, **_kwargs):
        self.model_name = model_name
        self.identity = {"backend": "fake", "model": model_name}

    def ask(self, _prompt, system=None):
        return SimpleNamespace(
            text="Answer: 1",
            input_tokens=2,
            output_tokens=2,
        )


class _Teacher:
    def __init__(self, model, **_kwargs):
        self.model = model
        self.identity = {"backend": "fake", "model": model}

    def call(self, _system, _prompt, max_tokens):
        return SimpleNamespace(
            text="Keep the prompt concise.",
            input_tokens=2,
            output_tokens=2,
        )


def _row(index: int) -> dict:
    return {
        "item_id": f"row-{index}",
        "question": "How many objects are there?",
        "reference": "1",
    }


def test_gepa_adapter_returns_seed_without_test_access(monkeypatch, tmp_path):
    pytest.importorskip("gepa")
    monkeypatch.setattr(gepa_baseline, "StudentModel", _Student)
    monkeypatch.setattr(gepa_baseline, "TeacherClient", _Teacher)
    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    config = RunConfig(
        track="guidance_track",
        dataset="bbh",
        baseline="gepa",
        benchmark_task="object_counting",
        student_backend=OLLAMA_BACKEND,
        optimizer_metric_budget=2,
    )

    prompt, protocol = gepa_baseline.optimize_prompt(
        config,
        RunRecorder(run_dir),
        run_dir,
        [_row(0), _row(1), _row(2)],
        [_row(3), _row(4)],
        "Answer carefully.",
    )

    assert prompt == "Answer carefully."
    assert protocol["metric_calls_used"] == 2
    assert protocol["test_access_during_optimization"] is False
    assert (run_dir / "artifacts" / "gepa_result.json").exists()


def test_gepa_label_adapter_uses_build_rows_only(tmp_path):
    pytest.importorskip("gepa")
    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    config = RunConfig(
        track="rules_track",
        dataset="synthetic_policies",
        baseline="gepa",
        optimizer_metric_budget=2,
    )
    rows = [
        {"item_id": f"build-{index}", "input": "Input: x", "answer": "1"}
        for index in range(2)
    ]

    prompt, protocol = gepa_baseline.optimize_label_prompt(
        config,
        RunRecorder(run_dir),
        run_dir,
        task_id="policy-000",
        rows=rows,
        starting_prompt="Apply the policy.",
        labels={"1"},
        student=_Student("fake"),
        teacher=_Teacher("fake"),
    )

    assert prompt == "Apply the policy."
    assert protocol["metric_calls_used"] == 2
    assert protocol["audit_access_during_optimization"] is False
    assert protocol["final_access_during_optimization"] is False
    assert (
        run_dir / "artifacts" / "policy-000_gepa_result.json"
    ).exists()
