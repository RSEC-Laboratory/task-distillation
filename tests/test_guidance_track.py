import json

import pytest

from taskcompiler.config import (
    GUIDANCE_TRACK,
    QUESTION_LEVEL_GUIDANCE,
    TASK_LEVEL_GUIDANCE,
    RunConfig,
)
from taskcompiler.guidance_track import pipeline
from taskcompiler.guidance_track.humaneval_exec import ExecutionResult
from taskcompiler.metrics import RunRecorder
from taskcompiler.student import StudentResponse
from taskcompiler.teacher import TeacherResponse


VALID_STRUCTURED_GUIDANCE = {
    "task_type": "open-ended explanation",
    "relevant_principles": ["Separate association from causal evidence."],
    "reasoning_plan": ["Identify the claim.", "Consider alternative causes."],
    "verification_checks": ["Check whether interventions were observed."],
    "uncertainties": ["State which causal assumptions remain unsupported."],
}


class _GuidanceTeacher:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def call(self, system, prompt, max_tokens=2000):
        self.__class__.calls.append(
            {"system": system, "prompt": prompt, "max_tokens": max_tokens}
        )
        if "Return only one JSON object" in system:
            text = json.dumps(VALID_STRUCTURED_GUIDANCE)
        elif "Answer the user's question directly" in system:
            text = "A direct teacher answer."
        else:
            text = "- Identify the claim.\n- Check evidence.\n- State uncertainty."
        return TeacherResponse(
            text=text,
            input_tokens=10,
            output_tokens=20,
            latency_seconds=0.0,
            model="fake-teacher",
        )


class _GuidanceStudent:
    calls = []

    def __init__(self, *args, **kwargs):
        self.__class__.calls.append({"init": kwargs})

    def ask(self, prompt, system=None):
        self.__class__.calls.append({"prompt": prompt, "system": system})
        return StudentResponse(
            text="A student answer.",
            input_tokens=12,
            output_tokens=4,
            latency_seconds=0.0,
        )


@pytest.fixture(autouse=True)
def _reset_fakes():
    _GuidanceTeacher.calls = []
    _GuidanceStudent.calls = []


def _run(monkeypatch, tmp_path, config):
    monkeypatch.setattr(pipeline, "TeacherClient", _GuidanceTeacher)
    monkeypatch.setattr(pipeline, "StudentModel", _GuidanceStudent)
    (tmp_path / "artifacts").mkdir()
    recorder = RunRecorder(tmp_path)
    metrics = pipeline.execute(config, recorder, tmp_path)
    return metrics


def test_structured_task_level_guidance_is_generated_once_and_reused(
    monkeypatch, tmp_path
):
    config = RunConfig(
        track=GUIDANCE_TRACK,
        dataset="open_ended",
        baseline=pipeline.STRUCTURED_GUIDANCE,
        guidance_mode=TASK_LEVEL_GUIDANCE,
        open_questions=["Question one?", "Question two?"],
        reference_answers=["", ""],
        num_examples=2,
    )

    metrics = _run(monkeypatch, tmp_path, config)

    assert len(_GuidanceTeacher.calls) == 1
    assert len([call for call in _GuidanceStudent.calls if "prompt" in call]) == 2
    assert metrics["guidance_events"] == 1
    assert metrics["online_teacher_access_to_questions"] is False
    events = (tmp_path / "artifacts" / "guidance_events.jsonl").read_text()
    assert '"scope": "task_level"' in events
    predictions = (tmp_path / "artifacts" / "predictions.jsonl").read_text()
    assert predictions.count('"guidance_scope": "task_level"') == 2


def test_freeform_question_level_guidance_is_generated_per_question(
    monkeypatch, tmp_path
):
    config = RunConfig(
        track=GUIDANCE_TRACK,
        dataset="open_ended",
        baseline=pipeline.FREEFORM_GUIDANCE,
        guidance_mode=QUESTION_LEVEL_GUIDANCE,
        open_questions=["Question one?", "Question two?"],
        reference_answers=["Reference one.", "Reference two."],
        num_examples=2,
    )

    metrics = _run(monkeypatch, tmp_path, config)

    assert len(_GuidanceTeacher.calls) == 2
    assert all(
        "Incoming question:" in call["prompt"] for call in _GuidanceTeacher.calls
    )
    assert all(
        "Reference one" not in call["prompt"] and "Reference two" not in call["prompt"]
        for call in _GuidanceTeacher.calls
    )
    assert metrics["guidance_events"] == 2
    assert metrics["online_teacher_access_to_questions"] is True
    assert metrics["num_references"] == 2


def test_teacher_answer_is_an_explicit_student_bypassing_comparator(
    monkeypatch, tmp_path
):
    config = RunConfig(
        track=GUIDANCE_TRACK,
        dataset="open_ended",
        baseline=pipeline.TEACHER_ANSWER,
        guidance_mode=QUESTION_LEVEL_GUIDANCE,
        open_questions=["Question one?"],
        reference_answers=[""],
        num_examples=1,
    )

    metrics = _run(monkeypatch, tmp_path, config)

    assert len(_GuidanceTeacher.calls) == 1
    assert not _GuidanceStudent.calls
    assert metrics["guidance_events"] == 0
    prediction = json.loads(
        (tmp_path / "artifacts" / "predictions.jsonl").read_text()
    )
    assert prediction["answer"] == "A direct teacher answer."
    assert prediction["guidance_scope"] is None


def test_structured_guidance_rejects_answer_field():
    invalid = dict(VALID_STRUCTURED_GUIDANCE)
    invalid["answer"] = "This must not be accepted."
    with pytest.raises(ValueError, match="keys do not match schema"):
        pipeline._validate_structured_guidance(invalid)


def test_structured_guidance_openrouter_schema_matches_validator():
    schema = pipeline._structured_guidance_response_format()
    required = schema["json_schema"]["schema"]["required"]

    assert set(required) == set(pipeline.STRUCTURED_FIELDS)


def test_open_ended_references_must_align_with_questions():
    config = RunConfig(
        track=GUIDANCE_TRACK,
        dataset="open_ended",
        open_questions=["One?", "Two?"],
        reference_answers=["Only one reference"],
    )
    with pytest.raises(ValueError, match="aligned one-to-one"):
        pipeline._load_open_rows(config)


@pytest.mark.parametrize("baseline", ["textgrad_tgd", "gepa"])
def test_optimizer_baselines_route_to_locked_bbh(monkeypatch, tmp_path, baseline):
    from taskcompiler.guidance_track import bbh

    seen = {}

    def fake_execute(config, recorder, run_dir):
        seen["baseline"] = config.baseline
        return {"baseline": config.baseline}

    monkeypatch.setattr(bbh, "execute", fake_execute)
    config = RunConfig(
        track=GUIDANCE_TRACK,
        dataset="bbh",
        baseline=baseline,
        benchmark_task="object_counting",
    )

    result = pipeline.execute(config, RunRecorder(tmp_path), tmp_path)

    assert seen["baseline"] == baseline
    assert result["baseline"] == baseline


def test_slr_bench_routes_to_external_benchmark_runner(monkeypatch, tmp_path):
    from taskcompiler.guidance_track import slr_bench

    seen = {}

    def fake_execute(config, recorder, run_dir):
        seen["dataset"] = config.dataset
        return {"dataset": config.dataset}

    monkeypatch.setattr(slr_bench, "execute", fake_execute)
    config = RunConfig(
        track=GUIDANCE_TRACK,
        dataset="slr_bench",
        baseline="raw_task",
    )

    result = pipeline.execute(config, RunRecorder(tmp_path), tmp_path)

    assert seen["dataset"] == "slr_bench"
    assert result["dataset"] == "slr_bench"


def test_humaneval_task_guidance_never_exposes_hidden_tests_to_teacher(
    monkeypatch, tmp_path
):
    hidden_test = "def check(candidate):\n    assert candidate(1, 2) == 3"
    row = {
        "task_id": "HumanEval/0",
        "prompt": 'def add(a, b):\n    """Add two values."""\n',
        "test": hidden_test,
        "entry_point": "add",
        "canonical_solution": "    return a + b",
    }
    monkeypatch.setattr(pipeline, "TeacherClient", _GuidanceTeacher)
    monkeypatch.setattr(pipeline, "StudentModel", _GuidanceStudent)
    monkeypatch.setattr(pipeline, "load_from_disk", lambda path: {"test": [row]})
    monkeypatch.setattr(
        pipeline, "inspect_docker_image", lambda image: "sha256:" + "a" * 64
    )
    executions = []

    def fake_run_candidate(prompt, completion, test_code, entry_point, **kwargs):
        executions.append(
            {
                "prompt": prompt,
                "completion": completion,
                "test_code": test_code,
                "entry_point": entry_point,
                "kwargs": kwargs,
            }
        )
        return ExecutionResult(passed=True, error=None, image=kwargs["image"])

    monkeypatch.setattr(pipeline, "run_candidate", fake_run_candidate)
    (tmp_path / "artifacts").mkdir()
    config = RunConfig(
        track=GUIDANCE_TRACK,
        dataset="humaneval",
        baseline=pipeline.STRUCTURED_GUIDANCE,
        guidance_mode=TASK_LEVEL_GUIDANCE,
        num_examples=1,
        enable_docker_sandbox=True,
    )
    metrics = pipeline.execute(config, RunRecorder(tmp_path), tmp_path)

    assert metrics["pass_at_1"] == 1.0
    assert len(_GuidanceTeacher.calls) == 1
    assert hidden_test not in _GuidanceTeacher.calls[0]["prompt"]
    assert row["canonical_solution"] not in _GuidanceTeacher.calls[0]["prompt"]
    assert executions[0]["test_code"] == hidden_test
    assert executions[0]["kwargs"]["enabled"] is True
