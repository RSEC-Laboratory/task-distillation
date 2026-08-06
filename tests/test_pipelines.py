import pytest
from contextlib import nullcontext

from taskcompiler.config import RULES_TRACK, RunConfig
from taskcompiler.guidance_track.humaneval_exec import extract_code_block, run_candidate
from taskcompiler.student import StudentResponse
from taskcompiler.teacher import TeacherResponse

HUMANEVAL_PROMPT = "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n"
HUMANEVAL_TEST = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(-1, 1) == 0
"""


def test_generated_code_execution_is_disabled_without_external_sandbox():
    completion = "    return a + b\n"
    with pytest.raises(RuntimeError, match="execution is disabled"):
        run_candidate(HUMANEVAL_PROMPT, completion, HUMANEVAL_TEST, "add")


def test_extract_code_block_strips_fences():
    text = "```python\ndef f():\n    return 1\n```"
    assert extract_code_block(text) == "def f():\n    return 1"


def test_extract_code_block_handles_plain_text():
    text = "    return a + b"
    assert extract_code_block(text) == "    return a + b"


class _Recorder:
    def stage(self, name):
        return nullcontext()

    def add_teacher_tokens(self, input_tokens, output_tokens):
        pass

    def add_student_tokens(self, input_tokens, output_tokens):
        pass

    def log(self, message):
        pass


class _PublicSpecTeacher:
    def __init__(self, *args, **kwargs):
        pass

    def call(self, system, prompt):
        return TeacherResponse(
            text="""```dsl
rule Entailed:
    when event.entails == TRUE
    then require label = entailment

rule Contradicted:
    when event.entails == FALSE
    then require label = contradiction

rule Unmentioned:
    when event.entails == UNKNOWN
    then require label = not_mentioned
```""",
            input_tokens=10,
            output_tokens=20,
            latency_seconds=0.0,
            model="fake",
        )


class _SymbolicStudent:
    def __init__(self, *args, **kwargs):
        pass

    def ask(self, prompt, system=None):
        if "'entails': 'TRUE'" in prompt:
            label = "entailment"
        elif "'entails': 'FALSE'" in prompt:
            label = "contradiction"
        else:
            label = "not_mentioned"
        return StudentResponse(label, 10, 1, 0.0)


def test_compilation_path_never_opens_benchmark_labels(
    monkeypatch, tmp_path
):
    from taskcompiler.rules_track import pipeline

    monkeypatch.setattr(pipeline, "TeacherClient", _PublicSpecTeacher)
    monkeypatch.setattr(pipeline, "StudentModel", _SymbolicStudent)
    monkeypatch.setattr(
        pipeline,
        "load_split",
        lambda split: (_ for _ in ()).throw(
            AssertionError("benchmark opened during compilation")
        ),
    )
    (tmp_path / "artifacts").mkdir()
    config = RunConfig(
        track=RULES_TRACK,
        dataset="contract_nli",
        baseline="full_method",
        max_repair_rounds=1,
    )
    program, bundle, compilation, metadata = pipeline._compile_upstream(
        config,
        _Recorder(),
        tmp_path,
        pipeline._public_task_spec("public hypothesis"),
    )
    assert metadata["dsl_build_pass_rate_after"] == 1.0
    assert metadata["student_audit_pass_rate"] == 1.0
    assert compilation.worst_obligation_violation == 0.0
    assert (tmp_path / "artifacts" / "instruction_bundle.json").exists()
    assert (tmp_path / "artifacts" / "dsl_teacher_response.txt").read_text(
        encoding="utf-8"
    ).startswith("```dsl")


def test_extract_dsl_source_handles_plain_and_fenced_text():
    from taskcompiler.rules_track.pipeline import _extract_dsl_source

    source = "rule Example:\n    when event.entails == TRUE\n    then abstain"

    assert _extract_dsl_source(source) == source
    assert _extract_dsl_source(f"```dsl\n{source}\n```") == source
    assert _extract_dsl_source(f"Here is the program:\n```\n{source}\n```") == source
