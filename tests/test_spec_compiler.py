from dataclasses import dataclass

from taskcompiler.rules_track.synthetic_policies import make_policy_task
from taskcompiler.spec_compiler import (
    compile_specification,
    extract_dsl_source,
    normalize_dsl_source,
)


@dataclass
class _Response:
    text: str
    input_tokens: int = 10
    output_tokens: int = 20


class _Teacher:
    def __init__(self, text: str | list[str]):
        self.texts = [text] if isinstance(text, str) else list(text)
        self.prompts = []

    def call(
        self,
        system,
        prompt,
        max_tokens=2000,
        response_format=None,
        seed=None,
    ):
        self.prompts.append(
            {
                "system": system,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "seed": seed,
            }
        )
        return _Response(self.texts.pop(0))


def test_extract_dsl_source_accepts_a_markdown_fence():
    assert extract_dsl_source("```dsl\nrule A:\n    when x == y\n    then abstain\n```") == (
        "rule A:\n    when x == y\n    then abstain"
    )


def test_extract_dsl_source_accepts_structured_json():
    assert extract_dsl_source(
        '{"program":"rule A:\\n    when x == y\\n    then abstain"}'
    ) == "rule A:\n    when x == y\n    then abstain"


def test_normalize_dsl_source_unquotes_identifier_literals():
    source = (
        'predicate Risk(event):\n'
        '    event.risk == "TRUE"\n\n'
        "rule A:\n"
        "    when event.role != 'guest'\n"
        '    then require label = "review"'
    )

    assert normalize_dsl_source(source) == (
        "predicate Risk(event):\n"
        "    event.risk == TRUE\n\n"
        "rule A:\n"
        "    when event.role != guest\n"
        "    then require label = review"
    )


def test_normalize_dsl_source_maps_default_and_uncertainty_forms():
    task = make_policy_task(0)
    source = (
        "rule Unknown:\n"
        "    when event.risk == UNKNOWN\n"
        "    then require label = uncertain\n\n"
        "rule Default:\n"
        "    when true\n"
        "    then prefer label = allow priority 1"
    )

    normalized = normalize_dsl_source(source, task.task_spec)

    assert "then abstain" in normalized
    assert "when event.role == admin or event.role != admin" in normalized


def test_normalize_dsl_source_accepts_gemma_priority_and_empty_condition():
    task = make_policy_task(0)
    source = (
        "rule Preferred:\n"
        "    when event.role == admin\n"
        "    then prefer label = allow priority = 90\n\n"
        "rule Default:\n"
        "    when\n"
        "    then prefer label = review priority = 1"
    )

    normalized = normalize_dsl_source(source, task.task_spec)

    assert "priority = " not in normalized
    assert "priority 90" in normalized
    assert "when event.role == admin or event.role != admin" in normalized


def test_normalize_dsl_source_inlines_an_earlier_rule_condition():
    task = make_policy_task(0)
    source = (
        "rule TrustedExport:\n"
        "    when event.context == partner and event.role != admin\n"
        "    then prefer label = allow priority 80\n\n"
        "rule ForbidDeny:\n"
        "    when TrustedExport.condition\n"
        "    then forbid label = deny"
    )

    normalized = normalize_dsl_source(source, task.task_spec)

    assert "TrustedExport.condition" not in normalized
    assert (
        "when (event.context == partner and event.role != admin)"
        in normalized
    )


def test_normalize_dsl_source_recovers_frozen_gemma_failures():
    from pathlib import Path

    from taskcompiler.dsl.parser import parse_program

    project_root = Path(__file__).resolve().parents[1]
    for task_id in [62, 63]:
        task = make_policy_task(task_id)
        source = (
            project_root
            / "experiments" / "prepared"
            / "teacher_programs"
            / f"t10_gemma_policy{task_id:03d}_invalid.dsl"
        ).read_text(encoding="utf-8")

        program = parse_program(
            normalize_dsl_source(source, task.task_spec),
            task_spec=task.task_spec,
        )

        assert program.rules


def test_public_spec_compiler_never_sends_final_cases_to_teacher():
    task = make_policy_task(0)
    from taskcompiler.dsl.parser import render_program_source

    teacher = _Teacher(render_program_source(task.oracle_program))
    compilation = compile_specification(
        teacher,
        task.task_spec,
        task.build_cases,
        task.audit_cases,
        max_repair_rounds=0,
        max_tokens=1200,
        source_id=task.task_spec.sources[0],
        seed=17,
    )

    prompt = teacher.prompts[0]["prompt"]
    development_records = [
        item.input_record
        for item in [*task.build_cases, *task.audit_cases]
    ]
    final_only = [
        case.input_record
        for case in task.final_cases
        if case.input_record not in development_records
    ]

    assert compilation.final_build.pass_rate == 1.0
    assert compilation.final_audit.pass_rate == 1.0
    assert teacher.prompts[0]["max_tokens"] == 1200
    assert teacher.prompts[0]["seed"] == 17
    assert all(str(record) not in prompt for record in final_only)


def test_full_program_repair_accepts_only_a_non_regressing_improvement():
    task = make_policy_task(0)
    from taskcompiler.dsl.parser import render_program_source

    correct = render_program_source(task.oracle_program)
    flawed = correct.replace(
        "then require label = review",
        "then require label = allow",
        1,
    )
    teacher = _Teacher([flawed, correct])

    compilation = compile_specification(
        teacher,
        task.task_spec,
        task.build_cases,
        task.audit_cases,
        max_repair_rounds=2,
        max_tokens=1800,
        source_id=task.task_spec.sources[0],
        repair=True,
        structured_output=True,
        seed=5,
    )

    assert compilation.initial_build.pass_rate < 1.0
    assert compilation.final_build.pass_rate == 1.0
    assert compilation.final_audit.pass_rate == 1.0
    assert len(compilation.repairs) == 2
    assert all(attempt.accepted for attempt in compilation.repairs)
    assert compilation.repairs[-1].rule_name == "<complete-program>"
    assert compilation.teacher_calls == 2
    assert teacher.prompts[0]["response_format"]["type"] == "json_schema"
    assert "Source-derived counterexamples" in teacher.prompts[1]["prompt"]
    assert "Fired rules:" in teacher.prompts[1]["prompt"]
    assert "Effects by rule:" in teacher.prompts[1]["prompt"]
    assert teacher.prompts[1]["seed"] == 7


def test_syntax_repair_uses_only_public_grammar_and_parser_error():
    task = make_policy_task(0)
    from taskcompiler.dsl.parser import render_program_source

    malformed = (
        "rule Broken:\n"
        "    when event.role ==\n"
        "    then require label = allow"
    )
    corrected = render_program_source(task.oracle_program)
    teacher = _Teacher([malformed, corrected])

    compilation = compile_specification(
        teacher,
        task.task_spec,
        task.build_cases,
        task.audit_cases,
        max_repair_rounds=0,
        max_syntax_repair_rounds=1,
        max_tokens=1800,
        source_id=task.task_spec.sources[0],
        repair=True,
        seed=11,
    )

    syntax_prompt = teacher.prompts[1]["prompt"]
    source_records = [
        case.input_record
        for case in [*task.build_cases, *task.audit_cases]
    ]
    assert compilation.raw_initial_source == malformed
    assert compilation.initial_source == corrected.strip()
    assert compilation.final_build.pass_rate == 1.0
    assert compilation.final_audit.pass_rate == 1.0
    assert compilation.repairs == []
    assert len(compilation.syntax_repairs) == 1
    assert compilation.syntax_repairs[0].accepted
    assert compilation.teacher_calls == 2
    assert "Parser error:" in syntax_prompt
    assert "Source-derived counterexamples" not in syntax_prompt
    assert all(str(record) not in syntax_prompt for record in source_records)
    assert teacher.prompts[1]["seed"] == 12


def test_frozen_initial_program_skips_duplicate_teacher_generation():
    task = make_policy_task(0)
    from taskcompiler.dsl.parser import render_program_source

    teacher = _Teacher([])
    compilation = compile_specification(
        teacher,
        task.task_spec,
        task.build_cases,
        task.audit_cases,
        max_repair_rounds=0,
        max_tokens=1800,
        source_id=task.task_spec.sources[0],
        repair=False,
        initial_source=render_program_source(task.oracle_program),
        seed=5,
    )

    assert compilation.final_build.pass_rate == 1.0
    assert compilation.teacher_calls == 0
    assert compilation.teacher_input_tokens == 0
    assert compilation.teacher_output_tokens == 0
    assert teacher.prompts == []


def test_local_exclusion_repairs_a_forbidden_expected_label():
    from pathlib import Path

    task = make_policy_task(26)
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments" / "prepared"
        / "teacher_programs"
        / "abstract_t2_policy026_gemma2_27b.dsl"
    ).read_text(encoding="utf-8")
    teacher = _Teacher([])

    compilation = compile_specification(
        teacher,
        task.task_spec,
        task.build_cases,
        task.audit_cases,
        max_repair_rounds=3,
        max_tokens=1800,
        source_id=task.task_spec.sources[0],
        initial_source=source,
        repair=True,
    )

    assert compilation.initial_build.pass_rate < 1.0
    assert compilation.final_build.pass_rate == 1.0
    assert compilation.final_audit.pass_rate == 1.0
    assert compilation.repairs[0].rule_name.startswith("local-exclusion:")
    assert compilation.teacher_calls == 0


def test_audit_counterexample_can_drive_source_refinement():
    from taskcompiler.dsl.parser import render_program_source

    task = make_policy_task(0)
    audit_case = next(
        case for case in task.audit_cases if case.expected_label is not None
    )
    event = audit_case.input_record["event"]
    wrong_label = next(
        label
        for label in task.task_spec.labels
        if label != audit_case.expected_label
    )
    condition = " and ".join(
        f"event.{field} == {value}"
        for field, value in event.items()
    )
    source = (
        render_program_source(task.oracle_program)
        + "\n\nrule InjectedAuditError:\n"
        + f"    when {condition}\n"
        + f"    then require label = {wrong_label}\n"
    )

    without_audit_refinement = compile_specification(
        _Teacher([]),
        task.task_spec,
        task.build_cases,
        task.audit_cases,
        max_repair_rounds=0,
        max_tokens=1800,
        source_id=task.task_spec.sources[0],
        initial_source=source,
    )
    with_audit_refinement = compile_specification(
        _Teacher([]),
        task.task_spec,
        task.build_cases,
        task.audit_cases,
        max_repair_rounds=3,
        max_tokens=1800,
        source_id=task.task_spec.sources[0],
        initial_source=source,
        refine_with_audit=True,
    )

    assert without_audit_refinement.final_build.pass_rate == 1.0
    assert without_audit_refinement.final_audit.pass_rate < 1.0
    assert with_audit_refinement.final_build.pass_rate == 1.0
    assert with_audit_refinement.final_audit.pass_rate == 1.0
    assert with_audit_refinement.repairs


def test_exact_counterexample_patch_preserves_source_cases():
    from taskcompiler.dsl.parser import render_program_source

    task = make_policy_task(1)
    audit_case = next(
        case for case in task.audit_cases if case.expected_label is not None
    )
    event = audit_case.input_record["event"]
    wrong_label = next(
        label
        for label in task.task_spec.labels
        if label != audit_case.expected_label
    )
    condition = " and ".join(
        f"event.{field} == {value}"
        for field, value in event.items()
    )
    source = (
        render_program_source(task.oracle_program)
        + "\n\nrule InjectedAuditError:\n"
        + f"    when {condition}\n"
        + f"    then require label = {wrong_label}\n"
    )

    compilation = compile_specification(
        _Teacher([]),
        task.task_spec,
        task.build_cases,
        task.audit_cases,
        max_repair_rounds=0,
        max_counterexample_patches=1,
        max_tokens=1800,
        source_id=task.task_spec.sources[0],
        initial_source=source,
        refine_with_audit=True,
    )

    assert compilation.final_build.pass_rate == 1.0
    assert compilation.final_audit.pass_rate == 1.0
    assert compilation.teacher_calls == 0
    assert compilation.repairs[0].rule_name.startswith(
        "source-counterexample:"
    )


def test_generalized_patches_recover_frozen_gemma_abstentions():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]

    class _NoTeacher:
        def call(self, *args, **kwargs):
            raise AssertionError("parent call not allowed")

    for task_id, patch_budget in [(62, 4), (63, 3)]:
        task = make_policy_task(task_id)
        source = (
            project_root
            / "experiments" / "prepared"
            / "teacher_programs"
            / f"t12_gemma_policy{task_id:03d}_initial.dsl"
        ).read_text(encoding="utf-8")
        compilation = compile_specification(
            _NoTeacher(),
            task.task_spec,
            task.build_cases,
            task.audit_cases,
            max_repair_rounds=0,
            max_counterexample_patches=patch_budget,
            max_tokens=1800,
            source_id=task.task_spec.sources[0],
            initial_source=source,
            refine_with_audit=True,
        )

        assert compilation.final_build.pass_rate == 1.0
        assert compilation.final_audit.pass_rate == 1.0
        assert compilation.teacher_calls == 0
        assert len(compilation.repairs) <= patch_budget
