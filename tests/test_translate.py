from taskcompiler.dsl.parser import parse_program
from taskcompiler.student import StudentResponse
from taskcompiler.teacher import TeacherResponse
from taskcompiler.translate import (
    InstructionBundle,
    InstructionEntry,
    Obligation,
    _candidate_edits,
    behavioral_compile,
    build_obligation_probes,
    generic_test_driven_compile,
    normalise_student_label,
    render_program,
    render_rule,
    roundtrip_check,
)

PROGRAM_SOURCE = """
predicate StateChanging(event):
    event.operation in {write_coil, write_register}

rule UnauthorizedStateChange:
    when StateChanging(event) and event.authorized == false
    then require label = manipulation
"""


def test_render_rule_produces_readable_sentence():
    program = parse_program(PROGRAM_SOURCE)
    rule = program.rules[0]
    text = render_rule(rule)
    assert "authorized is false" in text
    assert "manipulation" in text
    assert text.startswith("If ")


def test_render_program_renders_every_rule():
    program = parse_program(PROGRAM_SOURCE)
    rendered = render_program(program)
    assert len(rendered) == 3
    assert rendered[0].startswith("Definition StateChanging:")
    assert "StateChanging" in rendered[1]
    assert rendered[2].startswith("Resolution:")


def test_roundtrip_check_is_deterministic():
    program = parse_program(PROGRAM_SOURCE)
    rule = program.rules[0]
    assert roundtrip_check(rule, sample_inputs=[]) is True


def test_label_parser_accepts_an_explicit_final_answer():
    response = (
        "The allow and deny rules do not win.\n\n"
        "Therefore, the final label is: review"
    )

    assert normalise_student_label(
        response, {"allow", "deny", "review"}
    ) == "review"


def test_label_parser_rejects_an_ambiguous_final_answer():
    response = "Reasoning omitted.\nFinal answer: allow or deny"

    assert normalise_student_label(
        response, {"allow", "deny", "review"}
    ).startswith("__INVALID__:")


def test_label_parser_accepts_one_unambiguous_label_mention():
    assert normalise_student_label(
        "After checking the policy, I select review.",
        {"allow", "deny", "review"},
    ) == "review"


def test_label_parser_handles_a_final_abstention():
    assert normalise_student_label(
        "The evidence conflicts.\nFinal answer: uncertain",
        {"allow", "deny", "review"},
    ) is None


def test_label_parser_does_not_treat_unknown_input_as_abstention():
    response = (
        "The input risk is UNKNOWN and the matching rule returns allow.\n"
        "The output is: allow"
    )

    assert normalise_student_label(
        response, {"allow", "deny", "review"}
    ) == "allow"


class _StyleSensitiveStudent:
    def ask(self, prompt, system=None):
        if "Rule Main:" in (system or "") and "'x': 'yes'" in prompt:
            text = "matched"
        else:
            text = "uncertain"
        return StudentResponse(text=text, input_tokens=10, output_tokens=1, latency_seconds=0.0)


class _GenericRewriteTeacher:
    def call(self, system, prompt):
        return TeacherResponse(
            text=(
                "Rule Main: Check whether event.x is yes. Only when that "
                "complete condition is true, then it must be labeled matched."
            ),
            input_tokens=20,
            output_tokens=10,
            latency_seconds=0.0,
            model="fake",
        )


def test_obligation_compiler_accepts_typed_local_edit():
    program = parse_program(
        "rule Main: when event.x == yes then require label = matched"
    )
    _, probes = build_obligation_probes(
        program,
        [{"event": {"x": "yes"}}, {"event": {"x": "no"}}],
        "build",
    )
    result = behavioral_compile(
        _StyleSensitiveStudent(),
        "classify",
        program,
        probes,
        max_rounds=2,
        token_budget=200,
    )
    assert result.build_pass_rate == 1.0
    assert result.bundle.entries[0].style == "explicit"
    assert result.initial_build_results[0].matched is False
    assert result.student_input_tokens > 0


def test_override_candidates_add_visible_text_and_skip_duplicates():
    program = parse_program(
        "rule Main: when event.x == yes then require label = matched"
    )
    bundle = InstructionBundle(
        [InstructionEntry(program.rules[0])],
        override_texts=["Existing compiled representation."],
    )
    candidates = _candidate_edits(
        bundle,
        Obligation("Main:require:matched", "Main", "require:matched"),
        None,
    )

    assert candidates
    assert all(candidate.render() != bundle.render() for _, candidate in candidates)
    checklist = next(
        candidate
        for name, candidate in candidates
        if name == "resolver:execution_checklist"
    )
    assert checklist.render()[0] == "Existing compiled representation."
    assert checklist.render()[-1].startswith("Execution checklist:")


def test_generic_test_driven_baseline_uses_visible_tests_only():
    program = parse_program(
        "rule Main: when event.x == yes then require label = matched"
    )
    _, build = build_obligation_probes(
        program,
        [{"event": {"x": "yes"}}, {"event": {"x": "no"}}],
        "build",
    )
    _, audit = build_obligation_probes(
        program,
        [{"event": {"x": "yes"}, "_surface": "audit"}],
        "audit",
    )
    result = generic_test_driven_compile(
        _StyleSensitiveStudent(),
        _GenericRewriteTeacher(),
        "classify",
        render_program(program),
        build,
        audit,
        {"matched"},
        max_rounds=1,
    )
    assert result.build_pass_rate == 1.0
    assert result.teacher_input_tokens == 20
