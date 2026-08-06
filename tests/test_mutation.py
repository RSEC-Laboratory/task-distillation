from taskcompiler.dsl.mutate import generate_mutants, run_mutation_testing
from taskcompiler.dsl.parser import parse_program
from taskcompiler.verify import RuleCase, run_mutation_check, run_unit_tests
from taskcompiler.verify import repair_loop
from taskcompiler.teacher import TeacherResponse

PROGRAM_SOURCE = """
predicate StateChanging(event):
    event.operation in {write_coil, write_register}

rule UnauthorizedStateChange:
    when StateChanging(event) and event.authorized == false
    then require label = manipulation
"""

TEST_CASES = [
    RuleCase({"event": {"operation": "write_coil", "authorized": "false"}}, "manipulation"),
    RuleCase({"event": {"operation": "write_coil", "authorized": "true"}}, None),
    RuleCase({"event": {"operation": "read", "authorized": "false"}}, None),
]

SAMPLE_INPUTS = [case.input_record for case in TEST_CASES]


def test_generates_mutants_for_each_operator():
    program = parse_program(PROGRAM_SOURCE)
    mutants = generate_mutants(program)
    operators_seen = {m.operator for m in mutants}
    assert "negate_condition" in operators_seen
    assert "swap_effect" in operators_seen


def test_good_test_suite_kills_most_mutants():
    program = parse_program(PROGRAM_SOURCE)
    report = run_mutation_check(program, TEST_CASES, SAMPLE_INPUTS)
    assert report.valid_mutants > 0
    assert report.adequacy > 0.5


def test_weak_test_suite_leaves_mutants_alive():
    program = parse_program(PROGRAM_SOURCE)
    weak_tests = [RuleCase({"event": {"operation": "write_coil", "authorized": "false"}}, "manipulation")]
    report = run_mutation_testing(program, SAMPLE_INPUTS, [(c.input_record, c.expected_label) for c in weak_tests])
    assert report.adequacy < 1.0


def test_run_unit_tests_reports_pass_rate():
    program = parse_program(PROGRAM_SOURCE)
    report = run_unit_tests(program, TEST_CASES)
    assert report.total == 3
    assert report.pass_rate == 1.0


class _RegressingTeacher:
    def call(self, system, prompt):
        return TeacherResponse(
            text=(
                "rule Original:\n"
                "    when event.x in {b, c}\n"
                "    then require label = yes"
            ),
            input_tokens=1,
            output_tokens=1,
            latency_seconds=0.0,
            model="fake",
        )


def test_repair_rejects_aggregate_improvement_with_regression():
    from taskcompiler.translate import render_rule

    program = parse_program(
        "rule Original: when event.x in {a, e} then require label = yes"
    )
    cases = [
        RuleCase({"event": {"x": "a"}}, None),
        RuleCase({"event": {"x": "b"}}, "yes"),
        RuleCase({"event": {"x": "c"}}, "yes"),
        RuleCase({"event": {"x": "e"}}, "yes"),
        RuleCase({"event": {"x": "d"}}, None),
    ]
    repaired, attempts = repair_loop(
        program,
        render_rule=render_rule,
        parse_rule=lambda text: parse_program(text).rules[0],
        test_cases=cases,
        teacher=_RegressingTeacher(),
        max_rounds=1,
    )
    assert attempts[0].accepted is False
    assert [literal.value for literal in repaired.rules[0].condition.right] == ["a", "e"]
