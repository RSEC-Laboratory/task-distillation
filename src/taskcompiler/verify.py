"""Test DSL programs and refine rules from source-grounded failures."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from taskcompiler.dsl.ast import Program
from taskcompiler.dsl.interpreter import find_responsible_rule, run
from taskcompiler.dsl.mutate import MutationReport, run_mutation_testing
from taskcompiler.sandbox import run_sandboxed
from taskcompiler.teacher import TeacherClient


@dataclass
class RuleCase:
    input_record: dict
    expected_label: str | None


@dataclass
class TestReport:
    total: int
    passed: int
    failures: list[RuleCase] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 1.0


def run_unit_tests(program: Program, test_cases: list[RuleCase]) -> TestReport:
    passed = 0
    failures: list[RuleCase] = []
    for case in test_cases:
        result = run_sandboxed(program, case.input_record)
        if result.error or result.timed_out:
            failures.append(case)
            continue
        if result.trace.output_label == case.expected_label:
            passed += 1
        else:
            failures.append(case)
    return TestReport(total=len(test_cases), passed=passed, failures=failures)


def run_property_tests(
    program: Program,
    field_values: dict[str, list[str]],
    invariant_check,
    max_combinations: int = 200,
) -> TestReport:
    """Check an invariant over generated field combinations.

    Args:
        program: Program under test.
        field_values: Allowed values for each input field.
        invariant_check: Callable that checks an input and output label.
        max_combinations: Maximum combinations to evaluate.
    """
    keys = list(field_values.keys())
    value_lists = [field_values[k] for k in keys]
    combinations = list(itertools.islice(itertools.product(*value_lists), max_combinations))

    passed = 0
    failures: list[RuleCase] = []
    for combo in combinations:
        input_record = {"event": dict(zip(keys, combo))}
        result = run_sandboxed(program, input_record)
        if result.error or result.timed_out or result.trace is None:
            failures.append(RuleCase(input_record=input_record, expected_label=None))
            continue
        output_label = result.trace.output_label
        if invariant_check(input_record, output_label):
            passed += 1
        else:
            failures.append(RuleCase(input_record=input_record, expected_label=None))

    return TestReport(total=len(combinations), passed=passed, failures=failures)


def run_mutation_check(program: Program, test_cases: list[RuleCase], sample_inputs: list[dict]) -> MutationReport:
    pairs = [(case.input_record, case.expected_label) for case in test_cases]
    return run_mutation_testing(
        program,
        sample_inputs,
        pairs,
        exhaustive_equivalence_domain=False,
    )


REPAIR_SYSTEM_PROMPT = (
    "You repair a small rule-based specification language. You will be shown "
    "one rule that produced a wrong answer on a specific test case, and the "
    "answer the task specification says is correct. Rewrite only that rule "
    "so it produces the correct answer on this case, without breaking its "
    "correct behavior on other cases. If the marker says no existing rule "
    "fired, write one new rule covering only the missing source-grounded "
    "behavior. Reply with only the corrected or new rule, "
    "in the same syntax, no explanation."
)


@dataclass
class RepairAttempt:
    rule_name: str
    original_source: str
    proposed_source: str
    accepted: bool


def repair_loop(
    program: Program,
    render_rule,
    parse_rule,
    test_cases: list[RuleCase],
    teacher: TeacherClient,
    max_rounds: int = 3,
    validate_candidate=None,
    on_teacher_response=None,
) -> tuple[Program, list[RepairAttempt]]:
    """Refine failing rules without regressing passing cases.

    Args:
        program: Initial program.
        render_rule: Callable that renders one rule.
        parse_rule: Callable that parses one rule.
        test_cases: Source-grounded cases to satisfy.
        teacher: Model used to propose rule changes.
        max_rounds: Maximum refinement attempts.
        validate_candidate: Optional candidate validator.
        on_teacher_response: Optional response callback.
    """
    attempts: list[RepairAttempt] = []
    current = program

    for _ in range(max_rounds):
        report = run_unit_tests(current, test_cases)
        if not report.failures:
            break

        failing_case = report.failures[0]
        responsible_name = find_responsible_rule(
            current, failing_case.input_record, failing_case.expected_label
        )
        adding_rule = responsible_name is None
        if adding_rule:
            responsible_name = f"MissingBehaviorRound{len(attempts) + 1}"
            rule = None
            original_source = "<no existing rule fired; add one source-grounded rule>"
        else:
            rule = current.find_rule(responsible_name)
            original_source = render_rule(rule)
        prompt = (
            f"Rule:\n{original_source}\n\n"
            f"Failing input: {failing_case.input_record}\n"
            f"Correct label according to the task specification: {failing_case.expected_label}\n"
        )
        response = teacher.call(REPAIR_SYSTEM_PROMPT, prompt)
        if on_teacher_response is not None:
            on_teacher_response(response)
        try:
            proposed_rule = parse_rule(response.text)
        except Exception as exc:
            attempts.append(
                RepairAttempt(
                    rule_name=responsible_name,
                    original_source=original_source,
                    proposed_source=response.text,
                    accepted=False,
                )
            )
            break

        # Keep the original rule identity for provenance.
        proposed_rule.name = responsible_name
        proposed_rule.source_clause = rule.source_clause if rule else "source-grounded-repair"
        candidate_rules = (
            current.rules + [proposed_rule]
            if adding_rule
            else [
                proposed_rule if r.name == responsible_name else r
                for r in current.rules
            ]
        )
        candidate_program = Program(
            predicates=current.predicates, rules=candidate_rules, invariants=current.invariants
        )
        if validate_candidate is not None:
            try:
                validate_candidate(candidate_program)
            except Exception:
                attempts.append(
                    RepairAttempt(
                        rule_name=responsible_name,
                        original_source=original_source,
                        proposed_source=response.text,
                        accepted=False,
                    )
                )
                break
        candidate_report = run_unit_tests(candidate_program, test_cases)

        previously_passing = []
        for case in test_cases:
            old = run_sandboxed(current, case.input_record)
            if (
                old.trace is not None
                and not old.error
                and old.trace.output_label == case.expected_label
            ):
                previously_passing.append(case)
        protected = run_unit_tests(candidate_program, previously_passing)
        accepted = (
            protected.passed == protected.total
            and candidate_report.passed > report.passed
        )
        attempts.append(
            RepairAttempt(
                rule_name=responsible_name,
                original_source=original_source,
                proposed_source=response.text,
                accepted=accepted,
            )
        )
        if accepted:
            current = candidate_program
        else:
            break

    return current, attempts
