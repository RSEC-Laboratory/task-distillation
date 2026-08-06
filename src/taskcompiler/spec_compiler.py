"""Compile a public task specification into a verified DSL program."""

from __future__ import annotations

import json
import re
import copy
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

from taskcompiler.dsl.ast import (
    And,
    Comparison,
    Effect,
    FieldRef,
    Literal,
    Not,
    Program,
    Rule,
)
from taskcompiler.dsl.interpreter import find_responsible_rule, run
from taskcompiler.dsl.parser import (
    DslSyntaxError,
    parse_program,
    render_program_source,
    validate_program,
)
from taskcompiler.rules_track.prompts import (
    DSL_GENERATION_SYSTEM_PROMPT,
    build_dsl_generation_prompt,
)
from taskcompiler.schema import TaskSpec
from taskcompiler.teacher import TeacherClient, TeacherResponse
from taskcompiler.verify import (
    RepairAttempt,
    RuleCase,
    TestReport,
    run_unit_tests,
)


@dataclass
class SpecificationCompilation:
    """Record the initial and verified programs produced from one task spec."""

    initial_source: str
    initial_program: Program
    final_program: Program
    initial_build: TestReport
    final_build: TestReport
    final_audit: TestReport
    repairs: list[RepairAttempt]
    teacher_input_tokens: int
    teacher_output_tokens: int
    teacher_calls: int
    raw_initial_source: str = ""
    syntax_repairs: list[RepairAttempt] = field(default_factory=list)


def extract_dsl_source(text: str) -> str:
    """Read DSL source from plain text, JSON, or one Markdown code fence."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("program"), str):
            return payload["program"].strip()
    if stripped.startswith('"'):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, str):
            return payload.strip()
    if "```" not in stripped:
        return stripped
    parts = stripped.split("```")
    for candidate in parts[1::2]:
        source = candidate.strip()
        first_line, separator, remainder = source.partition("\n")
        if first_line.strip().casefold() in {"dsl", "text", "plaintext"}:
            source = remainder.strip() if separator else ""
        if source.startswith(("predicate ", "rule ")):
            return source
    return stripped


def _expand_rule_condition_references(source: str) -> str:
    """Inline references to conditions from earlier named rules."""
    conditions = {}
    current_rule = None
    lines = source.splitlines()
    for index, line in enumerate(lines):
        rule_match = re.match(
            r"^\s*rule\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$",
            line,
        )
        if rule_match:
            current_rule = rule_match.group(1)
            continue
        condition_match = re.match(r"^(\s*when\s+)(.+?)\s*$", line)
        if current_rule is None or condition_match is None:
            continue
        condition = condition_match.group(2)
        for rule_name, rule_condition in conditions.items():
            condition = re.sub(
                rf"\b{re.escape(rule_name)}\.condition\b",
                f"({rule_condition})",
                condition,
            )
        lines[index] = condition_match.group(1) + condition
        conditions[current_rule] = condition
    normalized = "\n".join(lines)
    return normalized + ("\n" if source.endswith("\n") else "")


def normalize_dsl_source(
    source: str,
    task_spec: TaskSpec | None = None,
) -> str:
    """Translate common surface forms into equivalent DSL syntax."""
    normalized = re.sub(
        r"""(["'])([A-Za-z_][A-Za-z0-9_-]*)\1""",
        r"\2",
        source,
    )
    normalized = re.sub(
        r"\bpriority\s*=\s*(\d+)\b",
        r"priority \1",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(?m)^(\s*)then\s+(?:require|prefer)\s+label\s*=\s*"
        r"(?:uncertain|unknown)(?:\s+priority\s+\d+)?\s*$",
        r"\1then abstain",
        normalized,
        flags=re.IGNORECASE,
    )
    if task_spec is not None:
        enum_values = {
            enum.name: enum.values for enum in task_spec.enums
        }
        field_ref = None
        field_value = None
        for record in task_spec.records:
            for field in record.fields:
                values = enum_values.get(field.type_name)
                if values:
                    field_ref = (
                        f"{record.name[:1].lower() + record.name[1:]}"
                        f".{field.name}"
                    )
                    field_value = values[0]
                    break
            if field_ref is not None:
                break
        if field_ref is not None and field_value is not None:
            tautology = (
                f"{field_ref} == {field_value} or "
                f"{field_ref} != {field_value}"
            )
            contradiction = (
                f"{field_ref} == {field_value} and "
                f"{field_ref} != {field_value}"
            )
            normalized = re.sub(
                r"(?mi)^(\s*when)\s+true\s*$",
                rf"\1 {tautology}",
                normalized,
            )
            normalized = re.sub(
                r"(?mi)^(\s*when)\s*$",
                rf"\1 {tautology}",
                normalized,
            )
            normalized = re.sub(
                r"(?mi)^(\s*when)\s+false\s*$",
                rf"\1 {contradiction}",
                normalized,
            )
    return _expand_rule_condition_references(normalized)


def _dsl_response_format() -> dict:
    """Describe the structured teacher response."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "dsl_program",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"program": {"type": "string"}},
                "required": ["program"],
                "additionalProperties": False,
            },
        },
    }


def _failure_text(program: Program, cases: list[RuleCase]) -> str:
    """Render source-derived counterexamples for the repair prompt."""
    rows = []
    for case in cases:
        trace = run(program, case.input_record)
        rows.append(
            f"- Input: {case.input_record}\n"
            f"  Expected: {case.expected_label or 'uncertain'}\n"
            f"  Current output: {trace.output_label or 'uncertain'}\n"
            f"  Fired rules: {trace.fired_rules}\n"
            f"  Effects by rule: {trace.effects_by_rule}\n"
            f"  Forbidden labels: {trace.forbidden_labels}\n"
            f"  Ranked preferences: {trace.preferred}"
        )
    return "\n".join(rows)


def _record_fields(record: dict) -> list[tuple[str, str]]:
    """Flatten declared record fields into DSL paths."""
    fields = []
    for root, values in record.items():
        if not isinstance(values, dict):
            continue
        fields.extend(
            (f"{root}.{name}", str(value))
            for name, value in values.items()
        )
    return fields


def _local_exclusion_repair(
    program: Program,
    report: TestReport,
    build_cases: list[RuleCase],
    task_spec: TaskSpec,
    source_id: str,
) -> tuple[Program, RepairAttempt] | None:
    """Find one minimal non-regressing exclusion from failing traces."""
    previously_passing = [
        case
        for case in build_cases
        if run(program, case.input_record).output_label
        == case.expected_label
    ]
    candidates = []
    for case in report.failures:
        responsible = find_responsible_rule(
            program,
            case.input_record,
            case.expected_label,
        )
        if responsible is None:
            continue
        original_rule = program.find_rule(responsible)
        if original_rule is None:
            continue
        for field, value in _record_fields(case.input_record):
            candidate = copy.deepcopy(program)
            candidate_rule = candidate.find_rule(responsible)
            candidate_rule.condition = And(
                candidate_rule.condition,
                Comparison(FieldRef(field), "!=", Literal(value)),
            )
            candidate_rule.source_clause = source_id
            try:
                validate_program(candidate, task_spec)
            except Exception:
                continue
            candidate_report = run_unit_tests(candidate, build_cases)
            protected = run_unit_tests(candidate, previously_passing)
            if (
                candidate_report.passed > report.passed
                and protected.passed == protected.total
            ):
                candidates.append(
                    (
                        candidate_report.passed,
                        responsible,
                        field,
                        value,
                        candidate,
                    )
                )
    if not candidates:
        return None
    _, responsible, field, value, candidate = max(
        candidates,
        key=lambda item: (item[0], item[2], item[3]),
    )
    return candidate, RepairAttempt(
        rule_name=f"local-exclusion:{responsible}",
        original_source=render_program_source(program),
        proposed_source=render_program_source(candidate),
        accepted=True,
    )


def _condition_from_fields(fields: list[tuple[str, str]]):
    """Build a conjunction from structured field values."""
    atoms = [
        Comparison(FieldRef(field), "==", Literal(value))
        for field, value in fields
    ]
    if not atoms:
        raise ValueError("counterexample record has no declared fields")
    condition = atoms[0]
    for atom in atoms[1:]:
        condition = And(condition, atom)
    return condition


def _counterexample_patch(
    program: Program,
    case: RuleCase,
    verification_cases: list[RuleCase],
    task_spec: TaskSpec,
    source_id: str,
    patch_index: int,
) -> tuple[Program, RepairAttempt] | None:
    original_report = run_unit_tests(program, verification_cases)
    previously_passing = [
        item
        for item in verification_cases
        if run(program, item.input_record).output_label
        == item.expected_label
    ]
    fields = _record_fields(case.input_record)
    candidates = []
    for width in range(1, len(fields) + 1):
        for selected in combinations(fields, width):
            candidate = copy.deepcopy(program)
            condition = _condition_from_fields(list(selected))
            trace = run(candidate, case.input_record)
            for rule_name in trace.fired_rules:
                rule = candidate.find_rule(rule_name)
                if rule is None:
                    continue
                if case.expected_label is None:
                    blocking = any(
                        effect.kind in {"require", "prefer"}
                        for effect in rule.effects
                    )
                else:
                    blocking = any(
                        effect.kind == "abstain"
                        or (
                            effect.kind == "require"
                            and effect.label != case.expected_label
                        )
                        or (
                            effect.kind == "forbid"
                            and effect.label == case.expected_label
                        )
                        for effect in rule.effects
                    )
                if blocking:
                    rule.condition = And(
                        rule.condition,
                        Not(copy.deepcopy(condition)),
                    )
            effects = (
                [Effect("abstain", None)]
                if case.expected_label is None
                else [Effect("require", case.expected_label)]
            )
            rule_name = f"SourceCounterexamplePatch{patch_index}"
            candidate.rules.append(
                Rule(
                    rule_name,
                    condition,
                    effects,
                    source_id,
                )
            )
            try:
                validate_program(candidate, task_spec)
            except Exception:
                continue
            candidate_report = run_unit_tests(
                candidate,
                verification_cases,
            )
            protected = run_unit_tests(candidate, previously_passing)
            if (
                candidate_report.passed > original_report.passed
                and protected.passed == protected.total
            ):
                candidates.append(
                    (
                        candidate_report.passed,
                        width,
                        tuple(selected),
                        candidate,
                    )
                )
    if not candidates:
        return None
    _, _, _, candidate = min(
        candidates,
        key=lambda item: (-item[0], item[1], item[2]),
    )
    return candidate, RepairAttempt(
        rule_name=(
            "source-counterexample:"
            f"SourceCounterexamplePatch{patch_index}"
        ),
        original_source=render_program_source(program),
        proposed_source=render_program_source(candidate),
        accepted=True,
    )


def compile_specification(
    teacher: TeacherClient,
    task_spec: TaskSpec,
    build_cases: list[RuleCase],
    audit_cases: list[RuleCase],
    *,
    max_repair_rounds: int,
    max_syntax_repair_rounds: int = 0,
    max_tokens: int,
    source_id: str,
    repair: bool = True,
    refine_with_audit: bool = False,
    max_counterexample_patches: int = 0,
    structured_output: bool = False,
    initial_source: str | None = None,
    seed: int | None = None,
    on_teacher_response: Callable[[TeacherResponse], None] | None = None,
    on_teacher_text: Callable[[str, str], None] | None = None,
) -> SpecificationCompilation:
    """Generate, test, and optionally repair a task program.

    Args:
        teacher: Parent model that receives the public specification.
        task_spec: Public task definition and declared schema.
        build_cases: Specification-derived scenarios used for refinement.
        audit_cases: Disjoint specification-derived scenarios used for audit.
        max_repair_rounds: Maximum accepted repair proposals.
        max_syntax_repair_rounds: Maximum syntax-only parent corrections.
        max_tokens: Maximum tokens for the initial DSL.
        source_id: Public provenance identifier attached to compiled rules.
        repair: Whether semantic failures may trigger teacher repair.
        refine_with_audit: Whether audit counterexamples may guide repair.
        max_counterexample_patches: Maximum exact source corrections.
        structured_output: Whether the teacher returns a JSON program field.
        initial_source: Optional frozen teacher DSL to validate and repair.
        seed: Optional teacher decoding seed.
        on_teacher_response: Optional callback for token accounting.
        on_teacher_text: Optional callback for raw response persistence.
    """
    prompt = build_dsl_generation_prompt(
        task_spec.description,
        task_spec.labels,
        requirements=task_spec.requirements,
        forbidden_shortcuts=task_spec.forbidden_shortcuts,
        sources=task_spec.sources,
        task_spec=task_spec,
    )
    response_format = _dsl_response_format() if structured_output else None
    response_instruction = (
        " Return one JSON object with a program string and no other fields."
        if structured_output
        else ""
    )
    if initial_source is None:
        response = teacher.call(
            DSL_GENERATION_SYSTEM_PROMPT + response_instruction,
            prompt,
            max_tokens=max_tokens,
            response_format=response_format,
            seed=seed,
        )
        if on_teacher_response is not None:
            on_teacher_response(response)
        if on_teacher_text is not None:
            on_teacher_text("initial", response.text)
        initial_source = normalize_dsl_source(
            extract_dsl_source(response.text),
            task_spec,
        )
        teacher_input_tokens = response.input_tokens
        teacher_output_tokens = response.output_tokens
        teacher_calls = 1
    else:
        initial_source = extract_dsl_source(initial_source)
        teacher_input_tokens = 0
        teacher_output_tokens = 0
        teacher_calls = 0
    raw_initial_source = initial_source
    syntax_repairs = []
    initial_program = None
    parse_error = None
    try:
        initial_program = parse_program(initial_source, task_spec=task_spec)
    except DslSyntaxError as exc:
        parse_error = exc
    if (
        initial_program is None
        and repair
        and max_syntax_repair_rounds > 0
    ):
        current_source = initial_source
        for round_index in range(1, max_syntax_repair_rounds + 1):
            syntax_prompt = (
                f"{prompt}\n\nInvalid DSL program:\n{current_source}\n\n"
                f"Parser error:\n{parse_error}\n\n"
                "Rewrite the complete DSL program so it follows the public "
                "grammar. Correct syntax and declared-schema errors only. "
                "Do not use examples or invent fields, values, or labels."
            )
            syntax_response = teacher.call(
                DSL_GENERATION_SYSTEM_PROMPT + response_instruction,
                syntax_prompt,
                max_tokens=max_tokens,
                response_format=response_format,
                seed=None if seed is None else seed + round_index,
            )
            if on_teacher_response is not None:
                on_teacher_response(syntax_response)
            if on_teacher_text is not None:
                on_teacher_text(
                    f"syntax-repair-{round_index}",
                    syntax_response.text,
                )
            teacher_input_tokens += syntax_response.input_tokens
            teacher_output_tokens += syntax_response.output_tokens
            teacher_calls += 1
            proposed_source = normalize_dsl_source(
                extract_dsl_source(syntax_response.text),
                task_spec,
            )
            accepted = False
            try:
                candidate = parse_program(
                    proposed_source,
                    task_spec=task_spec,
                )
                validate_program(candidate, task_spec)
                accepted = True
            except DslSyntaxError as exc:
                parse_error = exc
            syntax_repairs.append(
                RepairAttempt(
                    rule_name="<syntax-program>",
                    original_source=current_source,
                    proposed_source=proposed_source,
                    accepted=accepted,
                )
            )
            current_source = proposed_source
            if accepted:
                initial_source = proposed_source
                initial_program = candidate
                break
    if initial_program is None:
        assert parse_error is not None
        parse_error.failed_source = current_source if syntax_repairs else initial_source
        parse_error.syntax_repairs = syntax_repairs
        parse_error.teacher_input_tokens = teacher_input_tokens
        parse_error.teacher_output_tokens = teacher_output_tokens
        parse_error.teacher_calls = teacher_calls
        raise parse_error
    for rule in initial_program.rules:
        rule.source_clause = source_id

    initial_build = run_unit_tests(initial_program, build_cases)
    final_program = initial_program
    repairs: list[RepairAttempt] = []
    refinement_cases = (
        [*build_cases, *audit_cases]
        if refine_with_audit
        else build_cases
    )
    refinement_report = run_unit_tests(initial_program, refinement_cases)
    if repair and refinement_report.failures and max_repair_rounds:
        current_report = refinement_report
        rejected_notes = []
        for round_index in range(1, max_repair_rounds + 1):
            if not current_report.failures:
                break
            local_repair = _local_exclusion_repair(
                final_program,
                current_report,
                refinement_cases,
                task_spec,
                source_id,
            )
            if local_repair is not None:
                final_program, local_attempt = local_repair
                repairs.append(local_attempt)
                current_report = run_unit_tests(
                    final_program,
                    refinement_cases,
                )
                continue
            current_source = render_program_source(final_program)
            repair_prompt = (
                f"{prompt}\n\nCurrent DSL program:\n{current_source}\n\n"
                "Source-derived counterexamples:\n"
                f"{_failure_text(final_program, current_report.failures)}\n\n"
                "Rewrite the complete DSL program. Fix every counterexample "
                "while preserving the behavior that already passes. Do not "
                "invent fields, values, or labels."
            )
            if rejected_notes:
                repair_prompt += (
                    "\n\nPrevious proposals were rejected by deterministic "
                    "validation:\n- " + "\n- ".join(rejected_notes)
                )
            repair_response = teacher.call(
                DSL_GENERATION_SYSTEM_PROMPT + response_instruction,
                repair_prompt,
                max_tokens=max_tokens,
                response_format=response_format,
                seed=(
                    None
                    if seed is None
                    else seed + len(syntax_repairs) + round_index
                ),
            )
            if on_teacher_response is not None:
                on_teacher_response(repair_response)
            if on_teacher_text is not None:
                on_teacher_text(
                    f"repair-{round_index}",
                    repair_response.text,
                )
            teacher_input_tokens += repair_response.input_tokens
            teacher_output_tokens += repair_response.output_tokens
            teacher_calls += 1
            proposed_source = normalize_dsl_source(
                extract_dsl_source(repair_response.text),
                task_spec,
            )
            accepted = False
            rejection = ""
            try:
                candidate = parse_program(
                    proposed_source,
                    task_spec=task_spec,
                )
                for rule in candidate.rules:
                    rule.source_clause = source_id
                validate_program(candidate, task_spec)
                candidate_report = run_unit_tests(
                    candidate,
                    refinement_cases,
                )
                previously_passing = [
                    case
                    for case in refinement_cases
                    if run(final_program, case.input_record).output_label
                    == case.expected_label
                ]
                protected_report = run_unit_tests(
                    candidate,
                    previously_passing,
                )
                accepted = (
                    protected_report.passed == protected_report.total
                    and candidate_report.passed > current_report.passed
                )
                if accepted:
                    final_program = candidate
                    current_report = candidate_report
                else:
                    rejection = (
                        f"round {round_index}: passed "
                        f"{candidate_report.passed}/{candidate_report.total}; "
                        f"protected {protected_report.passed}/"
                        f"{protected_report.total}"
                    )
            except Exception as exc:
                rejection = (
                    f"round {round_index}: {type(exc).__name__}: {exc}"
                )
            repairs.append(
                RepairAttempt(
                    rule_name="<complete-program>",
                    original_source=current_source,
                    proposed_source=proposed_source,
                    accepted=accepted,
                )
            )
            if not accepted:
                rejected_notes.append(rejection)

    current_report = run_unit_tests(final_program, refinement_cases)
    for patch_index in range(1, max_counterexample_patches + 1):
        if not current_report.failures:
            break
        patch = _counterexample_patch(
            final_program,
            current_report.failures[0],
            refinement_cases,
            task_spec,
            source_id,
            patch_index,
        )
        if patch is None:
            break
        final_program, attempt = patch
        repairs.append(attempt)
        current_report = run_unit_tests(final_program, refinement_cases)

    for rule in final_program.rules:
        rule.source_clause = source_id
    validate_program(final_program, task_spec)
    final_build = run_unit_tests(final_program, build_cases)
    final_audit = run_unit_tests(final_program, audit_cases)
    return SpecificationCompilation(
        initial_source=initial_source,
        initial_program=initial_program,
        final_program=final_program,
        initial_build=initial_build,
        final_build=final_build,
        final_audit=final_audit,
        repairs=repairs,
        teacher_input_tokens=teacher_input_tokens,
        teacher_output_tokens=teacher_output_tokens,
        teacher_calls=teacher_calls,
        raw_initial_source=raw_initial_source,
        syntax_repairs=syntax_repairs,
    )
