"""Evaluate a DSL program against one input record."""

from __future__ import annotations

from dataclasses import dataclass, field

from taskcompiler.dsl.ast import (
    And,
    Comparison,
    Effect,
    FieldRef,
    Not,
    Or,
    PredicateCall,
    Program,
    Rule,
)
from taskcompiler.schema import Boolean3


class InterpreterError(Exception):
    pass


@dataclass
class ExecutionTrace:
    fired_rules: list[str] = field(default_factory=list)
    rule_conditions: dict[str, Boolean3] = field(default_factory=dict)
    predicate_values: dict[str, Boolean3] = field(default_factory=dict)
    unknown_rules: list[str] = field(default_factory=list)
    effects_by_rule: dict[str, list[tuple[str, str | None, int]]] = field(default_factory=dict)
    required_labels: list[str] = field(default_factory=list)
    forbidden_labels: list[str] = field(default_factory=list)
    preferred: list[tuple[str, int]] = field(default_factory=list)
    abstained: bool = False
    output_label: str | None = None
    conflict: bool = False
    invariant_values: dict[str, Boolean3] = field(default_factory=dict)
    invariant_violations: list[str] = field(default_factory=list)
    source_clauses: dict[str, str | None] = field(default_factory=dict)


@dataclass
class _EvalContext:
    predicates: dict
    remaining_steps: int
    cache: dict[str, Boolean3] = field(default_factory=dict)
    active: set[str] = field(default_factory=set)

    def step(self) -> None:
        self.remaining_steps -= 1
        if self.remaining_steps < 0:
            raise InterpreterError("DSL evaluation step budget exceeded")


def _to_boolean3(value) -> Boolean3:
    if isinstance(value, Boolean3):
        return value
    if value is None:
        return Boolean3.UNKNOWN
    if isinstance(value, bool):
        return Boolean3.TRUE if value else Boolean3.FALSE
    raise InterpreterError(f"cannot interpret {value!r} as a three-valued fact")


def _lookup_field(input_record: dict, field_path: str):
    parts = field_path.split(".")
    value = input_record
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def eval_expr(expr, input_record: dict, predicates: dict | _EvalContext) -> Boolean3:
    context = predicates if isinstance(predicates, _EvalContext) else _EvalContext(predicates, 10_000)
    context.step()
    if isinstance(expr, Comparison):
        actual = _lookup_field(input_record, expr.left.field)
        if actual is None:
            return Boolean3.UNKNOWN
        if isinstance(actual, Boolean3):
            actual = actual.value
        if expr.op == "==":
            return Boolean3.TRUE if str(actual) == expr.right.value else Boolean3.FALSE
        if expr.op == "!=":
            return Boolean3.TRUE if str(actual) != expr.right.value else Boolean3.FALSE
        if expr.op == "in":
            allowed = {lit.value for lit in expr.right}
            return Boolean3.TRUE if str(actual) in allowed else Boolean3.FALSE
        raise InterpreterError(f"unknown comparison operator: {expr.op}")

    if isinstance(expr, PredicateCall):
        if expr.name not in context.predicates:
            raise InterpreterError(f"undeclared predicate: {expr.name}")
        if expr.name in context.cache:
            return context.cache[expr.name]
        if expr.name in context.active:
            raise InterpreterError(f"recursive predicate evaluation: {expr.name}")
        context.active.add(expr.name)
        value = eval_expr(context.predicates[expr.name].body, input_record, context)
        context.active.remove(expr.name)
        context.cache[expr.name] = value
        return value

    if isinstance(expr, Not):
        return eval_expr(expr.operand, input_record, context).negate()

    if isinstance(expr, And):
        return eval_expr(expr.left, input_record, context) & eval_expr(expr.right, input_record, context)

    if isinstance(expr, Or):
        return eval_expr(expr.left, input_record, context) | eval_expr(expr.right, input_record, context)

    if isinstance(expr, FieldRef):
        return _to_boolean3(_lookup_field(input_record, expr.field))

    raise InterpreterError(f"unhandled expression node: {expr!r}")


def run(program: Program, input_record: dict, max_steps: int = 10_000) -> ExecutionTrace:
    predicates = {p.name: p for p in program.predicates}
    context = _EvalContext(predicates=predicates, remaining_steps=max_steps)
    trace = ExecutionTrace()

    for rule in program.rules:
        condition_value = eval_expr(rule.condition, input_record, context)
        trace.rule_conditions[rule.name] = condition_value
        trace.source_clauses[rule.name] = rule.source_clause
        if condition_value is not Boolean3.TRUE:
            if condition_value is Boolean3.UNKNOWN:
                trace.unknown_rules.append(rule.name)
            continue
        trace.fired_rules.append(rule.name)
        trace.effects_by_rule[rule.name] = [
            (effect.kind, effect.label, effect.priority) for effect in rule.effects
        ]
        _apply_effects(rule.effects, trace)

    _resolve(trace)
    trace.predicate_values.update(context.cache)
    for invariant in program.invariants:
        value = eval_expr(invariant.condition, input_record, context)
        trace.invariant_values[invariant.name] = value
        if value is Boolean3.FALSE:
            trace.invariant_violations.append(invariant.name)
    return trace


def _apply_effects(effects: list[Effect], trace: ExecutionTrace) -> None:
    for effect in effects:
        if effect.kind == "require":
            trace.required_labels.append(effect.label)
        elif effect.kind == "forbid":
            trace.forbidden_labels.append(effect.label)
        elif effect.kind == "prefer":
            trace.preferred.append((effect.label, effect.priority))
        elif effect.kind == "abstain":
            trace.abstained = True
        else:
            raise InterpreterError(f"unknown effect kind: {effect.kind}")


def _resolve(trace: ExecutionTrace) -> None:
    required = set(trace.required_labels)
    forbidden = set(trace.forbidden_labels)

    if trace.abstained:
        if required:
            trace.conflict = True
        trace.output_label = None
        return

    if len(required) > 1 or (required and required & forbidden):
        trace.conflict = True
        trace.abstained = True
        trace.output_label = None
        return

    if required:
        trace.output_label = next(iter(required))
        return

    allowed_preferences = [(label, priority) for label, priority in trace.preferred if label not in forbidden]
    if allowed_preferences:
        allowed_preferences.sort(key=lambda pair: pair[1], reverse=True)
        highest = allowed_preferences[0][1]
        winners = {label for label, priority in allowed_preferences if priority == highest}
        if len(winners) > 1:
            trace.conflict = True
            trace.abstained = True
            trace.output_label = None
            return
        trace.output_label = next(iter(winners))
        return

    trace.output_label = None


def find_responsible_rule(program: Program, input_record: dict, expected_label: str | None) -> str | None:
    """Find the rule most likely responsible for an unexpected output.

    Args:
        program: Program to inspect.
        input_record: Input that produced the unexpected output.
        expected_label: Label the program should produce.
    """
    trace = run(program, input_record)
    if trace.output_label == expected_label:
        return None

    for rule in trace.fired_rules:
        reduced = Program(
            predicates=program.predicates,
            rules=[r for r in program.rules if r.name != rule],
            invariants=program.invariants,
        )
        reduced_trace = run(reduced, input_record)
        if reduced_trace.output_label == expected_label:
            return rule

    return trace.fired_rules[0] if trace.fired_rules else None
