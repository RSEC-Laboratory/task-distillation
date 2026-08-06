"""Measure whether tests detect small changes to DSL rules."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from taskcompiler.dsl.ast import And, Effect, Program, Rule
from taskcompiler.dsl.interpreter import run
from taskcompiler.dsl.parser import DslSyntaxError, validate_program

MUTATION_OPERATORS = ("drop_condition", "negate_condition", "swap_effect")


@dataclass
class Mutant:
    operator: str
    rule_name: str
    program: Program


def _drop_condition(rule: Rule) -> Rule | None:
    """Drop the right side of a top-level ``and`` condition."""
    if isinstance(rule.condition, And):
        return Rule(
            name=rule.name,
            condition=rule.condition.left,
            effects=copy.deepcopy(rule.effects),
            source_clause=rule.source_clause,
        )
    return None


def _negate_condition(rule: Rule) -> Rule:
    from taskcompiler.dsl.ast import Not

    return Rule(
        name=rule.name,
        condition=Not(copy.deepcopy(rule.condition)),
        effects=copy.deepcopy(rule.effects),
        source_clause=rule.source_clause,
    )


def _swap_effect(rule: Rule) -> Rule | None:
    if not any(effect.label for effect in rule.effects):
        return None
    swapped = [
        Effect(
            kind=("forbid" if e.kind == "require" else "require" if e.kind == "forbid" else e.kind),
            label=e.label,
            priority=e.priority,
        )
        for e in rule.effects
    ]
    if swapped == rule.effects:
        return None
    return Rule(
        name=rule.name,
        condition=copy.deepcopy(rule.condition),
        effects=swapped,
        source_clause=rule.source_clause,
    )


def generate_mutants(program: Program) -> list[Mutant]:
    mutants: list[Mutant] = []
    for rule in program.rules:
        candidates = {
            "drop_condition": _drop_condition(rule),
            "negate_condition": _negate_condition(rule),
            "swap_effect": _swap_effect(rule),
        }
        for operator, mutated_rule in candidates.items():
            if mutated_rule is None:
                continue
            new_rules = [mutated_rule if r.name == rule.name else r for r in program.rules]
            mutant_program = Program(
                predicates=copy.deepcopy(program.predicates),
                rules=new_rules,
                invariants=copy.deepcopy(program.invariants),
            )
            try:
                validate_program(mutant_program)
            except DslSyntaxError:
                continue
            mutants.append(Mutant(operator=operator, rule_name=rule.name, program=mutant_program))
    return mutants


def is_observationally_equivalent(
    original: Program, mutant: Mutant, exhaustive_inputs: list[dict]
) -> bool:
    """Compare two programs over a finite input domain.

    Args:
        original: Program used as the reference.
        mutant: Mutated program to compare.
        exhaustive_inputs: Complete set of declared inputs.
    """
    for input_record in exhaustive_inputs:
        original_trace = run(original, input_record)
        mutant_trace = run(mutant.program, input_record)
        if (
            original_trace.output_label,
            original_trace.conflict,
            original_trace.abstained,
            original_trace.invariant_violations,
        ) != (
            mutant_trace.output_label,
            mutant_trace.conflict,
            mutant_trace.abstained,
            mutant_trace.invariant_violations,
        ):
            return False
    return True


@dataclass
class MutationReport:
    total_mutants: int
    equivalent_mutants: int
    valid_mutants: int
    killed_mutants: int
    original_suite_passed: bool = True

    @property
    def adequacy(self) -> float | None:
        if self.valid_mutants == 0:
            return None
        return self.killed_mutants / self.valid_mutants


def run_mutation_testing(
    program: Program,
    sample_inputs: list[dict],
    test_cases: list[tuple[dict, str | None]],
    *,
    exhaustive_equivalence_domain: bool = False,
) -> MutationReport:
    """Measure how many valid mutants the test cases detect.

    Args:
        program: Program under test.
        sample_inputs: Inputs used to identify equivalent mutants.
        test_cases: ``(input_record, expected_label)`` pairs.
        exhaustive_equivalence_domain: Whether sample inputs cover the full domain.
    """
    for input_record, expected_label in test_cases:
        if run(program, input_record).output_label != expected_label:
            raise ValueError(
                "mutation adequacy is undefined because the original program "
                "does not pass the supplied source-grounded test suite"
            )

    mutants = generate_mutants(program)
    equivalent_count = 0
    killed_count = 0
    valid_count = 0

    for mutant in mutants:
        if exhaustive_equivalence_domain and is_observationally_equivalent(
            program, mutant, sample_inputs
        ):
            equivalent_count += 1
            continue
        valid_count += 1
        killed = False
        for input_record, expected_label in test_cases:
            mutant_trace = run(mutant.program, input_record)
            if mutant_trace.output_label != expected_label:
                killed = True
                break
        if killed:
            killed_count += 1

    return MutationReport(
        total_mutants=len(mutants),
        equivalent_mutants=equivalent_count,
        valid_mutants=valid_count,
        killed_mutants=killed_count,
    )
