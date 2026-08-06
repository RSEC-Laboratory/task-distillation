"""Define the task DSL syntax tree."""

from __future__ import annotations

from dataclasses import dataclass


class Expr:
    """Base class for boolean expressions inside a rule condition."""


@dataclass
class FieldRef(Expr):
    """A reference to one field of the input record, e.g. event.operation."""

    field: str


@dataclass
class Literal(Expr):
    """A literal value to compare against, e.g. an enum member name."""

    value: str


@dataclass
class Comparison(Expr):
    """field == value, field != value, or field in {a, b, c}."""

    left: FieldRef
    op: str  # "==", "!=", or "in"
    right: Literal | list[Literal]


@dataclass
class PredicateCall(Expr):
    """A call to a previously declared predicate, e.g. StateChanging(event)."""

    name: str


@dataclass
class Not(Expr):
    operand: Expr


@dataclass
class And(Expr):
    left: Expr
    right: Expr


@dataclass
class Or(Expr):
    left: Expr
    right: Expr


@dataclass
class PredicateDecl:
    name: str
    parameter: str
    body: Expr


EFFECT_KINDS = ("require", "forbid", "prefer", "abstain")


@dataclass
class Effect:
    kind: str  # one of EFFECT_KINDS
    label: str | None  # None only for "abstain"
    priority: int = 0


@dataclass
class Rule:
    name: str
    condition: Expr
    effects: list[Effect]
    source_clause: str | None = None


@dataclass
class Invariant:
    name: str
    condition: Expr


@dataclass
class Program:
    predicates: list[PredicateDecl]
    rules: list[Rule]
    invariants: list[Invariant]

    def find_rule(self, name: str) -> Rule | None:
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    def find_predicate(self, name: str) -> PredicateDecl | None:
        for predicate in self.predicates:
            if predicate.name == name:
                return predicate
        return None
