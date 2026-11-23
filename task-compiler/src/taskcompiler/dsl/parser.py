"""Parse and validate the restricted task DSL."""

from __future__ import annotations

import re
from collections import Counter

from taskcompiler.dsl.ast import (
    And,
    Comparison,
    Effect,
    FieldRef,
    Invariant,
    Literal,
    Not,
    Or,
    PredicateCall,
    PredicateDecl,
    Program,
    Rule,
)
from taskcompiler.schema import TaskSpec

TOKEN_PATTERN = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<COMMENT>\#[^\n]*)
  | (?P<EQ>==)
  | (?P<NEQ>!=)
  | (?P<LBRACE>\{)
  | (?P<RBRACE>\})
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<COMMA>,)
  | (?P<COLON>:)
  | (?P<DOT>\.)
  | (?P<EQUALS>=)
  | (?P<NAME>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<INT>-?[0-9]+)
    """,
    re.VERBOSE,
)

KEYWORDS = {
    "predicate", "rule", "invariant", "when", "then", "and", "or", "not", "in",
    "require", "forbid", "prefer", "abstain", "label", "priority",
}


class DslSyntaxError(Exception):
    pass


class Token:
    def __init__(self, kind: str, value: str):
        self.kind = kind
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.kind!r}, {self.value!r})"


def tokenize(source: str) -> list[Token]:
    tokens = []
    pos = 0
    while pos < len(source):
        match = TOKEN_PATTERN.match(source, pos)
        if not match:
            raise DslSyntaxError(f"unexpected character at position {pos}: {source[pos]!r}")
        pos = match.end()
        kind = match.lastgroup
        value = match.group()
        if kind in ("WS", "COMMENT"):
            continue
        if kind == "NAME" and value in KEYWORDS:
            kind = value
        tokens.append(Token(kind, value))
    tokens.append(Token("EOF", ""))
    return tokens


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _expect(self, kind: str) -> Token:
        token = self._peek()
        if token.kind != kind:
            raise DslSyntaxError(f"expected {kind!r} but found {token.kind!r} ({token.value!r})")
        return self._advance()

    def parse_program(self) -> Program:
        predicates: list[PredicateDecl] = []
        rules: list[Rule] = []
        invariants: list[Invariant] = []

        while self._peek().kind != "EOF":
            if self._peek().kind == "predicate":
                predicates.append(self._parse_predicate())
            elif self._peek().kind == "rule":
                rules.append(self._parse_rule())
            elif self._peek().kind == "invariant":
                invariants.append(self._parse_invariant())
            else:
                raise DslSyntaxError(f"expected predicate, rule, or invariant, found {self._peek()}")

        if not rules:
            raise DslSyntaxError("a program must declare at least one rule")
        program = Program(predicates=predicates, rules=rules, invariants=invariants)
        validate_program(program)
        return program

    def _parse_predicate(self) -> PredicateDecl:
        self._expect("predicate")
        name = self._expect("NAME").value
        self._expect("LPAREN")
        parameter = ""
        if self._peek().kind == "NAME":
            parameter = self._advance().value
        self._expect("RPAREN")
        self._expect("COLON")
        body = self._parse_bool_expr()
        return PredicateDecl(name=name, parameter=parameter, body=body)

    def _parse_rule(self) -> Rule:
        self._expect("rule")
        name = self._expect("NAME").value
        self._expect("COLON")
        self._expect("when")
        condition = self._parse_bool_expr()
        self._expect("then")
        effects = [self._parse_effect()]
        while self._peek().kind == "COMMA":
            self._advance()
            effects.append(self._parse_effect())
        return Rule(name=name, condition=condition, effects=effects)

    def _parse_effect(self) -> Effect:
        kind_token = self._advance()
        if kind_token.kind not in ("require", "forbid", "prefer", "abstain"):
            raise DslSyntaxError(f"expected an effect keyword, found {kind_token}")
        kind = kind_token.kind
        if kind == "abstain":
            return Effect(kind="abstain", label=None)

        self._expect("label")
        self._expect("EQUALS")
        label = self._expect("NAME").value
        priority = 0
        if kind == "prefer" and self._peek().kind == "priority":
            self._advance()
            priority = int(self._expect("INT").value)
        return Effect(kind=kind, label=label, priority=priority)

    def _parse_invariant(self) -> Invariant:
        self._expect("invariant")
        name = self._expect("NAME").value
        self._expect("COLON")
        condition = self._parse_bool_expr()
        return Invariant(name=name, condition=condition)

    def _parse_bool_expr(self):
        return self._parse_or()

    def _parse_or(self):
        left = self._parse_and()
        while self._peek().kind == "or":
            self._advance()
            right = self._parse_and()
            left = Or(left, right)
        return left

    def _parse_and(self):
        left = self._parse_unary()
        while self._peek().kind == "and":
            self._advance()
            right = self._parse_unary()
            left = And(left, right)
        return left

    def _parse_unary(self):
        if self._peek().kind == "not":
            self._advance()
            return Not(self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self):
        token = self._peek()
        if token.kind == "LPAREN":
            self._advance()
            expr = self._parse_bool_expr()
            self._expect("RPAREN")
            return expr
        if token.kind == "NAME":
            return self._parse_name_expr()
        raise DslSyntaxError(f"unexpected token in expression: {token}")

    def _parse_name_expr(self):
        first = self._expect("NAME")
        if self._peek().kind == "LPAREN":
            self._advance()
            self._expect("NAME")
            self._expect("RPAREN")
            return PredicateCall(name=first.value)
        if self._peek().kind == "DOT":
            self._advance()
            field = self._expect("NAME").value
            field_ref = FieldRef(field=f"{first.value}.{field}")
            return self._parse_comparison(field_ref)
        raise DslSyntaxError(f"expected '.' or '(' after {first.value!r}")

    def _parse_comparison(self, field_ref: FieldRef) -> Comparison:
        op_token = self._advance()
        if op_token.kind == "EQ":
            op = "=="
        elif op_token.kind == "NEQ":
            op = "!="
        elif op_token.kind == "in":
            op = "in"
        else:
            raise DslSyntaxError(f"expected '==', '!=', or 'in', found {op_token}")

        if op == "in":
            self._expect("LBRACE")
            values = [Literal(self._expect("NAME").value)]
            while self._peek().kind == "COMMA":
                self._advance()
                values.append(Literal(self._expect("NAME").value))
            self._expect("RBRACE")
            return Comparison(left=field_ref, op=op, right=values)

        value = self._expect("NAME").value
        return Comparison(left=field_ref, op=op, right=Literal(value))


def _walk_expr(expr):
    yield expr
    if isinstance(expr, (And, Or)):
        yield from _walk_expr(expr.left)
        yield from _walk_expr(expr.right)
    elif isinstance(expr, Not):
        yield from _walk_expr(expr.operand)


def validate_program(program: Program, task_spec: TaskSpec | None = None) -> None:
    """Validate a program and its optional task schema.

    Args:
        program: Parsed program to validate.
        task_spec: Optional field, enum, and label schema.
    """
    predicate_names = [p.name for p in program.predicates]
    rule_names = [r.name for r in program.rules]
    invariant_names = [i.name for i in program.invariants]
    for kind, names in (
        ("predicate", predicate_names),
        ("rule", rule_names),
        ("invariant", invariant_names),
    ):
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise DslSyntaxError(f"duplicate {kind} declaration(s): {', '.join(duplicates)}")

    declared_predicates = set(predicate_names)
    dependencies: dict[str, set[str]] = {name: set() for name in predicate_names}
    expressions = (
        [(f"predicate {p.name}", p.body, p.name) for p in program.predicates]
        + [(f"rule {r.name}", r.condition, None) for r in program.rules]
        + [(f"invariant {i.name}", i.condition, None) for i in program.invariants]
    )
    for owner, expr, predicate_owner in expressions:
        for node in _walk_expr(expr):
            if isinstance(node, PredicateCall):
                if node.name not in declared_predicates:
                    raise DslSyntaxError(f"{owner} calls undeclared predicate {node.name!r}")
                if predicate_owner is not None:
                    dependencies[predicate_owner].add(node.name)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise DslSyntaxError(f"recursive predicate cycle involving {name!r}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in dependencies[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in predicate_names:
        visit(name)

    for rule in program.rules:
        requires = {e.label for e in rule.effects if e.kind == "require"}
        forbids = {e.label for e in rule.effects if e.kind == "forbid"}
        if len(requires) > 1 or requires & forbids:
            raise DslSyntaxError(f"rule {rule.name!r} contains a static hard-effect conflict")

    if task_spec is None:
        return

    labels = set(task_spec.labels)
    enum_values = {enum.name: set(enum.values) for enum in task_spec.enums}
    records = {
        record.name.lower(): {field.name: field.type_name for field in record.fields}
        for record in task_spec.records
    }
    for owner, expr, _ in expressions:
        for node in _walk_expr(expr):
            if not isinstance(node, Comparison):
                continue
            root, field = node.left.field.split(".", 1)
            record_fields = records.get(root.lower())
            if record_fields is None or field not in record_fields:
                raise DslSyntaxError(f"{owner} references undeclared field {node.left.field!r}")
            type_name = record_fields[field]
            values = node.right if isinstance(node.right, list) else [node.right]
            if type_name in enum_values:
                invalid = [literal.value for literal in values if literal.value not in enum_values[type_name]]
                if invalid:
                    raise DslSyntaxError(
                        f"{owner} uses invalid {type_name} value(s): {', '.join(invalid)}"
                    )
            if type_name == "Boolean3":
                allowed = {"TRUE", "FALSE", "UNKNOWN"}
                invalid = [literal.value for literal in values if literal.value not in allowed]
                if invalid:
                    raise DslSyntaxError(
                        f"{owner} uses invalid Boolean3 value(s): {', '.join(invalid)}"
                    )

    for rule in program.rules:
        for effect in rule.effects:
            if effect.label is not None and effect.label not in labels:
                raise DslSyntaxError(
                    f"rule {rule.name!r} uses undeclared output label {effect.label!r}"
                )


def parse_program(source: str, task_spec: TaskSpec | None = None) -> Program:
    tokens = tokenize(source)
    program = Parser(tokens).parse_program()
    if task_spec is not None:
        validate_program(program, task_spec)
    return program


def _render_expr_source(expr) -> str:
    if isinstance(expr, Comparison):
        if isinstance(expr.right, list):
            right = "{" + ", ".join(literal.value for literal in expr.right) + "}"
        else:
            right = expr.right.value
        return f"{expr.left.field} {expr.op} {right}"
    if isinstance(expr, PredicateCall):
        return f"{expr.name}(event)"
    if isinstance(expr, Not):
        return f"not ({_render_expr_source(expr.operand)})"
    if isinstance(expr, And):
        return f"({_render_expr_source(expr.left)}) and ({_render_expr_source(expr.right)})"
    if isinstance(expr, Or):
        return f"({_render_expr_source(expr.left)}) or ({_render_expr_source(expr.right)})"
    raise ValueError(f"cannot render expression node: {expr!r}")


def render_program_source(program: Program) -> str:
    """Render a program syntax tree as DSL source."""
    blocks = []
    for predicate in program.predicates:
        parameter = predicate.parameter or "event"
        blocks.append(
            f"predicate {predicate.name}({parameter}):\n"
            f"    {_render_expr_source(predicate.body)}"
        )
    for rule in program.rules:
        effects = []
        for effect in rule.effects:
            if effect.kind == "abstain":
                effects.append("abstain")
            else:
                suffix = (
                    f" priority {effect.priority}" if effect.kind == "prefer" else ""
                )
                effects.append(f"{effect.kind} label = {effect.label}{suffix}")
        blocks.append(
            f"rule {rule.name}:\n"
            f"    when {_render_expr_source(rule.condition)}\n"
            f"    then {', '.join(effects)}"
        )
    for invariant in program.invariants:
        blocks.append(
            f"invariant {invariant.name}:\n"
            f"    {_render_expr_source(invariant.condition)}"
        )
    return ("\n\n".join(blocks) + "\n") if blocks else ""
