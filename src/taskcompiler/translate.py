"""Render DSL rules and compile them into instruction bundles."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

from taskcompiler.dsl.ast import (
    And,
    Comparison,
    Effect,
    FieldRef,
    Literal,
    Not,
    Or,
    PredicateCall,
    Rule,
)
from taskcompiler.dsl.interpreter import run
from taskcompiler.student import StudentModel


EFFECT_PHRASES = {
    "require": "then it must be labeled",
    "forbid": "then it must not be labeled",
    "prefer": "then it should usually be labeled",
}


def _render_expr(expr) -> str:
    if isinstance(expr, Comparison):
        field = expr.left.field
        if expr.op == "==":
            return f"{field} is {expr.right.value}"
        if expr.op == "!=":
            return f"{field} is not {expr.right.value}"
        if expr.op == "in":
            values = ", ".join(v.value for v in expr.right)
            return f"{field} is one of ({values})"
    if isinstance(expr, PredicateCall):
        return expr.name
    if isinstance(expr, Not):
        return f"not ({_render_expr(expr.operand)})"
    if isinstance(expr, And):
        return f"({_render_expr(expr.left)}) and ({_render_expr(expr.right)})"
    if isinstance(expr, Or):
        return f"({_render_expr(expr.left)}) or ({_render_expr(expr.right)})"
    raise ValueError(f"cannot render expression: {expr!r}")


def render_rule(rule: Rule, style: str = "canonical") -> str:
    condition_text = _render_expr(rule.condition)
    effect_sentences = []
    for effect in rule.effects:
        if effect.kind == "abstain":
            effect_sentences.append("then return uncertain")
        else:
            phrase = EFFECT_PHRASES[effect.kind]
            priority = (
                f" with priority {effect.priority} (higher priorities win)"
                if effect.kind == "prefer"
                else ""
            )
            effect_sentences.append(f"{phrase} {effect.label}{priority}")
    effects_text = "; ".join(effect_sentences)
    if style == "explicit":
        return (
            f"Rule {rule.name}: Check whether {condition_text}. "
            f"Only when that complete condition is true, {effects_text}. "
            "If it is false or unknown, do not apply this rule."
        )
    if style != "canonical":
        raise ValueError(f"unknown rendering style: {style}")
    return f"If {condition_text}, {effects_text}."


def render_program(program) -> list[str]:
    """Render every decision-relevant part of a program."""
    definitions = [
        f"Definition {predicate.name}: {_render_expr(predicate.body)}."
        for predicate in program.predicates
    ]
    rules = [render_rule(rule) for rule in program.rules]
    resolution = [
        (
            "Resolution: apply only rules whose complete condition is true. "
            "A required label overrides preferences. Conflicting required labels, "
            "an explicit abstention, or an equal-priority preference tie returns "
            "uncertain. Ignore forbidden labels, then choose the remaining "
            "highest-priority preference. If nothing decides, return uncertain."
        )
    ]
    diagnostics = [
        (
            f"Diagnostic invariant {invariant.name}: "
            f"{_render_expr(invariant.condition)}. "
            "This is checked for validation and does not choose the output label."
        )
        for invariant in program.invariants
    ]
    return definitions + rules + resolution + diagnostics


def _expr_to_data(expr):
    if isinstance(expr, Comparison):
        right = (
            [literal.value for literal in expr.right]
            if isinstance(expr.right, list)
            else expr.right.value
        )
        return {"type": "comparison", "field": expr.left.field, "op": expr.op, "right": right}
    if isinstance(expr, PredicateCall):
        return {"type": "predicate", "name": expr.name}
    if isinstance(expr, Not):
        return {"type": "not", "operand": _expr_to_data(expr.operand)}
    if isinstance(expr, And):
        return {"type": "and", "left": _expr_to_data(expr.left), "right": _expr_to_data(expr.right)}
    if isinstance(expr, Or):
        return {"type": "or", "left": _expr_to_data(expr.left), "right": _expr_to_data(expr.right)}
    raise ValueError(f"cannot serialize expression: {expr!r}")


def _expr_from_data(data):
    kind = data["type"]
    if kind == "comparison":
        right = data["right"]
        literal = [Literal(value) for value in right] if isinstance(right, list) else Literal(right)
        return Comparison(FieldRef(data["field"]), data["op"], literal)
    if kind == "predicate":
        return PredicateCall(data["name"])
    if kind == "not":
        return Not(_expr_from_data(data["operand"]))
    if kind == "and":
        return And(_expr_from_data(data["left"]), _expr_from_data(data["right"]))
    if kind == "or":
        return Or(_expr_from_data(data["left"]), _expr_from_data(data["right"]))
    raise ValueError(f"unknown serialized expression type: {kind!r}")


def _rule_to_data(rule: Rule) -> dict:
    return {
        "name": rule.name,
        "source_clause": rule.source_clause,
        "condition": _expr_to_data(rule.condition),
        "effects": [asdict(effect) for effect in rule.effects],
    }


def _rule_from_data(data: dict) -> Rule:
    return Rule(
        name=data["name"],
        source_clause=data.get("source_clause"),
        condition=_expr_from_data(data["condition"]),
        effects=[Effect(**effect) for effect in data["effects"]],
    )


def roundtrip_check(
    rule: Rule, sample_inputs: list[dict], predicates: list | None = None
) -> bool:
    """Check that a rule survives serialization without behavior changes.

    Args:
        rule: Rule to serialize and recover.
        sample_inputs: Inputs used for the behavior check.
        predicates: Predicate declarations used by the rule.
    """
    payload = json.dumps(_rule_to_data(rule), sort_keys=True)
    recovered = _rule_from_data(json.loads(payload))
    if _rule_to_data(recovered) != _rule_to_data(rule):
        return False
    if render_rule(recovered) != render_rule(rule):
        return False
    if not sample_inputs:
        return True
    from taskcompiler.dsl.ast import Program

    original_program = Program(predicates or [], [rule], [])
    recovered_program = Program(copy.deepcopy(predicates or []), [recovered], [])
    for input_record in sample_inputs:
        left = run(original_program, input_record)
        right = run(recovered_program, input_record)
        if (
            left.output_label,
            left.conflict,
            left.abstained,
        ) != (
            right.output_label,
            right.conflict,
            right.abstained,
        ):
            return False
    return True


@dataclass
class InstructionEntry:
    rule: Rule
    style: str = "canonical"
    examples: list[tuple[dict, str | None]] = field(default_factory=list)

    @property
    def rule_name(self) -> str:
        return self.rule.name

    def render(self) -> str:
        text = render_rule(self.rule, self.style)
        if self.examples:
            examples = " ".join(
                f"Example input {record} => {label or 'uncertain'}."
                for record, label in self.examples
            )
            text = f"{text} {examples}"
        return text


@dataclass
class InstructionBundle:
    entries: list[InstructionEntry]
    prefixes: list[str] = field(default_factory=list)
    suffixes: list[str] = field(default_factory=list)
    override_texts: list[str] | None = None
    representation_kind: str = "rule_render"
    semantic_certificate: dict | None = None

    def render(self) -> list[str]:
        if self.override_texts is not None:
            return list(self.override_texts)
        return (
            list(self.prefixes)
            + [entry.render() for entry in self.entries]
            + list(self.suffixes)
        )

    def structural_signature(self) -> list[dict]:
        return [_rule_to_data(entry.rule) for entry in self.entries]

    def find(self, rule_name: str) -> InstructionEntry | None:
        return next((entry for entry in self.entries if entry.rule_name == rule_name), None)


@dataclass
class FreeTextBundle:
    """Unconstrained prompt artifact used only by the matched baseline."""

    texts: list[str]

    def render(self) -> list[str]:
        return list(self.texts)


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    rule_name: str
    kind: str
    hard: bool = True


@dataclass
class ProbeCase:
    probe_id: str
    input_record: dict
    expected_label: str | None
    obligations: tuple[str, ...]


@dataclass
class ProbeResult:
    probe_id: str
    expected_label: str | None
    student_label: str | None
    matched: bool
    obligations: tuple[str, ...]
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CompilationRound:
    round_index: int
    targeted_obligation: str
    accepted: bool
    objective_before: tuple
    objective_after: tuple
    edit: str
    protected_regressions: int = 0


@dataclass
class BehavioralCompilationResult:
    bundle: InstructionBundle
    build_pass_rate: float
    audit_pass_rate: float | None
    worst_obligation_violation: float
    rounds: list[CompilationRound]
    build_results: list[ProbeResult]
    audit_results: list[ProbeResult]
    initial_build_results: list[ProbeResult]
    student_input_tokens: int
    student_output_tokens: int
    student_calls: int


@dataclass
class GenericCompilationResult:
    bundle: FreeTextBundle
    build_pass_rate: float
    audit_pass_rate: float | None
    rounds: list[dict]
    student_input_tokens: int
    student_output_tokens: int
    teacher_input_tokens: int
    teacher_output_tokens: int
    student_calls: int
    teacher_calls: int


def normalise_student_label(text: str, labels: set[str]) -> str | None:
    """Extract one declared label from a model response."""
    candidate = text.strip().lower().strip("`'\".,:;()[]{} ")
    abstentions = {"uncertain", "unknown", "abstain", "none"}
    if candidate in abstentions:
        return None
    if candidate in labels:
        return candidate
    first = candidate.split()[0].strip("`'\".,:;()[]{} ") if candidate else ""
    if first in abstentions:
        return None
    if first in labels:
        return first

    final_choices = sorted(labels | abstentions, key=len, reverse=True)
    final_choice_pattern = re.compile(
        r"(?<![\w-])("
        + "|".join(re.escape(item) for item in final_choices)
        + r")(?![\w-])",
        re.IGNORECASE,
    )
    mention_choices = sorted(labels | {"uncertain"}, key=len, reverse=True)
    mention_pattern = re.compile(
        r"(?<![\w-])("
        + "|".join(re.escape(item) for item in mention_choices)
        + r")(?![\w-])",
        re.IGNORECASE,
    )
    final_marker = re.compile(
        r"\b(?:final|answer|therefore|thus|hence|conclusion|"
        r"classification|declared\s+label|(?:the\s+)?output)\b|"
        r"^\s*(?:label|prediction)\s*:",
        re.IGNORECASE,
    )
    for line in reversed(text.splitlines()):
        if not final_marker.search(line):
            continue
        matches = {
            match.casefold() for match in final_choice_pattern.findall(line)
        }
        if len(matches) == 1:
            selected = matches.pop()
            return None if selected in abstentions else selected
        if len(matches) > 1:
            return f"__INVALID__:{candidate[:80]}"

    matches = {match.casefold() for match in mention_pattern.findall(text)}
    if len(matches) == 1:
        selected = matches.pop()
        return None if selected in abstentions else selected
    return f"__INVALID__:{candidate[:80]}"


def probe_student(
    student: StudentModel,
    task_description: str,
    rendered_rules: list[str],
    probe: ProbeCase,
    labels: set[str],
    label_decoder: dict[str, str | None] | None = None,
) -> ProbeResult:
    rules_text = "\n".join(f"- {rule}" for rule in rendered_rules)
    output_contract = (
        "Output contract: return exactly one of "
        + ", ".join(sorted(label_decoder))
        + ". Do not include reasoning or formatting."
        if label_decoder is not None
        else (
            "Output contract: return exactly one declared label, or uncertain "
            "if the rules do not decide. Do not include reasoning or formatting."
        )
    )
    system = (
        f"Task: {task_description}\n\n"
        f"Rules:\n{rules_text}\n\n"
        f"{output_contract}"
    )
    response = student.ask(f"Input: {probe.input_record}", system=system)
    response_labels = (
        set(label_decoder)
        if label_decoder is not None
        else labels
    )
    predicted = normalise_student_label(response.text, response_labels)
    if label_decoder is not None and predicted is not None:
        predicted = label_decoder.get(
            predicted,
            f"__INVALID__:{predicted}",
        )
    return ProbeResult(
        probe_id=probe.probe_id,
        expected_label=probe.expected_label,
        student_label=predicted,
        matched=predicted == probe.expected_label,
        obligations=probe.obligations,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def build_obligation_probes(
    program,
    inputs: list[dict],
    split_name: str,
    *,
    oracle_program=None,
    causal_only: bool = False,
) -> tuple[list[Obligation], list[ProbeCase]]:
    """Build probes using an optional independent source-grounded oracle."""
    obligations: dict[str, Obligation] = {}
    probes: list[ProbeCase] = []
    for index, input_record in enumerate(inputs):
        trace = run(oracle_program or program, input_record)
        ids: list[str] = []
        for rule in program.rules:
            if causal_only and rule.name not in trace.fired_rules:
                continue
            value = trace.rule_conditions[rule.name].value.lower()
            obligation_id = f"{rule.name}:condition:{value}"
            obligations.setdefault(
                obligation_id,
                Obligation(obligation_id, rule.name, f"condition_{value}"),
            )
            ids.append(obligation_id)
            if rule.name in trace.fired_rules:
                for effect in rule.effects:
                    effect_id = f"{rule.name}:effect:{effect.kind}:{effect.label or 'none'}"
                    obligations.setdefault(
                        effect_id,
                        Obligation(effect_id, rule.name, f"effect_{effect.kind}"),
                    )
                    ids.append(effect_id)
        probes.append(
            ProbeCase(
                probe_id=f"{split_name}-{index:04d}",
                input_record=copy.deepcopy(input_record),
                expected_label=trace.output_label,
                obligations=tuple(ids),
            )
        )
    return list(obligations.values()), probes


def _evaluate_bundle(
    student: StudentModel,
    task_description: str,
    bundle: InstructionBundle,
    probes: list[ProbeCase],
    labels: set[str],
    render_for_probe: Callable[[InstructionBundle, ProbeCase], list[str]]
    | None = None,
    label_decoder: dict[str, str | None] | None = None,
) -> list[ProbeResult]:
    return [
        probe_student(
            student,
            task_description,
            (
                render_for_probe(bundle, probe)
                if render_for_probe is not None
                else bundle.render()
            ),
            probe,
            labels,
            label_decoder,
        )
        for probe in probes
    ]


def _objective(
    results: list[ProbeResult],
    obligations: list[Obligation],
    bundle: InstructionBundle,
    mode: str,
) -> tuple:
    rates = {}
    for obligation in obligations:
        relevant = [result for result in results if obligation.obligation_id in result.obligations]
        if relevant:
            rates[obligation.obligation_id] = 1 - sum(r.matched for r in relevant) / len(relevant)
    aggregate = 1 - (sum(result.matched for result in results) / len(results) if results else 1.0)
    length = sum(len(text.split()) for text in bundle.render())
    if mode == "test_driven":
        return (aggregate, length)
    worst = max(rates.values(), default=0.0)
    weighted = sum(rates.values())
    false_confidence = sum(
        1
        for result in results
        if result.expected_label is None and result.student_label is not None
    )
    return (aggregate, worst, weighted, false_confidence, length)


def _worst_obligation(
    results: list[ProbeResult], obligations: list[Obligation]
) -> Obligation | None:
    scored = []
    for obligation in obligations:
        relevant = [result for result in results if obligation.obligation_id in result.obligations]
        if relevant:
            violation = 1 - sum(r.matched for r in relevant) / len(relevant)
            scored.append((violation, obligation.obligation_id, obligation))
    return max(scored, default=(0.0, "", None))[2]


def compile_verified_decision_list(
    program,
    build_probes: list[ProbeCase],
) -> list[str] | None:
    """Compile a small finite policy domain into label-pure rules.

    Args:
        program: Verified DSL program that defines the policy.
        build_probes: Public probes used to identify finite field domains.
    """
    domain = _finite_probe_domain(build_probes)
    if domain is None:
        return None

    paths, domains = domain
    outcomes = {}
    for values in itertools.product(*domains):
        record = _domain_record(paths, values)
        outcomes[values] = run(program, record).output_label or "uncertain"

    decision_rules = []
    for output in sorted(set(outcomes.values())):
        uncovered = {
            values for values, label in outcomes.items() if label == output
        }
        pure_patterns = {}
        for values in uncovered:
            for wildcard_mask in range(1 << len(paths)):
                pattern = tuple(
                    None if wildcard_mask & (1 << index) else value
                    for index, value in enumerate(values)
                )
                matches = {
                    candidate
                    for candidate in outcomes
                    if all(
                        expected is None or actual == expected
                        for actual, expected in zip(candidate, pattern)
                    )
                }
                if matches and {
                    outcomes[candidate] for candidate in matches
                } == {output}:
                    pure_patterns[pattern] = matches
        while uncovered:
            pattern, coverage = max(
                pure_patterns.items(),
                key=lambda item: (
                    len(item[1] & uncovered),
                    -sum(value is not None for value in item[0]),
                    tuple(value or "" for value in item[0]),
                ),
            )
            conditions = [
                f"{path.split('.', 1)[1]}={value}"
                for path, value in zip(paths, pattern)
                if value is not None
            ]
            decision_rules.append(
                f"If {', '.join(conditions) or 'any input'}, return {output}."
            )
            uncovered -= coverage
    return [
        (
            "Verified decision list. Find a line whose stated field conditions "
            "all match the input, then return its label. Unstated fields can "
            "have any value. Matching lines agree."
        ),
        *decision_rules,
    ]


def _finite_probe_domain(
    build_probes: list[ProbeCase],
) -> tuple[list[str], list[list[str]]] | None:
    """Read a small finite field domain from public probes.

    Args:
        build_probes: Public probes that declare observed field values.
    """
    flattened: dict[str, set[str]] = {}
    for probe in build_probes:
        for root, fields in probe.input_record.items():
            if root.startswith("_") or not isinstance(fields, dict):
                continue
            for field, value in fields.items():
                flattened.setdefault(f"{root}.{field}", set()).add(str(value))
    paths = list(flattened)
    domains = [sorted(flattened[path]) for path in paths]
    domain_size = 1
    for values in domains:
        domain_size *= len(values)
    if not (
        len(paths) >= 3
        and 16 <= domain_size <= 128
        and all(len(values) >= 2 for values in domains)
    ):
        return None
    return paths, domains


def _domain_record(paths: list[str], values: tuple[str, ...]) -> dict:
    record: dict[str, dict[str, str]] = {}
    for path, value in zip(paths, values):
        root, field = path.split(".", 1)
        record.setdefault(root, {})[field] = value
    return record


@dataclass(frozen=True)
class _DecisionLeaf:
    label: str


@dataclass(frozen=True)
class _DecisionSplit:
    path: str
    branches: tuple[tuple[str, object], ...]


def _tree_size(node) -> tuple[int, int, int]:
    if isinstance(node, _DecisionLeaf):
        return 1, 0, 0
    child_sizes = [_tree_size(child) for _, child in node.branches]
    return (
        sum(size[0] for size in child_sizes),
        1 + sum(size[1] for size in child_sizes),
        1 + max(size[2] for size in child_sizes),
    )


def compile_verified_decision_tree(
    program,
    build_probes: list[ProbeCase],
    *,
    reverse_leaves: bool = False,
) -> list[str] | None:
    """Compile a finite policy into disjoint decision-tree leaves.

    Args:
        program: Verified DSL program that defines the policy.
        build_probes: Public probes that declare the finite field domain.
        reverse_leaves: Reverse the leaf order without changing semantics.
    """
    domain = _finite_probe_domain(build_probes)
    if domain is None:
        return None
    paths, domains = domain
    rows = []
    for values in itertools.product(*domains):
        label = run(program, _domain_record(paths, values)).output_label
        rows.append((values, label or "uncertain"))

    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], object] = {}

    def build(row_ids: tuple[int, ...], fields: tuple[int, ...]):
        key = row_ids, fields
        if key in cache:
            return cache[key]
        labels = {rows[row_id][1] for row_id in row_ids}
        if len(labels) == 1:
            node = _DecisionLeaf(next(iter(labels)))
            cache[key] = node
            return node
        if not fields:
            raise ValueError("finite policy cannot be separated by declared fields")

        candidates = []
        for field_index in fields:
            remaining = tuple(index for index in fields if index != field_index)
            branches = []
            for value in domains[field_index]:
                members = tuple(
                    row_id
                    for row_id in row_ids
                    if rows[row_id][0][field_index] == value
                )
                if members:
                    branches.append((value, build(members, remaining)))
            node = _DecisionSplit(paths[field_index], tuple(branches))
            leaves, splits, depth = _tree_size(node)
            candidates.append(((leaves, depth, splits, paths[field_index]), node))
        node = min(candidates, key=lambda item: item[0])[1]
        cache[key] = node
        return node

    tree = build(tuple(range(len(rows))), tuple(range(len(paths))))
    leaves: list[tuple[tuple[tuple[str, str], ...], str]] = []

    def collect(node, conditions: tuple[tuple[str, str], ...] = ()) -> None:
        if isinstance(node, _DecisionLeaf):
            leaves.append((conditions, node.label))
            return
        for value, child in node.branches:
            collect(child, conditions + ((node.path, value),))

    collect(tree)
    if reverse_leaves:
        leaves.reverse()
    lines = [
        (
            "Verified decision tree. Match the one leaf whose field conditions "
            "all hold, then return its label. Every declared input matches "
            "exactly one leaf."
        )
    ]
    for conditions, label in leaves:
        condition_text = ", ".join(
            f"{path.split('.', 1)[1]}={value}" for path, value in conditions
        )
        lines.append(f"If {condition_text or 'any input'}, return {label}.")
    return lines


def certify_verified_decision_list(
    program,
    build_probes: list[ProbeCase],
    decision_list: list[str],
) -> dict:
    """Check a compiled decision list over its complete finite domain.

    Args:
        program: Executable policy used as the semantic reference.
        build_probes: Public probes that declare each field's finite values.
        decision_list: Natural-language decision rules to check.
    """
    domain = _finite_probe_domain(build_probes)
    if domain is None:
        return {
            "finite_domain_size": 0,
            "decision_rule_count": 0,
            "exhaustive_coverage": False,
            "label_pure": False,
            "semantically_equivalent": False,
            "representation_sha256": hashlib.sha256(
                json.dumps(decision_list, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "oracle_outcomes_sha256": None,
        }
    paths, domains = domain
    suffixes = [path.split(".", 1)[1] for path in paths]
    field_index = {path: index for index, path in enumerate(paths)}
    field_index.update({
        suffix: index
        for index, suffix in enumerate(suffixes)
        if suffixes.count(suffix) == 1
    })

    rules = []
    for line in decision_list[1:]:
        if not line.startswith("If ") or ", return " not in line:
            continue
        conditions_text, label = line.removeprefix("If ").rsplit(
            ", return ",
            1,
        )
        conditions = {}
        if conditions_text != "any input":
            for condition in conditions_text.split(", "):
                field, value = condition.split("=", 1)
                conditions[field] = value
        rules.append((conditions, label.removesuffix(".")))

    rows = []
    exhaustive = True
    label_pure = True
    equivalent = True
    for values in itertools.product(*domains):
        record = _domain_record(paths, values)
        expected = run(program, record).output_label or "uncertain"
        matches = []
        for conditions, label in rules:
            if all(
                field in field_index
                and values[field_index[field]] == value
                for field, value in conditions.items()
            ):
                matches.append(label)
        exhaustive &= bool(matches)
        label_pure &= len(set(matches)) <= 1
        equivalent &= bool(matches) and set(matches) == {expected}
        rows.append({"values": list(values), "output": expected})

    return {
        "finite_domain_size": len(rows),
        "decision_rule_count": len(rules),
        "exhaustive_coverage": exhaustive,
        "label_pure": label_pure,
        "semantically_equivalent": equivalent,
        "representation_sha256": hashlib.sha256(
            json.dumps(decision_list, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "oracle_outcomes_sha256": hashlib.sha256(
            json.dumps(
                {"paths": paths, "rows": rows},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _candidate_edits(
    bundle: InstructionBundle,
    target: Obligation,
    diagnostic: ProbeCase | None,
    *,
    program=None,
    build_probes: list[ProbeCase] | None = None,
    results: list[ProbeResult] | None = None,
) -> list[tuple[str, InstructionBundle]]:
    candidates = []

    def append_text(candidate: InstructionBundle, text: str) -> None:
        if candidate.override_texts is not None:
            candidate.override_texts.append(text)
        else:
            candidate.suffixes.append(text)

    def distinct() -> list[tuple[str, InstructionBundle]]:
        seen = {tuple(bundle.render())}
        unique = []
        for name, candidate in candidates:
            rendered = tuple(candidate.render())
            if rendered in seen:
                continue
            seen.add(rendered)
            unique.append((name, candidate))
        return unique

    if program is not None and build_probes:
        decision_list = compile_verified_decision_list(program, build_probes)
        if decision_list is not None:
            certificate = certify_verified_decision_list(
                program, build_probes, decision_list
            )
            table_candidate = copy.deepcopy(bundle)
            table_candidate.override_texts = decision_list
            table_candidate.representation_kind = "verified_decision_list"
            table_candidate.semantic_certificate = certificate
            if certificate["semantically_equivalent"]:
                candidates.append(
                    ("render:verified_decision_list", table_candidate)
                )

    checklist_text = (
        "Execution checklist: First evaluate every complete rule condition. "
        "If any true rule abstains, return uncertain. Otherwise collect required "
        "and forbidden labels. Conflicting required labels, or a required label "
        "that is forbidden, returns uncertain; one valid required label wins. "
        "If there is no required label, discard forbidden preferences and choose "
        "the unique highest-priority preference. A top-priority tie or no "
        "remaining preference returns uncertain."
    )
    if checklist_text not in bundle.render():
        checklist = copy.deepcopy(bundle)
        append_text(checklist, checklist_text)
        candidates.append(("resolver:execution_checklist", checklist))

    if program is not None and build_probes and results:
        probes_by_id = {probe.probe_id: probe for probe in build_probes}
        failed = [
            probes_by_id[result.probe_id]
            for result in results
            if not result.matched and result.probe_id in probes_by_id
        ]
        selected = []
        if diagnostic is not None:
            selected.append(diagnostic)
        seen_labels = {diagnostic.expected_label if diagnostic else object()}
        for probe in failed:
            if probe.expected_label not in seen_labels:
                selected.append(probe)
                seen_labels.add(probe.expected_label)
            if len(selected) >= 4:
                break
        for probe in failed:
            if probe not in selected:
                selected.append(probe)
            if len(selected) >= 4:
                break

        trace_texts = []
        for probe in selected:
            trace = run(program, probe.input_record)
            preferred = ", ".join(
                f"{label}@{priority}" for label, priority in trace.preferred
            ) or "none"
            trace_texts.append(
                "Verified trace: input "
                f"{json.dumps(probe.input_record, sort_keys=True, separators=(',', ':'))}; "
                f"true rules {trace.fired_rules or ['none']}; "
                f"required {trace.required_labels or ['none']}; "
                f"forbidden {trace.forbidden_labels or ['none']}; "
                f"preferences {preferred}; output "
                f"{probe.expected_label or 'uncertain'}."
            )
        if trace_texts:
            traces = copy.deepcopy(bundle)
            for text in trace_texts:
                append_text(traces, text)
            candidates.append(("examples:obligation_balanced_traces", traces))

            combined = copy.deepcopy(bundle)
            if checklist_text not in combined.render():
                append_text(combined, checklist_text)
            for text in trace_texts:
                append_text(combined, text)
            candidates.append(("resolver:checklist_and_traces", combined))

    position = next(
        (index for index, entry in enumerate(bundle.entries) if entry.rule_name == target.rule_name),
        None,
    )
    if position is None:
        return distinct()

    moved = copy.deepcopy(bundle)
    entry = moved.entries.pop(position)
    moved.entries.insert(0, entry)
    candidates.append((f"move:{target.rule_name}:front", moved))

    explicit = copy.deepcopy(bundle)
    explicit_entry = explicit.find(target.rule_name)
    explicit_entry.style = "explicit"
    candidates.append((f"style:{target.rule_name}:explicit", explicit))

    if diagnostic is not None:
        example = copy.deepcopy(bundle)
        example_entry = example.find(target.rule_name)
        pair = (copy.deepcopy(diagnostic.input_record), diagnostic.expected_label)
        if pair not in example_entry.examples:
            example_entry.examples.append(pair)
            candidates.append((f"example:{target.rule_name}:{diagnostic.probe_id}", example))
    return distinct()


def _certified_representation_candidates(
    bundle: InstructionBundle,
    program,
    build_probes: list[ProbeCase],
) -> list[tuple[str, InstructionBundle]]:
    """Generate machine-checked representations for student selection."""
    generated = [
        (
            "certified:decision_list",
            "verified_decision_list",
            compile_verified_decision_list(program, build_probes),
        ),
        (
            "certified:decision_tree",
            "verified_decision_tree",
            compile_verified_decision_tree(program, build_probes),
        ),
        (
            "certified:decision_tree_reversed",
            "verified_decision_tree_reversed",
            compile_verified_decision_tree(
                program, build_probes, reverse_leaves=True
            ),
        ),
    ]
    candidates = []
    seen = set()
    for name, kind, texts in generated:
        if texts is None or tuple(texts) in seen:
            continue
        certificate = certify_verified_decision_list(
            program, build_probes, texts
        )
        if not (
            certificate["exhaustive_coverage"]
            and certificate["label_pure"]
            and certificate["semantically_equivalent"]
        ):
            continue
        candidate = copy.deepcopy(bundle)
        candidate.override_texts = texts
        candidate.representation_kind = kind
        candidate.semantic_certificate = {
            **certificate,
            "representation_kind": kind,
        }
        candidates.append((name, candidate))
        seen.add(tuple(texts))
    return candidates


def behavioral_compile(
    student: StudentModel,
    task_description: str,
    program,
    build_probes: list[ProbeCase],
    audit_probes: list[ProbeCase] | None = None,
    *,
    max_rounds: int = 3,
    token_budget: int = 800,
    mode: str = "obligation_balanced",
    max_student_calls: int | None = None,
    max_protected_regressions: int = 0,
    render_for_probe: Callable[[InstructionBundle, ProbeCase], list[str]]
    | None = None,
    label_decoder: dict[str, str | None] | None = None,
) -> BehavioralCompilationResult:
    """Compile a typed instruction bundle against public probes.

    Args:
        student: Frozen model used to evaluate bundles.
        task_description: Public description shown to the model.
        program: DSL program to compile.
        build_probes: Probes used during compilation.
        audit_probes: Optional held-out probes.
        max_rounds: Maximum compilation rounds.
        token_budget: Maximum bundle size in tokens.
        mode: Compilation objective to optimize.
        max_student_calls: Optional cap on student probe evaluations.
        max_protected_regressions: Passing probes a better candidate may lose.
        render_for_probe: Optional local routing function used for each probe.
        label_decoder: Optional local map from neutral codes to task labels.
    """
    if mode not in {
        "obligation_balanced",
        "test_driven",
        "certified_search",
    }:
        raise ValueError(f"unknown compilation mode: {mode}")
    if max_protected_regressions < 0:
        raise ValueError("max_protected_regressions must be non-negative")
    labels = {
        effect.label.lower()
        for rule in program.rules
        for effect in rule.effects
        if effect.label is not None
    }
    obligations_by_id = {}
    for probe in build_probes:
        for obligation_id in probe.obligations:
            parts = obligation_id.split(":")
            obligations_by_id.setdefault(
                obligation_id,
                Obligation(obligation_id, parts[0], ":".join(parts[1:])),
            )
    obligations = list(obligations_by_id.values())
    rendered = render_program(program)
    prefix_count = len(program.predicates)
    suffix_start = prefix_count + len(program.rules)
    bundle = InstructionBundle(
        [InstructionEntry(rule=copy.deepcopy(rule)) for rule in program.rules],
        prefixes=rendered[:prefix_count],
        suffixes=rendered[suffix_start:],
    )
    for entry in bundle.entries:
        if not roundtrip_check(
            entry.rule,
            [probe.input_record for probe in build_probes],
            program.predicates,
        ):
            raise ValueError(f"structured round-trip failed for rule {entry.rule_name}")

    results = _evaluate_bundle(
        student,
        task_description,
        bundle,
        build_probes,
        labels,
        render_for_probe,
        label_decoder,
    )
    initial_results = copy.deepcopy(results)
    all_evaluations = list(results)
    objective = _objective(results, obligations, bundle, mode)
    protected = {result.probe_id for result in results if result.matched}
    rounds: list[CompilationRound] = []

    if mode == "certified_search":
        evaluated = []
        candidates = _certified_representation_candidates(
            bundle, program, build_probes
        )
        for round_index, (edit_name, candidate) in enumerate(
            candidates[:max_rounds]
        ):
            if sum(len(text.split()) for text in candidate.render()) > token_budget:
                continue
            if (
                max_student_calls is not None
                and len(all_evaluations) + len(build_probes) > max_student_calls
            ):
                break
            candidate_results = _evaluate_bundle(
                student,
                task_description,
                candidate,
                build_probes,
                labels,
                render_for_probe,
                label_decoder,
            )
            all_evaluations.extend(candidate_results)
            candidate_pairs = {
                result.probe_id
                for result in candidate_results
                if result.matched
            }
            candidate_objective = _objective(
                candidate_results, obligations, candidate, mode
            )
            regression_count = len(protected - candidate_pairs)
            evaluated.append(
                (
                    candidate_objective,
                    edit_name,
                    candidate,
                    candidate_results,
                    regression_count,
                    round_index,
                )
            )
        admissible = [
            item
            for item in evaluated
            if item[4] <= max_protected_regressions
        ]
        if not evaluated:
            raise RuntimeError(
                "no certified representation fit the compilation budget"
            )
        selected = min(
            admissible or evaluated,
            key=lambda item: item[0],
        )
        for (
            candidate_objective,
            edit_name,
            candidate,
            candidate_results,
            regression_count,
            round_index,
        ) in evaluated:
            rounds.append(
                CompilationRound(
                    round_index,
                    "all:semantic_equivalence",
                    edit_name == selected[1],
                    objective,
                    candidate_objective,
                    edit_name,
                    regression_count,
                )
            )
        objective, _, bundle, results, _, _ = selected
    else:
        for round_index in range(max_rounds):
            target = _worst_obligation(results, obligations)
            if target is None or objective[0] == 0:
                break
            diagnostic_result = next(
                (
                    result
                    for result in results
                    if not result.matched
                    and target.obligation_id in result.obligations
                ),
                None,
            )
            diagnostic = next(
                (
                    probe
                    for probe in build_probes
                    if diagnostic_result
                    and probe.probe_id == diagnostic_result.probe_id
                ),
                None,
            )
            accepted = False
            budget_exhausted = False
            for edit_name, candidate in _candidate_edits(
                bundle,
                target,
                diagnostic,
                program=program,
                build_probes=build_probes,
                results=results,
            ):
                if (
                    sum(len(text.split()) for text in candidate.render())
                    > token_budget
                ):
                    continue
                if (
                    candidate.structural_signature()
                    != bundle.structural_signature()
                    and sorted(
                        candidate.structural_signature(),
                        key=lambda item: item["name"],
                    )
                    != sorted(
                        bundle.structural_signature(),
                        key=lambda item: item["name"],
                    )
                ):
                    continue
                if (
                    max_student_calls is not None
                    and len(all_evaluations) + len(build_probes)
                    > max_student_calls
                ):
                    budget_exhausted = True
                    break
                candidate_results = _evaluate_bundle(
                    student,
                    task_description,
                    candidate,
                    build_probes,
                    labels,
                    render_for_probe,
                    label_decoder,
                )
                all_evaluations.extend(candidate_results)
                candidate_pairs = {
                    result.probe_id
                    for result in candidate_results
                    if result.matched
                }
                candidate_objective = _objective(
                    candidate_results, obligations, candidate, mode
                )
                regression_count = len(protected - candidate_pairs)
                preserves_regressions = (
                    regression_count <= max_protected_regressions
                )
                is_better = candidate_objective < objective
                rounds.append(
                    CompilationRound(
                        round_index,
                        target.obligation_id,
                        preserves_regressions and is_better,
                        objective,
                        candidate_objective,
                        edit_name,
                        regression_count,
                    )
                )
                if preserves_regressions and is_better:
                    bundle = candidate
                    results = candidate_results
                    objective = candidate_objective
                    protected |= candidate_pairs
                    accepted = True
                    break
            if not accepted:
                break
            if budget_exhausted:
                break

    audit_results = (
        _evaluate_bundle(
            student,
            task_description,
            bundle,
            audit_probes,
            labels,
            render_for_probe,
            label_decoder,
        )
        if audit_probes
        else []
    )
    all_evaluations.extend(audit_results)
    pass_rate = sum(result.matched for result in results) / len(results) if results else 1.0
    audit_rate = (
        sum(result.matched for result in audit_results) / len(audit_results)
        if audit_results
        else None
    )
    full_objective = _objective(results, obligations, bundle, "obligation_balanced")
    return BehavioralCompilationResult(
        bundle=bundle,
        build_pass_rate=pass_rate,
        audit_pass_rate=audit_rate,
        worst_obligation_violation=full_objective[1],
        rounds=rounds,
        build_results=results,
        audit_results=audit_results,
        initial_build_results=initial_results,
        student_input_tokens=sum(result.input_tokens for result in all_evaluations),
        student_output_tokens=sum(result.output_tokens for result in all_evaluations),
        student_calls=len(all_evaluations),
    )


GENERIC_TEST_DRIVEN_SYSTEM = (
    "You are a test-driven prompt compiler. Rewrite the complete instruction "
    "bundle so that a frozen model satisfies the visible behavioral tests. "
    "You may freely change wording, structure, and ordering, but must preserve "
    "the public task specification. Return only the complete instruction "
    "bundle as one instruction per line, with no markdown fences."
)


def _parse_instruction_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        if line[:2] in {"- ", "* "}:
            line = line[2:].strip()
        lines.append(line)
    return lines


def generic_test_driven_compile(
    student: StudentModel,
    teacher,
    task_description: str,
    initial_rules: list[str],
    build_probes: list[ProbeCase],
    audit_probes: list[ProbeCase],
    labels: set[str],
    *,
    max_rounds: int = 3,
    token_budget: int = 800,
    max_student_calls: int | None = None,
    max_protected_regressions: int = 0,
) -> GenericCompilationResult:
    """Compile free-form instructions against visible tests.

    Args:
        student: Frozen model used to evaluate instructions.
        teacher: Model used to rewrite instructions.
        task_description: Public description shown to both models.
        initial_rules: Instructions used as the starting point.
        build_probes: Visible probes used during compilation.
        audit_probes: Held-out probes used after compilation.
        labels: Valid output labels.
        max_rounds: Maximum compilation rounds.
        token_budget: Maximum instruction size in tokens.
        max_student_calls: Optional cap on student probe evaluations.
        max_protected_regressions: Passing probes a better candidate may lose.
    """
    if max_protected_regressions < 0:
        raise ValueError("max_protected_regressions must be non-negative")
    bundle = FreeTextBundle(list(initial_rules))
    results = [
        probe_student(student, task_description, bundle.render(), probe, labels)
        for probe in build_probes
    ]
    all_results = list(results)
    protected = {result.probe_id for result in results if result.matched}
    rounds = []
    teacher_input = 0
    teacher_output = 0
    teacher_calls = 0
    rejected_rewrites: list[list[str]] = []

    for round_index in range(max_rounds):
        failures = [result for result in results if not result.matched]
        if not failures:
            break
        visible = []
        for failure in failures:
            probe = next(
                probe for probe in build_probes if probe.probe_id == failure.probe_id
            )
            visible.append(
                {
                    "probe_id": probe.probe_id,
                    "input": probe.input_record,
                    "expected": probe.expected_label or "uncertain",
                    "student_output": failure.student_label,
                }
            )
        prompt = (
            f"Public task:\n{task_description}\n\n"
            f"Current instruction bundle:\n{json.dumps(bundle.render(), indent=2)}\n\n"
            f"Visible failing tests:\n{json.dumps(visible, indent=2)}"
        )
        if rejected_rewrites:
            prompt += (
                "\n\nPreviously rejected rewrites:\n"
                f"{json.dumps(rejected_rewrites, indent=2)}\n"
                "Propose a materially different complete instruction bundle."
            )
        response = teacher.call(GENERIC_TEST_DRIVEN_SYSTEM, prompt)
        teacher_calls += 1
        teacher_input += response.input_tokens
        teacher_output += response.output_tokens
        candidate_texts = _parse_instruction_lines(response.text)
        if not candidate_texts:
            rounds.append({"round": round_index, "accepted": False, "reason": "empty"})
            break
        if sum(len(text.split()) for text in candidate_texts) > token_budget:
            rounds.append(
                {"round": round_index, "accepted": False, "reason": "over_budget"}
            )
            break
        candidate = FreeTextBundle(candidate_texts)
        if (
            max_student_calls is not None
            and len(all_results) + len(build_probes) > max_student_calls
        ):
            rounds.append(
                {
                    "round": round_index,
                    "accepted": False,
                    "reason": "student_call_budget_exhausted",
                }
            )
            break
        candidate_results = [
            probe_student(
                student, task_description, candidate.render(), probe, labels
            )
            for probe in build_probes
        ]
        all_results.extend(candidate_results)
        candidate_passed = {
            result.probe_id for result in candidate_results if result.matched
        }
        old_rate = sum(result.matched for result in results) / len(results)
        new_rate = sum(result.matched for result in candidate_results) / len(
            candidate_results
        )
        regressions = len(protected - candidate_passed)
        accepted = (
            regressions <= max_protected_regressions
            and new_rate > old_rate
        )
        rounds.append(
            {
                "round": round_index,
                "accepted": accepted,
                "old_pass_rate": old_rate,
                "new_pass_rate": new_rate,
                "protected_regressions": regressions,
            }
        )
        if not accepted:
            rejected_rewrites.append(candidate_texts)
            continue
        bundle = candidate
        results = candidate_results
        protected |= candidate_passed

    audit_results = [
        probe_student(student, task_description, bundle.render(), probe, labels)
        for probe in audit_probes
    ]
    all_results.extend(audit_results)
    build_rate = sum(result.matched for result in results) / len(results)
    audit_rate = (
        sum(result.matched for result in audit_results) / len(audit_results)
        if audit_results
        else None
    )
    return GenericCompilationResult(
        bundle=bundle,
        build_pass_rate=build_rate,
        audit_pass_rate=audit_rate,
        rounds=rounds,
        student_input_tokens=sum(result.input_tokens for result in all_results),
        student_output_tokens=sum(result.output_tokens for result in all_results),
        teacher_input_tokens=teacher_input,
        teacher_output_tokens=teacher_output,
        student_calls=len(all_results),
        teacher_calls=teacher_calls,
    )
