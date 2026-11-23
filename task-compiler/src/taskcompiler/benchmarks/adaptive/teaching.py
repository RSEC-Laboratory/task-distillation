"""Version-2 public teaching packs with larger, group-disjoint selection sets.

This module leaves the original benchmark teaching protocol unchanged. Schema and
coverage checks do not prove task semantics; the optional Boolean check only
covers recognisable expressions composed of literal booleans and and/or/not.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import random
import re

from taskcompiler.benchmarks import teaching as original
from taskcompiler.corrections.schema import identifiers, require, text
from taskcompiler.procedural.experiment import normalized_input_hash
from taskcompiler.procedural.schema import content_hash

DESCRIPTIONS = original.DESCRIPTIONS
IFEVAL_CONTRACT = original.IFEVAL_CONTRACT
PROTOCOL = "adaptive_components_v2"
parse_teacher_response = original.parse_teacher_response
all_cases = original.all_cases

SYSTEM = original.SYSTEM
for before, after in (
    ("groups: exactly EIGHT", "groups: exactly SIXTEEN"),
    ("FOUR independently varied", "EIGHT independently varied"),
    ("All 24 case IDs", "All 40 case IDs"),
    ("across all four groups", "across all eight groups"),
    ("each family's four groups", "each family's eight groups"),
):
    if before not in SYSTEM:
        raise RuntimeError("The inherited teaching protocol changed unexpectedly")
    SYSTEM = SYSTEM.replace(before, after)
SYSTEM += """
Difficulty diversity: vary the reasoning depth, distractors, boundary cases and
constraint interactions across ALL EIGHT groups of EACH coverage family. Do not
reserve easy or hard cases for a particular group number. Every group must be
self-contained and comparably representative, with both straightforward and
challenging instances distributed across the groups. Avoid ambiguous inputs.
The compiler assigns two groups per family to optimization, four to selection,
and two to an untouched synthetic holdout. The four demonstration pairs remain
separate. These counts total 32 practice cases plus 8 demonstration cases.
"""


def teacher_request(task_id: str, seed: int) -> tuple[str, str]:
    """Accept only a public task identifier and generation seed."""
    require(type(seed) is int, "Generation seed must be an integer")
    _, prompt = original.teacher_request(task_id, seed)
    return SYSTEM, prompt


def _groups(pack: dict):
    require(isinstance(pack, dict) and type(pack.get("version")) is int
            and pack["version"] == 2, "Expected a version-2 teaching pack")
    families = identifiers(pack["coverage_families"], "coverage families", 2)
    groups = identifiers(pack["groups"], "practice groups", 16)
    require(Counter(g.get("family_id") for g in pack["groups"])
            == Counter({family: 8 for family in families}),
            "Each family requires eight practice groups")
    for group in pack["groups"]:
        identifiers(group["cases"], "group cases", 2)
    return families, groups


def _canonical_boolean_answer(value: str) -> str | None:
    # A conservative recognizer: arbitrary natural-language Boolean questions
    # and other notations are intentionally not claimed to be verified.
    expression = value.rsplit(":", 1)[-1].strip()
    tokens = re.findall(r"True|False|and|or|not|[()]", expression)
    if not tokens or re.sub(r"True|False|and|or|not|[()]|\s", "", expression):
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("Malformed canonical Boolean practice expression") from exc

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not evaluate(node.operand)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [evaluate(child) for child in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise ValueError("Unsupported canonical Boolean practice expression")

    return str(evaluate(tree.body))


def validate_pack(pack: dict, evaluator) -> dict:
    families, groups = _groups(pack)
    # Reuse all established v1 checks on a valid-sized projection. Extra groups
    # are then checked below; neither candidates nor references are rewritten.
    projected_groups = []
    for family in pack["coverage_families"]:
        selected = sorted((g for g in pack["groups"] if g["family_id"] == family["id"]),
                          key=lambda group: group["id"])
        projected_groups.extend(selected[:4])
    original.validate_pack(dict(pack, version=1, groups=projected_groups), evaluator)
    require(pack.get("output_schema") == {}, "Native-text packs require an empty output schema")
    pairs = identifiers(pack["demo_pairs"], "demo pairs", 4)
    require(not groups & pairs, "Practice and demonstration group IDs overlap")
    cases = all_cases(pack)
    case_ids = identifiers(cases, "all generated cases", 40)
    require(not case_ids & (groups | pairs), "Case and group IDs must be globally unique")
    require(len({normalized_input_hash(case["input"]) for case in cases}) == 40,
            "Generated inputs overlap")
    task = pack["task_id"]
    for case in cases:
        text(case.get("input"), "input", 1200)
        text(case.get("reference"), "reference", 1200)
        require(case.get("task") == task and case.get("benchmark")
                == ("ifeval" if task == "ifeval" else "bbh"), "Practice task mismatch")
        evaluator.validate_practice_case(case)
        require(evaluator(case, case["reference"])["correct"],
                f"Generated reference fails its verifier: {case['id']}")
        if task == "boolean_expressions":
            expected = _canonical_boolean_answer(case["input"])
            require(expected is None or case["reference"] == expected,
                    f"Generated reference fails canonical Boolean truth check: {case['id']}")
    if task == "boolean_expressions":
        require(all({case["reference"] for case in group["cases"]} == {"True", "False"}
                    for group in pack["groups"]), "Boolean practice groups must contrast both outcomes")
    if task == "logical_deduction_three_objects":
        family_pairs = {pack["coverage_families"][0]["id"]: {"(A)", "(B)"},
                        pack["coverage_families"][1]["id"]: {"(B)", "(C)"}}
        require(all({case["reference"] for case in group["cases"]}
                    == family_pairs[group["family_id"]] for group in pack["groups"]),
                "Logical practice lacks fixed paired outcome coverage")
    require(pack["provenance"].get("teaching_protocol") == PROTOCOL,
            "Wrong teaching protocol provenance")
    return pack


def split_cases(pack: dict, seed: int = 0) -> dict:
    _groups(pack)
    require(type(seed) is int, "Split seed must be an integer")
    result = {"optimization": [], "selection": [], "synthetic_holdout": []}
    for family in pack["coverage_families"]:
        groups = sorted((g for g in pack["groups"] if g["family_id"] == family["id"]),
                        key=lambda group: group["id"])
        random.Random(content_hash([seed, family["id"]])).shuffle(groups)
        for phase, selected in (("optimization", groups[:2]), ("selection", groups[2:6]),
                                ("synthetic_holdout", groups[6:])):
            result[phase].extend(dict(case, group_id=group["id"])
                                 for group in selected for case in group["cases"])
    return result


def public_pack(body: dict, task_id: str, seed: int, teacher_model: str) -> dict:
    require(type(seed) is int, "Generation seed must be an integer")
    pack = original.public_pack(deepcopy(body), task_id, seed, teacher_model)
    pack["version"] = 2
    pack["provenance"]["teaching_protocol"] = PROTOCOL
    return pack
