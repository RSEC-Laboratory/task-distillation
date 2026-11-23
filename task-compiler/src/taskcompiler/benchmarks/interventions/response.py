"""Explicit response vectors, sparse edit projection, and control objectives.

These are development algorithms, not correctness certificates or a novelty
claim. A semantic-response objective is not identical to augmented accuracy:
shared output errors can cancel in a rule-change difference. Original-task
accuracy is therefore an independent guard, and both quantities are reported.
"""
from __future__ import annotations

from itertools import combinations
from taskcompiler.procedural.schema import strict_json_loads

from .program import RULES, variants, require


def parse_output(text, program):
    try:
        value = strict_json_loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {"slot_values"} or not isinstance(value["slot_values"], dict):
        return None
    slots = {s["name"] for s in program["slots"]}
    if not set(value["slot_values"]) <= slots:
        return None
    if not all(isinstance(v, list) and len(v) == 1 and all(isinstance(x, str) and x for x in v) for v in value["slot_values"].values()):
        return None
    return value


def atoms(output):
    if output is None:
        return {("__invalid__", "__invalid__")}
    return {(slot, value) for slot, values in output["slot_values"].items() for value in values}


def equal(left, right):
    return left is not None and atoms(left) == atoms(right)


def edges():
    settings = variants()
    result = []
    for left, a in settings.items():
        for right, b in settings.items():
            changed = [r for r in RULES if a[r] != b[r]]
            if left < right and len(changed) == 1:
                result.append((left, right, changed[0]))
    return result


def response_residual(predictions, references, group_ids):
    """Signed incidence changes, with a coordinate for invalid responses.

Coordinate universe includes predicted atoms, so hallucinated values contribute
to the loss rather than disappearing from the reference vocabulary.
"""
    result = {}
    for group in group_ids:
        for left, right, rule in edges():
            pa, pb = atoms(predictions[(group, left)]), atoms(predictions[(group, right)])
            ya, yb = atoms(references[(group, left)]), atoms(references[(group, right)])
            for atom in sorted(pa | pb | ya | yb):
                result[(group, left, right, rule, *atom)] = (int(atom in pb) - int(atom in pa)
                                                         - int(atom in yb) + int(atom in ya))
    return result


def squared_norm(vector):
    return sum(v * v for v in vector.values())


def diagnostics(predictions, references, group_ids):
    require(bool(group_ids), "No diagnostic groups")
    residual = response_residual(predictions, references, group_ids)
    original = sum(equal(predictions[(g, "o0s0")], references[(g, "o0s0")]) for g in group_ids)
    total = sum(equal(predictions[(g, state)], references[(g, state)]) for g in group_ids for state in variants())
    losses = {r: sum(v*v for key, v in residual.items() if key[3] == r) for r in RULES}
    return {"original_correct": original, "original_count": len(group_ids),
            "augmented_correct": total, "augmented_count": len(group_ids)*len(variants()),
            "response_squared_error": squared_norm(residual), "rule_errors": losses}


def projection(base_predictions, edit_predictions, references, group_ids, *, max_edits=2):
    """Choose a sparse edit set using measured single-edit response changes.

We minimize the predicted squared residual, then edit count and stable edit ID.
This is exact bounded subset enumeration, not a new combinatorial solver. Joint
effects are not assumed zero: the returned composition MUST be remeasured.
"""
    require(type(max_edits) is int and 0 <= max_edits <= len(edit_predictions), "Invalid edit budget")
    base = response_residual(base_predictions, references, group_ids)
    effects = {}
    universe = set(base)
    for key, predictions in edit_predictions.items():
        residual = response_residual(predictions, references, group_ids)
        universe |= set(residual)
        effects[key] = residual
    deltas = {key: {k: vec.get(k, 0)-base.get(k, 0) for k in universe} for key, vec in effects.items()}
    rows = []
    for size in range(max_edits+1):
        for selected in combinations(sorted(deltas), size):
            predicted = {k: base.get(k, 0)+sum(deltas[e][k] for e in selected) for k in universe}
            rows.append({"edits": list(selected), "predicted_error": squared_norm(predicted)})
    best = min(rows, key=lambda row: (row["predicted_error"], len(row["edits"]), row["edits"]))
    return {"selected": best["edits"], "predicted_error": best["predicted_error"],
            "base_error": squared_norm(base), "subsets": rows,
            "requires_joint_measurement": len(best["edits"]) > 1}


def select_candidate(predictions, references, group_ids, *, objective, baseline="teacher"):
    require(objective in {"original_accuracy", "augmented_accuracy", "response"}, "Unknown selection objective")
    reports = {key: diagnostics(values, references, group_ids) for key, values in predictions.items()}
    require(baseline in reports, "Missing protected baseline")
    protected = {g for g in group_ids if equal(predictions[baseline][(g, "o0s0")], references[(g, "o0s0")])}
    eligible = [key for key, values in predictions.items()
                if all(equal(values[(g, "o0s0")], references[(g, "o0s0")]) for g in protected)]
    def rank(key):
        row = reports[key]
        if objective == "original_accuracy":
            main = (-row["original_correct"], -row["augmented_correct"])
        elif objective == "augmented_accuracy":
            main = (-row["augmented_correct"], -row["original_correct"])
        else:
            main = (row["response_squared_error"], -row["original_correct"])
        return (*main, key != baseline, key)
    winner = min(eligible, key=rank)
    return {"selected": winner, "objective": objective, "scores": reports,
            "protected_original_groups": sorted(protected), "eligible": sorted(eligible)}
