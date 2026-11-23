"""Prospective, finite four-clause response models; no model or dataset access.

The eleven degree <= 2 observations identify finite differences, not a general
response law. Five configurations test the frozen predictions on the SAME
explicit dialogue/edge scope. Opening that holdout never changes a selection.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import combinations, product
import json
import math

CONFIGURATIONS = tuple("".join(bits) for bits in product("01", repeat=4))
FIT_CONFIGURATIONS = tuple(c for c in CONFIGURATIONS if c.count("1") <= 2)
HELDOUT_CONFIGURATIONS = tuple(c for c in CONFIGURATIONS if c.count("1") > 2)
METHODS = ("linear_vector", "second_order_vector", "scalar_quadratic")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, *, low=None, high=None):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, f"{name} must be finite numeric, excluding bool")
    _require(low is None or value >= low, f"{name} below its lower bound")
    _require(high is None or value <= high, f"{name} above its upper bound")
    return float(value)


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()) and len(value) <= 100000,
             f"{name} requires bounded nonempty text")
    return value


def _json_check(value):
    if isinstance(value, dict):
        _require(all(isinstance(key, str) for key in value), "JSON keys must be strings")
        for item in value.values():
            _json_check(item)
    elif isinstance(value, list):
        for item in value:
            _json_check(item)
    elif type(value) in (int, float):
        _number(value, "JSON number")
    else:
        _require(value is None or type(value) in (str, bool), "Unsupported JSON value")


def _canonical(value):
    _json_check(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def content_hash(value):
    """SHA-256 of canonical JSON, distinct from a raw instruction byte hash."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def validate_specification(spec):
    _require(isinstance(spec, dict) and set(spec) == {"version", "contract", "output_schema", "clauses"},
             "Specification fields must be version, contract, output_schema, clauses")
    _require(type(spec["version"]) is int and spec["version"] == 1, "Unsupported specification version")
    _text(spec["contract"], "contract")
    _require(isinstance(spec["output_schema"], dict) and bool(spec["output_schema"]), "Require one shared output schema object")
    _json_check(spec["output_schema"])
    _require(isinstance(spec["clauses"], list) and len(spec["clauses"]) == 4, "Require exactly four clauses")
    ids = []
    for clause in spec["clauses"]:
        _require(isinstance(clause, dict) and set(clause) == {"id", "baseline", "replacement"},
                 "A replacement cannot override the shared contract/schema or another clause")
        ids.append(_text(clause["id"], "clause ID"))
        _text(clause["baseline"], "baseline clause")
        _text(clause["replacement"], "replacement clause")
        _require(clause["baseline"] != clause["replacement"], "Replacement must change its own clause")
    _require(len(set(ids)) == 4, "Conflicting duplicate clause IDs")
    return deepcopy(spec)


def _configuration(configuration):
    _require(type(configuration) is str and configuration in CONFIGURATIONS, "Configuration must be a four-bit string")
    return configuration


def compose_instruction(spec, configuration):
    """Apply exact fixed strings once, in declared clause order; no rewriting."""
    spec = validate_specification(spec)
    _configuration(configuration)
    parts = [spec["contract"]]
    parts.extend(f"[{clause['id']}]\n{clause['replacement'] if bit == '1' else clause['baseline']}"
                 for bit, clause in zip(configuration, spec["clauses"]))
    parts.append("Output schema:\n" + _canonical(spec["output_schema"]))
    return "\n\n".join(parts)


def instruction_hash(spec, configuration):
    return hashlib.sha256(compose_instruction(spec, configuration).encode("utf-8")).hexdigest()


def _scope(scope):
    _require(isinstance(scope, dict) and set(scope) == {"group_ids", "edges"}, "Scope requires group_ids and directed edges")
    groups, edges = scope["group_ids"], scope["edges"]
    _require(isinstance(groups, list) and 1 <= len(groups) <= 4096, "Invalid group inventory")
    for group in groups:
        _text(group, "group ID")
    _require(len(set(groups)) == len(groups), "Duplicate group ID")
    _require(isinstance(edges, list) and 1 <= len(edges) <= 64, "Invalid edge inventory")
    edge_ids, pairs = [], []
    for edge in edges:
        _require(isinstance(edge, dict) and set(edge) == {"id", "left", "right"}, "Malformed directed edge")
        for key in edge:
            _text(edge[key], "edge field")
        _require(edge["left"] != edge["right"], "Self edges carry no response information")
        edge_ids.append(edge["id"])
        pairs.append((edge["left"], edge["right"]))
    _require(len(set(edge_ids)) == len(edges) and len(set(pairs)) == len(edges), "Duplicate directed edge")
    return deepcopy(scope)


def coordinate(group, edge, slot=None, value=None):
    """Encode a fact coordinate, or the invalid-output atom when both are None."""
    _text(group, "group ID"); _text(edge, "edge ID")
    if slot is None and value is None:
        fields = [group, edge, "invalid", "", ""]
    else:
        fields = [group, edge, "fact", _text(slot, "slot"), _text(value, "value")]
    return _canonical(fields)


def _vector(vector, scope):
    _require(isinstance(vector, dict) and len(vector) <= 200000, "Vector must be a bounded sparse mapping")
    groups, edges = set(scope["group_ids"]), {edge["id"] for edge in scope["edges"]}
    result = {}
    for key, value in vector.items():
        _require(isinstance(key, str), "Coordinate must be a canonical JSON string")
        try:
            fields = json.loads(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("Malformed vector coordinate") from exc
        _require(isinstance(fields, list) and len(fields) == 5 and all(type(x) is str for x in fields), "Malformed vector coordinate")
        group, edge, kind, slot, atom = fields
        _require(group in groups and edge in edges, "Coordinate belongs to a different dialogue/edge scope")
        _require(kind in {"fact", "invalid"}, "Unknown coordinate kind")
        expected = coordinate(group, edge) if kind == "invalid" else coordinate(group, edge, slot, atom)
        _require(expected == key, "Noncanonical or malformed invalid coordinate")
        number = _number(value, "vector entry")
        if number != 0:
            result[key] = number
    return dict(sorted(result.items()))


def response_vector(scope, outputs):
    """Build signed edge differences, retaining predicted/hallucinated atoms.

outputs[group][setting] is a parsed {'slot_values': {slot: [value]}} or None for
an invalid response. The caller owns parsing and task-schema checking. No
reference vocabulary filters predicted atoms. Common errors can cancel across
edges: this is not an exact-accuracy metric or a deployment certificate.
"""
    scope = _scope(scope)
    settings = {edge[key] for edge in scope["edges"] for key in ("left", "right")}
    _require(isinstance(outputs, dict) and set(outputs) == set(scope["group_ids"]), "Output group inventory differs from scope")
    def atoms(output):
        if output is None:
            return {(None, None)}
        _require(isinstance(output, dict) and set(output) == {"slot_values"}
                 and isinstance(output["slot_values"], dict), "Expected parsed state or None for invalid output")
        result = set()
        for slot, values in output["slot_values"].items():
            _text(slot, "slot")
            _require(isinstance(values, list) and len(values) == 1, "Require one predicted value per slot")
            result.add((slot, _text(values[0], "value")))
        return result
    result = {}
    for group in scope["group_ids"]:
        _require(isinstance(outputs[group], dict) and set(outputs[group]) == settings, "Output settings differ from scope")
        parsed = {setting: atoms(value) for setting, value in outputs[group].items()}
        for edge in scope["edges"]:
            left, right = parsed[edge["left"]], parsed[edge["right"]]
            for slot, value in left | right:
                difference = int((slot, value) in right) - int((slot, value) in left)
                if difference:
                    result[coordinate(group, edge["id"], slot, value)] = float(difference)
    return dict(sorted(result.items()))


def observation(spec, scope, configuration, vector, *, augmented_accuracy, original_accuracy):
    """Bind a caller-supplied measurement to exact strings, schema and scope.

This creates metadata, not proof that inference ran or that labels are correct.
External experiments must retain their call/annotation evidence and costs.
"""
    spec, scope = validate_specification(spec), _scope(scope)
    return {"instruction_sha256": instruction_hash(spec, configuration),
            "schema_sha256": content_hash(spec["output_schema"]), "scope_sha256": content_hash(scope),
            "vector": _vector(vector, scope),
            "augmented_accuracy": _number(augmented_accuracy, "augmented_accuracy", low=0, high=1),
            "original_accuracy": _number(original_accuracy, "original_accuracy", low=0, high=1)}


def _measurements(spec, scope, measurements, configurations):
    _require(isinstance(measurements, dict) and set(measurements) == set(configurations), "Missing or extra measured configurations")
    normalized = {}
    for config in configurations:
        row = measurements[config]
        _require(isinstance(row, dict) and set(row) == {"instruction_sha256", "schema_sha256", "scope_sha256", "vector", "augmented_accuracy", "original_accuracy"}, "Unexpected observation fields")
        expected = observation(spec, scope, config, row["vector"], augmented_accuracy=row["augmented_accuracy"], original_accuracy=row["original_accuracy"])
        _require(all(row[key] == expected[key] for key in ("instruction_sha256", "schema_sha256", "scope_sha256")), "Observation instruction/schema/scope binding differs")
        normalized[config] = expected
    return normalized


def _policy(policy):
    fields = {"max_edits", "edit_penalty", "prediction_error_l2_threshold", "noise_l2", "noise_multiplier", "max_noise_l2"}
    _require(isinstance(policy, dict) and set(policy) == fields, "Declare all search and future-round thresholds before fitting")
    _require(type(policy["max_edits"]) is int and 0 <= policy["max_edits"] <= 4, "Invalid edit budget")
    return {"max_edits": policy["max_edits"], **{key: _number(policy[key], key, low=0) for key in sorted(fields - {"max_edits"})}}


def _finite_sum(values, name):
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} overflowed finite arithmetic") from exc
    return _number(total, name)


def _linear_combination(terms):
    keys = set().union(*(vector for _, vector in terms))
    result = {}
    for key in sorted(keys):
        value = _finite_sum((weight * vector.get(key, 0.0) for weight, vector in terms), "computed vector entry")
        if value:
            result[key] = value
    return result


def _squared_error(left, right):
    differences = [left.get(key, 0.0) - right.get(key, 0.0) for key in left.keys() | right.keys()]
    return _finite_sum((value * value for value in differences), "squared error")


def _single(index):
    return "".join("1" if i == index else "0" for i in range(4))


def _pair(i, j):
    return "".join("1" if k in (i, j) else "0" for k in range(4))


def _best(scores, configurations):
    return min(configurations, key=lambda config: (scores[config], config.count("1"), config))


def fit_response_model(spec, scope, measurements, target_vector, policy):
    """Freeze 16 predictions and choices from exactly the 11 degree<=2 cells.

Both vector controls and scalar interpolation receive identical observations.
The first-order control uses their baseline/single effects; measured pairs are
available but cannot alter its declared first-order representation. Scalar
quadratic predictions are not clipped to [0,1]. Enumeration is exact and tiny.
"""
    spec, scope, policy = validate_specification(spec), _scope(scope), _policy(policy)
    measurements = _measurements(spec, scope, measurements, FIT_CONFIGURATIONS)
    target = _vector(target_vector, scope)
    base = measurements["0000"]["vector"]
    first = {str(i): _linear_combination([(1, measurements[_single(i)]["vector"]), (-1, base)]) for i in range(4)}
    mixed = {f"{i},{j}": _linear_combination([(1, measurements[_pair(i, j)]["vector"]), (-1, measurements[_single(i)]["vector"]),
                                             (-1, measurements[_single(j)]["vector"]), (1, base)]) for i, j in combinations(range(4), 2)}
    scalar_base = measurements["0000"]["augmented_accuracy"]
    scalar_first = {str(i): measurements[_single(i)]["augmented_accuracy"] - scalar_base for i in range(4)}
    scalar_mixed = {f"{i},{j}": measurements[_pair(i, j)]["augmented_accuracy"] - measurements[_single(i)]["augmented_accuracy"]
                    - measurements[_single(j)]["augmented_accuracy"] + scalar_base for i, j in combinations(range(4), 2)}
    predictions = {}
    for config in CONFIGURATIONS:
        active = [i for i, bit in enumerate(config) if bit == "1"]
        linear = _linear_combination([(1, base)] + [(1, first[str(i)]) for i in active])
        quadratic = _linear_combination([(1, linear)] + [(1, mixed[f"{i},{j}"]) for i, j in combinations(active, 2)])
        accuracy = _number(math.fsum([scalar_base] + [scalar_first[str(i)] for i in active]
                                     + [scalar_mixed[f"{i},{j}"] for i, j in combinations(active, 2)]), "predicted scalar accuracy")
        predictions[config] = {"instruction_sha256": instruction_hash(spec, config), "edit_count": len(active),
                               "linear_vector": linear, "second_order_vector": quadratic,
                               "linear_response_squared_error": _squared_error(linear, target),
                               "second_order_response_squared_error": _squared_error(quadratic, target),
                               "scalar_quadratic_accuracy": accuracy}
    eligible = [c for c in CONFIGURATIONS if c.count("1") <= policy["max_edits"]]
    penalty = policy["edit_penalty"]
    objectives = {method: {c: _number(((-predictions[c]["scalar_quadratic_accuracy"]) if method == "scalar_quadratic"
                                     else predictions[c][method.replace("_vector", "_response_squared_error")])
                                     + penalty * c.count("1"), "predicted objective") for c in CONFIGURATIONS} for method in METHODS}
    measured_eligible = [c for c in FIT_CONFIGURATIONS if c in eligible]
    selections = {method: _best(objectives[method], eligible) for method in METHODS}
    selections["measured_best_response"] = _best({c: _squared_error(row["vector"], target) + penalty * c.count("1") for c, row in measurements.items()}, measured_eligible)
    selections["measured_best_accuracy"] = _best({c: -row["augmented_accuracy"] + penalty * c.count("1") for c, row in measurements.items()}, measured_eligible)
    frozen = {"version": 1, "kind": "prospective_response_composition", "specification": spec, "scope": scope,
              "scope_sha256": content_hash(scope), "target_vector": target, "policy": policy,
              "fit_configurations": list(FIT_CONFIGURATIONS), "heldout_configurations": list(HELDOUT_CONFIGURATIONS),
              "measurements": measurements, "effects": {"first": first, "mixed": mixed},
              "scalar_coefficients": {"baseline": scalar_base, "first": scalar_first, "mixed": scalar_mixed},
              "predictions": predictions, "predicted_objectives": objectives, "eligible_configurations": eligible, "selections": selections,
              "tie_break": "minimum objective, then fewer edits, then lexicographic four-bit string",
              "deployment_authorized": False}
    frozen["freeze_sha256"] = content_hash(frozen)
    return frozen


def verify_freeze(frozen):
    """Check the content commitment and replay the entire frozen prediction grid.

A hash is not a timestamp: an external runner must persist it before measuring
held-out configurations. The pure numerical API cannot establish chronology.
"""
    _require(isinstance(frozen, dict) and "freeze_sha256" in frozen, "Missing prediction freeze")
    body = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    _require(content_hash(body) == frozen["freeze_sha256"], "Prediction freeze hash changed")
    expected = fit_response_model(frozen["specification"], frozen["scope"], frozen["measurements"], frozen["target_vector"], frozen["policy"])
    _require(_canonical(expected) == _canonical(frozen), "Frozen predictions/selection fail exact reconstruction")
    return frozen["freeze_sha256"]


def _ranking(predicted, actual):
    sign = lambda value: (value > 0) - (value < 0)
    pairs = [(sign(predicted[a] - predicted[b]), sign(actual[a] - actual[b])) for a, b in combinations(HELDOUT_CONFIGURATIONS, 2)]
    return {"pair_count": len(pairs), "order_matches_including_ties": sum(a == b for a, b in pairs),
            "strict_inversions": sum(a * b == -1 for a, b in pairs),
            "predicted_ties": sum(a == 0 for a, _ in pairs), "actual_ties": sum(b == 0 for _, b in pairs)}


def compare_heldout(frozen, heldout):
    """Audit the five composition holdouts; NEVER select a replacement winner.

All rows share the frozen dialogue/edge scope. Original-task regret here is on
those same generated groups, not fresh deployment inputs. Exhaustive references
are labelled post-hoc oracles and require all sixteen configuration observations.
"""
    freeze_hash = verify_freeze(frozen)
    heldout = _measurements(frozen["specification"], frozen["scope"], heldout, HELDOUT_CONFIGURATIONS)
    actual = {**frozen["measurements"], **heldout}
    target, penalty = frozen["target_vector"], frozen["policy"]["edit_penalty"]
    losses = {c: _squared_error(row["vector"], target) for c, row in actual.items()}
    errors = {method: {} for method in METHODS}
    for config, row in heldout.items():
        for method in METHODS:
            predicted = frozen["predictions"][config]
            if method == "scalar_quadratic":
                error = abs(predicted["scalar_quadratic_accuracy"] - row["augmented_accuracy"])
                errors[method][config] = {"absolute_accuracy_error": error}
            else:
                squared = _squared_error(predicted[method], row["vector"])
                errors[method][config] = {"squared_l2": squared, "l2": math.sqrt(squared)}
    error_summary = {}
    rankings = {}
    for method in METHODS:
        field = "absolute_accuracy_error" if method == "scalar_quadratic" else "l2"
        values = [row[field] for row in errors[method].values()]
        error_summary[method] = {"metric": field, "mean": math.fsum(values) / len(values), "maximum": max(values)}
        prediction = {c: -frozen["predictions"][c]["scalar_quadratic_accuracy"] if method == "scalar_quadratic"
                      else frozen["predictions"][c][method.replace("_vector", "_response_squared_error")] for c in HELDOUT_CONFIGURATIONS}
        truth = {c: -heldout[c]["augmented_accuracy"] if method == "scalar_quadratic" else losses[c] for c in HELDOUT_CONFIGURATIONS}
        rankings[method] = _ranking(prediction, truth)
    eligible = frozen["eligible_configurations"]
    response_objectives = {c: losses[c] + penalty * c.count("1") for c in eligible}
    accuracy_objectives = {c: -actual[c]["augmented_accuracy"] + penalty * c.count("1") for c in eligible}
    best_original = max(actual[c]["original_accuracy"] for c in eligible)
    def assessment(config):
        row = actual[config]
        return {"configuration": config, "instruction_sha256": frozen["predictions"][config]["instruction_sha256"],
                "response_squared_error": losses[config], "augmented_accuracy": row["augmented_accuracy"], "original_accuracy": row["original_accuracy"],
                "original_accuracy_regret": best_original - row["original_accuracy"],
                "response_objective_regret": response_objectives[config] - min(response_objectives.values()),
                "accuracy_objective_regret": accuracy_objectives[config] - min(accuracy_objectives.values()),
                "original_accuracy_change_from_baseline": row["original_accuracy"] - actual["0000"]["original_accuracy"],
                "harmful_vs_baseline_original_accuracy": row["original_accuracy"] < actual["0000"]["original_accuracy"]}
    audit = {"version": 1, "kind": "composition_holdout_audit", "freeze_sha256": freeze_hash,
             "heldout_measurements_sha256": content_hash(heldout), "heldout_measurements": heldout,
             "scope_sha256": frozen["scope_sha256"], "scope": "composition generalization on the same dialogue/edge groups; not fresh-group transfer",
             "frozen_selections": deepcopy(frozen["selections"]), "selection_assessments": {method: assessment(c) for method, c in frozen["selections"].items()},
             "errors": errors, "error_summary": error_summary, "heldout_rankings": rankings,
             "ranking_objectives": "Unpenalized response squared error, or negative augmented accuracy; distinct from penalized selection objectives.",
             "regularization_note": "The same numeric edit penalty has different units across vector loss and accuracy; it is not matched effective regularization.",
             "exhaustive_reference": {"posthoc_only": True, "configuration_observations_required": 16,
                                      "best_response": assessment(_best(response_objectives, eligible)),
                                      "best_accuracy": assessment(_best(accuracy_objectives, eligible)), "best_original_accuracy": best_original},
             "harmful_heldout_compositions": {"count": sum(row["original_accuracy"] < actual["0000"]["original_accuracy"] for row in heldout.values()), "denominator": 5},
             "observation_counts": {"fit": 11, "composition_holdout": 5, "total": 16, "actual_model_calls_and_tokens": None},
             "deployment_authorized": False, "reselections_after_holdout": 0}
    audit["audit_sha256"] = content_hash(audit)
    return audit


def trust_region_decision(frozen, audit):
    """Reduce only a FUTURE round's edit budget, using predeclared thresholds.

This is neither a deployment guard nor a new selection on opened holdouts.
Noise is a caller-supplied predeclared scale, not evidence that repeats occurred.
"""
    verify_freeze(frozen)
    _require(isinstance(audit, dict) and audit.get("freeze_sha256") == frozen["freeze_sha256"], "Audit belongs to a different prediction freeze")
    expected = compare_heldout(frozen, audit["heldout_measurements"])
    _require(_canonical(expected) == _canonical(audit), "Heldout audit changed or fails reconstruction")
    policy = frozen["policy"]
    threshold = _number(policy["prediction_error_l2_threshold"] + policy["noise_multiplier"] * policy["noise_l2"], "effective threshold")
    error = audit["error_summary"]["second_order_vector"]["maximum"]
    reasons = []
    if policy["noise_l2"] > policy["max_noise_l2"]:
        reasons.append("predeclared_noise_limit_exceeded")
    if error > threshold:
        reasons.append("heldout_prediction_error_exceeded")
    old = policy["max_edits"]
    return {"freeze_sha256": frozen["freeze_sha256"], "audit_sha256": audit["audit_sha256"],
            "applies_to": "future_new_measurement_round_only", "previous_max_edits": old,
            "next_max_edits": max(0, old - 1) if reasons else old, "reduction_reasons": reasons,
            "maximum_heldout_error_l2": error, "effective_error_l2_threshold": threshold,
            "current_frozen_selections": deepcopy(frozen["selections"]), "reselections_after_holdout": 0,
            "deployment_authorized": False}
