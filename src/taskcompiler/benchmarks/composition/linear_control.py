"""Classical all-observation OLS control for four-clause response composition.

This is an ordinary linear surrogate, not new mathematics or a deployment
certificate. It uses all eleven measured degree-at-most-two configurations.
Five higher-order configurations may audit the frozen fit but never reselect it.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import math

from .response_model import (
    CONFIGURATIONS, FIT_CONFIGURATIONS, HELDOUT_CONFIGURATIONS,
    _best, _canonical, _finite_sum, _linear_combination, _measurements,
    _number, _policy, _ranking, _require, _scope, _squared_error, _vector,
    content_hash, instruction_hash, validate_specification,
)


@lru_cache(maxsize=1)
def _projection_weights():
    """Exact rational (X'X)^-1 X' for columns [1,z0,z1,z2,z3]."""
    design = tuple((1, *map(int, config)) for config in FIT_CONFIGURATIONS)
    gram = [[sum(row[i] * row[j] for row in design) for j in range(5)] for i in range(5)]
    augmented = [[Fraction(value) for value in row] + [Fraction(i == j) for j in range(5)]
                 for i, row in enumerate(gram)]
    for column in range(5):
        pivot = next((i for i in range(column, 5) if augmented[i][column]), None)
        _require(pivot is not None, "OLS design is not full rank")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for i in range(5):
            if i != column:
                multiplier = augmented[i][column]
                augmented[i] = [left - multiplier * right for left, right in zip(augmented[i], augmented[column])]
    inverse = [row[5:] for row in augmented]
    return tuple(tuple(sum(inverse[i][j] * row[j] for j in range(5)) for row in design) for i in range(5))


def _weighted_value(weights, values):
    # Exact accumulation of represented input floats prevents spurious slopes
    # when all eleven measurements are identical. Only the final coefficient is
    # rounded to float; no tolerance or outcome-dependent clipping is introduced.
    rational = sum((weight * Fraction(value) for weight, value in zip(weights, values)), Fraction(0))
    try:
        value = float(rational)
    except (ValueError, OverflowError) as exc:
        raise ValueError("OLS coefficient overflowed finite arithmetic") from exc
    return _number(value, "OLS coefficient")


def fit_linear_control(spec, scope, measurements, target_vector, policy):
    """Freeze all sixteen OLS vector predictions and one bounded selection.

    All eleven equally weighted observations fit five coefficients per sparse
    coordinate. Accuracy fields are retained for later audits, never fitted or
    used to select this vector-loss control. Actual model costs belong to the
    external experiment that supplied the shared observations.
    """
    spec, scope, policy = validate_specification(spec), _scope(scope), _policy(policy)
    measurements = _measurements(spec, scope, measurements, FIT_CONFIGURATIONS)
    target = _vector(target_vector, scope)
    coordinates = sorted(set(target).union(*(row["vector"] for row in measurements.values())))
    weights = _projection_weights()
    coefficients = [{} for _ in range(5)]
    for atom in coordinates:
        values = [measurements[config]["vector"].get(atom, 0.0) for config in FIT_CONFIGURATIONS]
        for column in range(5):
            value = _weighted_value(weights[column], values)
            if value:
                coefficients[column][atom] = value
    predictions = {}
    for config in CONFIGURATIONS:
        vector = _linear_combination([(1, coefficients[0])] + [(int(bit), coefficients[i + 1]) for i, bit in enumerate(config)])
        loss = _squared_error(vector, target)
        objective = _finite_sum((loss, policy["edit_penalty"] * config.count("1")), "OLS selection objective")
        predictions[config] = {"instruction_sha256": instruction_hash(spec, config), "edit_count": config.count("1"),
                               "linear_vector": vector, "response_squared_error": loss, "objective": objective}
    eligible = [config for config in CONFIGURATIONS if config.count("1") <= policy["max_edits"]]
    selected = _best({config: row["objective"] for config, row in predictions.items()}, eligible)
    residual_sum = _finite_sum((_squared_error(predictions[config]["linear_vector"], measurements[config]["vector"])
                                for config in FIT_CONFIGURATIONS), "OLS training residual sum")
    frozen = {
        "version": 1, "kind": "all_observation_linear_response_control", "fit_method": "ordinary_least_squares",
        "specification": spec, "scope": scope, "scope_sha256": content_hash(scope), "target_vector": target, "policy": policy,
        "fit_configurations": list(FIT_CONFIGURATIONS), "heldout_configurations": list(HELDOUT_CONFIGURATIONS),
        "measurements": measurements, "design_columns": ["intercept", "z0", "z1", "z2", "z3"], "design_rank": 5,
        "observation_weighting": "all eleven observations have equal weight", "fitted_coordinates": coordinates,
        "coefficients": {"intercept": coefficients[0], "main_effects": {str(i): coefficients[i + 1] for i in range(4)}},
        "fit_residual_squared_error_sum": residual_sum, "predictions": predictions,
        "eligible_configurations": eligible, "selected": selected, "selections": {"ols_linear_vector": selected},
        "tie_break": "minimum objective, then fewer edits, then lexicographic four-bit string",
        "novel_mathematics_claim": False, "deployment_authorized": False,
        "limitations": ["Classical OLS is a matched information control, not a new optimization method.",
                        "Linear effects can miss interactions and shared response errors; predictions are not clipped or certified.",
                        "A content hash does not establish chronology; the external runner must persist this before observing the five holdouts."],
    }
    frozen["freeze_sha256"] = content_hash(frozen)
    return frozen


def verify_linear_control(frozen):
    """Verify the content commitment and reconstruct fit, predictions and choice."""
    _require(isinstance(frozen, dict) and "freeze_sha256" in frozen, "Missing OLS prediction freeze")
    _require(content_hash({key: value for key, value in frozen.items() if key != "freeze_sha256"}) == frozen["freeze_sha256"],
             "OLS freeze hash changed")
    expected = fit_linear_control(frozen["specification"], frozen["scope"], frozen["measurements"], frozen["target_vector"], frozen["policy"])
    _require(_canonical(expected) == _canonical(frozen), "OLS fit or selection fails exact reconstruction")
    return frozen["freeze_sha256"]


def compare_linear_control_heldout(frozen, heldout):
    """Score the five held-out compositions without changing the frozen choice.

    This evaluates composition on the same dialogue/edge scope, not fresh-group
    transfer. Exhaustive regret uses all sixteen observations as a post-hoc
    diagnostic and grants no permission to deploy or tune on the holdouts.
    """
    freeze_hash = verify_linear_control(frozen)
    heldout = _measurements(frozen["specification"], frozen["scope"], heldout, HELDOUT_CONFIGURATIONS)
    actual = {**frozen["measurements"], **heldout}
    target, policy = frozen["target_vector"], frozen["policy"]
    errors = {}
    for config, row in heldout.items():
        squared = _squared_error(frozen["predictions"][config]["linear_vector"], row["vector"])
        errors[config] = {"squared_l2": squared, "l2": math.sqrt(squared)}
    values = [row["l2"] for row in errors.values()]
    losses = {config: _squared_error(row["vector"], target) for config, row in actual.items()}
    eligible = frozen["eligible_configurations"]
    objectives = {config: _finite_sum((losses[config], policy["edit_penalty"] * config.count("1")), "Actual response objective") for config in eligible}
    best_original = max(actual[config]["original_accuracy"] for config in eligible)

    def assessment(config):
        row = actual[config]
        return {"configuration": config, "instruction_sha256": frozen["predictions"][config]["instruction_sha256"],
                "response_squared_error": losses[config], "augmented_accuracy": row["augmented_accuracy"], "original_accuracy": row["original_accuracy"],
                "original_accuracy_regret": best_original - row["original_accuracy"],
                "response_objective_regret": objectives[config] - min(objectives.values()),
                "original_accuracy_change_from_baseline": row["original_accuracy"] - actual["0000"]["original_accuracy"],
                "harmful_vs_baseline_original_accuracy": row["original_accuracy"] < actual["0000"]["original_accuracy"]}

    result = {
        "version": 1, "kind": "ols_linear_composition_holdout_audit", "freeze_sha256": freeze_hash,
        "heldout_measurements_sha256": content_hash(heldout), "heldout_measurements": heldout, "scope_sha256": frozen["scope_sha256"],
        "scope": "composition generalization on the same dialogue/edge groups; not fresh-group transfer",
        "frozen_selection": frozen["selected"], "selection_assessment": assessment(frozen["selected"]), "errors": errors,
        "error_summary": {"metric": "l2", "mean": _finite_sum(values, "Mean OLS holdout error numerator") / 5, "maximum": max(values)},
        "heldout_ranking": _ranking({config: frozen["predictions"][config]["response_squared_error"] for config in HELDOUT_CONFIGURATIONS},
                                     {config: losses[config] for config in HELDOUT_CONFIGURATIONS}),
        "ranking_objective": "Unpenalized response squared error, distinct from penalized selection objectives.",
        "exhaustive_reference": {"posthoc_only": True, "configuration_observations_required": 16,
                                 "best_response": assessment(_best(objectives, eligible)), "best_original_accuracy": best_original},
        "observation_counts": {"fit": 11, "composition_holdout": 5, "total": 16, "actual_model_calls_and_tokens": None},
        "reselections_after_holdout": 0, "novel_mathematics_claim": False, "deployment_authorized": False,
    }
    result["audit_sha256"] = content_hash(result)
    return result


def verify_linear_control_heldout(frozen, audit):
    """Verify an audit against its original freeze; never revise the selection."""
    _require(isinstance(audit, dict) and audit.get("freeze_sha256") == verify_linear_control(frozen), "OLS audit belongs to a different freeze")
    expected = compare_linear_control_heldout(frozen, audit["heldout_measurements"])
    _require(_canonical(expected) == _canonical(audit), "OLS heldout audit fails exact reconstruction")
    return audit["audit_sha256"]
