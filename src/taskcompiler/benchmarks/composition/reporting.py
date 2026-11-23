"""Prespecified four-service decision gates; no data or model access.

Inputs must first pass the external retained-call/source/chronology audit.
This numerical summary cannot establish that the supplied evidence is genuine.
"""
from __future__ import annotations

import math

SERVICES = ("Media_2", "Banks_2", "Homes_1", "RentalCars_1")
COMPARATORS = ("linear_vector", "ols_linear_vector", "scalar_quadratic")
METHODS = ("baseline", "linear_vector", "second_order_vector", "scalar_quadratic",
           "measured_best_response", "measured_best_accuracy", "ols_linear_vector")


def _number(value, name, *, maximum=None):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise ValueError(f"Invalid finite nonnegative {name}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Out-of-range {name}")
    return float(value)


def summarize_gate(cells, *, integrity_verified):
    """Apply v1 plus all-fit-linear v2 without selecting favorable services.

Each cell contains completed audited fit/holdout/guard/audit artifacts. Missing
or failed cells remain in the planned inventory and make the gate inconclusive.
No metric here changes a frozen instruction or authorizes a larger live run.
"""
    if type(integrity_verified) is not bool or not isinstance(cells, dict) or set(cells) != set(SERVICES):
        raise ValueError("Require the exact four-service inventory and an explicit integrity flag")
    missing = [service for service in SERVICES if cells[service].get("status") != "completed"]
    if missing or not integrity_verified:
        return {"planned_services": list(SERVICES), "completed_services": [s for s in SERVICES if s not in missing],
                "incomplete_services": missing, "integrity_verified": integrity_verified,
                "outcome": "inconclusive_integrity_or_incomplete_grid", "scale_allowed": False,
                "eligible_for_separately_frozen_scaling": False, "automatic_run_authorized": False,
                "publication_ready": False, "gate_thresholds": {"noise_max": 0.25,
                    "prediction_relative_reduction_min": 0.10, "services_with_lower_error_min": 3},
                "instruction_reselection": False}
    rows = {}
    for service in SERVICES:
        cell = cells[service]
        for stage in ("fit", "holdout", "guard", "audit"):
            if cell[stage]["status"] != "completed":
                raise ValueError("Completed cell contains an incomplete stage")
        fit, held, audit = cell["fit"], cell["holdout"], cell["audit"]
        if fit["noise"]["normalization"] != 4 or fit["noise"]["extra_passes"] != 2 or fit["noise"]["averaged_into_fit"] is not False:
            raise ValueError("Unexpected noise normalization or calibration policy")
        noise = _number(fit["noise"]["normalized_l2"], "baseline noise")
        errors = {method: _number(held["comparison"]["error_summary"][method]["mean"], "mean holdout L2") / 4
                  for method in ("linear_vector", "second_order_vector")}
        errors["ols_linear_vector"] = _number(held["ols_comparison"]["error_summary"]["mean"], "OLS mean holdout L2") / 4
        if any(held["comparison"]["error_summary"][method]["metric"] != "l2" for method in errors if method != "ols_linear_vector") or held["ols_comparison"]["error_summary"]["metric"] != "l2":
            raise ValueError("Prediction error unit changed")
        scalar_error = _number(held["comparison"]["error_summary"]["scalar_quadratic"]["mean"], "scalar absolute accuracy error")
        if held["comparison"]["error_summary"]["scalar_quadratic"]["metric"] != "absolute_accuracy_error":
            raise ValueError("Scalar error unit changed")
        assessments = {**held["comparison"]["selection_assessments"],
                       "ols_linear_vector": held["ols_comparison"]["selection_assessment"]}
        regrets = {method: _number(assessments[method]["original_accuracy_regret"], "original-task regret", maximum=1)
                   for method in (*COMPARATORS, "second_order_vector", "measured_best_response", "measured_best_accuracy")}
        if set(audit["scores"]) != set(METHODS):
            raise ValueError("Canonical audit method inventory changed")
        accuracies = {}
        for method in METHODS:
            score = audit["scores"][method]
            if score["denominator"] != 4 or type(score["correct"]) is not int or not 0 <= score["correct"] <= 4:
                raise ValueError("Canonical audit denominator/count changed")
            accuracies[method] = _number(score["accuracy"], "canonical accuracy", maximum=1)
            if accuracies[method] != score["correct"] / 4:
                raise ValueError("Canonical audit score inconsistent with its count")
        rows[service] = {"normalized_baseline_noise": noise, "normalized_mean_prediction_l2": errors,
                         "scalar_mean_absolute_accuracy_error": scalar_error, "original_regret": regrets,
                         "canonical_audit_accuracy": accuracies,
                         "fallbacks": cell["guard"]["fallbacks"],
                         "heldout_rankings": {**held["comparison"]["heldout_rankings"],
                                             "ols_linear_vector": held["ols_comparison"]["heldout_ranking"]},
                         "harmful_heldout_compositions": held["comparison"]["harmful_heldout_compositions"]}
    mean = lambda values: math.fsum(values) / 4
    noise = mean(row["normalized_baseline_noise"] for row in rows.values())
    prediction = {method: mean(row["normalized_mean_prediction_l2"][method] for row in rows.values())
                  for method in ("linear_vector", "ols_linear_vector", "second_order_vector")}
    comparisons = {}
    for method in ("linear_vector", "ols_linear_vector"):
        reduction = prediction[method] - prediction["second_order_vector"]
        lower = sum(row["normalized_mean_prediction_l2"]["second_order_vector"] < row["normalized_mean_prediction_l2"][method]
                    for row in rows.values())
        relative = reduction / prediction[method] if prediction[method] > 0 else None
        comparisons[method] = {"absolute_reduction": reduction, "relative_reduction": relative,
                               "services_with_lower_error": lower,
                               "pass": (relative is not None and prediction["second_order_vector"] <= 0.9 * prediction[method] and reduction > 0
                                        and reduction > noise and lower >= 3)}
    regrets = {method: mean(row["original_regret"][method] for row in rows.values())
               for method in (*COMPARATORS, "second_order_vector", "measured_best_response", "measured_best_accuracy")}
    accuracy = {method: mean(row["canonical_audit_accuracy"][method] for row in rows.values()) for method in METHODS}
    stability = all(row["normalized_baseline_noise"] <= 0.25 for row in rows.values())
    prediction_pass = all(row["pass"] for row in comparisons.values())
    decision_pass = (all(regrets["second_order_vector"] <= regrets[method] for method in COMPARATORS)
                     and any(regrets["second_order_vector"] < regrets[method] for method in COMPARATORS))
    transfer_pass = (all(accuracy["second_order_vector"] >= accuracy[method] for method in ("baseline", *COMPARATORS))
                     and any(accuracy["second_order_vector"] > accuracy[method] for method in COMPARATORS))
    passed = stability and prediction_pass and decision_pass and transfer_pass
    outcome = ("small_controlled_gate_passed" if passed else
               "inconclusive_excess_baseline_noise" if not stability else
               "prediction_gate_failed_no_scaling" if not prediction_pass else
               "limited_prediction_result_no_instruction_superiority")
    return {"planned_services": list(SERVICES), "completed_services": list(SERVICES), "incomplete_services": [],
            "integrity_verified": True, "per_service": rows,
            "pooled": {"normalized_baseline_noise": noise, "normalized_mean_prediction_l2": prediction,
                       "prediction_comparisons": comparisons, "original_regret": regrets,
                       "canonical_audit_accuracy": accuracy},
            "gates": {"integrity_stability": stability, "composition_prediction": prediction_pass,
                      "decision_usefulness": decision_pass, "fresh_controlled_transfer": transfer_pass},
            "outcome": outcome, "scale_allowed": False,
            "eligible_for_separately_frozen_scaling": passed, "automatic_run_authorized": False,
            "publication_ready": False,
            "instruction_reselection": False,
            "limitations": ["Four related services, one student and one preparation; no broad significance claim.",
                            "Controlled renderer transfer is not unrestricted natural-language confirmation.",
                            "Passing does not establish novelty, superiority to native optimizers, or publication readiness.",
                            "Scaling requires a separate frozen protocol; this summary starts no experiment.",
                            "Measured-best matches preclude a surrogate-efficiency claim at this scope."]}
