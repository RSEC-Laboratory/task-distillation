"""SGD goal-state scoring in the official exact-match mode.

Categorical values match ignoring case. Noncategorical values match any exact
reference alternative, with case and spacing preserved. This follows the pinned
official metric's use_fuzzy_match=False setting, not its fuzzy leaderboard mode.
Invalid predictions are incorrect; unassigned gold slots still count in joint
goal accuracy, so adding unsupported constraints cannot improve that metric.
"""
from __future__ import annotations

from collections import Counter
import math

from .dataset import strict_json


def validate_reference(state, service):
    if not isinstance(state, dict) or not isinstance(state.get("slot_values"), dict):
        raise ValueError("Reference requires slot_values")
    names = {slot["name"] for slot in service["slots"]}
    for name, values in state["slot_values"].items():
        if name not in names or not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
            raise ValueError("Malformed reference slot assignment")
    return state


def parse_state(output, service):
    """Require JSON with the task contract, accepting native one-string lists.

Bare scalar values are normalized to one-element lists for interoperability;
this documented representation normalization does not alter semantic text.
Only one predicted alternative is permitted, matching native SGD scoring.
"""
    try:
        state = strict_json(output) if isinstance(output, (str, bytes)) else output
    except (ValueError, TypeError, UnicodeError):
        return None, "invalid_json"
    if not isinstance(state, dict) or set(state) != {"slot_values"} or not isinstance(state["slot_values"], dict):
        return None, "invalid_state_schema"
    names = {slot["name"] for slot in service["slots"]}
    result = {}
    for name, values in state["slot_values"].items():
        if name not in names:
            return None, "unknown_slot"
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str) or not values[0]:
            return None, "invalid_slot_value"
        result[name] = list(values)
    return {"slot_values": result}, None


def score_state(reference, output, service):
    """Score all declared slots, including absent gold slots and format failures."""
    validate_reference(reference, service)
    prediction, error = parse_state(output, service)
    truth = reference["slot_values"]
    pred = {} if prediction is None else prediction["slot_values"]
    rows = []
    for slot in service["slots"]:
        name, categorical = slot["name"], slot["is_categorical"]
        present = name in truth
        if prediction is None:
            correct = 0.0
        elif present:
            if name not in pred:
                correct = 0.0
            elif categorical:
                correct = float(truth[name][0].lower() == pred[name][0].lower())
            else:
                correct = float(pred[name][0] in truth[name])
        else:
            correct = float(name not in pred)
        rows.append({"slot": name, "present": present, "categorical": categorical,
                     "accuracy": correct, "predicted_present": name in pred})
    def average(predicate):
        values = [row["accuracy"] for row in rows if predicate(row)]
        return sum(values) / len(values) if values else None
    def joint(predicate):
        values = [row["accuracy"] for row in rows if predicate(row)]
        return math.prod(values) if values else None
    active = average(lambda row: row["present"])
    return {"valid_format": prediction is not None, "format_error": error,
            "joint_goal_accuracy": joint(lambda row: True),
            "active_slot_accuracy": active, "average_goal_accuracy": active,
            "all_slot_accuracy": average(lambda row: True),
            "absent_slot_accuracy": average(lambda row: not row["present"]),
            "present_slot_accuracy": active,
            "average_cat_accuracy": average(lambda row: row["present"] and row["categorical"]),
            "average_noncat_accuracy": average(lambda row: row["present"] and not row["categorical"]),
            "joint_cat_accuracy": joint(lambda row: row["categorical"]),
            "joint_noncat_accuracy": joint(lambda row: not row["categorical"]),
            "slot_scores": rows}


def summarize_scores(rows):
    """Equal-frame macro metrics and explicit denominators; no skipped failures."""
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    names = ("joint_goal_accuracy", "active_slot_accuracy", "average_goal_accuracy",
             "all_slot_accuracy", "absent_slot_accuracy", "present_slot_accuracy",
             "average_cat_accuracy", "average_noncat_accuracy", "joint_cat_accuracy", "joint_noncat_accuracy")
    metrics, denominators = {}, {}
    for name in names:
        values = [row[name] for row in rows if row[name] is not None]
        metrics[name] = sum(values) / len(values) if values else None
        denominators[name] = len(values)
    slots = [slot for row in rows for slot in row["slot_scores"]]
    metrics["micro_active_slot_accuracy"] = (sum(x["accuracy"] for x in slots if x["present"])
                                             / sum(x["present"] for x in slots)) if any(x["present"] for x in slots) else None
    return {"frame_count": len(rows), "valid_format_count": sum(row["valid_format"] for row in rows),
            "format_failure_counts": dict(Counter(row["format_error"] for row in rows if row["format_error"])),
            "valid_format_rate": sum(row["valid_format"] for row in rows) / len(rows),
            "metrics": metrics, "metric_frame_denominators": denominators,
            "active_slot_count": sum(x["present"] for x in slots),
            "statistical_unit": "service task and dialogue clusters; frames are dependent",
            "mode": "official_exact_match_mode"}
