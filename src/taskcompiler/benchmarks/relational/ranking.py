"""Exact candidate-ranking bounds under coupled, partially known labels.

Certification under unread labels is established prior work. This prototype
specializes it to finite relational components and records its conditional scope.
It neither guarantees minimum acquisition cost nor unseen-query accuracy.
"""
from __future__ import annotations

from itertools import combinations
import json
from .constraints import feasible_components, audit_graph, InconsistentGraphError


class AnchorVerificationError(RuntimeError):
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace


def _equal(left, right):
    return type(left) is type(right) and left == right


def _validate_predictions(graph, predictions, order):
    nodes = {node["id"] for node in graph["nodes"]}
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("At least one candidate prediction map is required")
    if set(order) != set(predictions) or len(order) != len(predictions):
        raise ValueError("Candidate order must list each candidate exactly once")
    if any(not isinstance(identifier, str) or not identifier for identifier in order):
        raise ValueError("Candidate identifiers must be nonempty strings")
    for row in predictions.values():
        if not isinstance(row, dict) or set(row) != nodes:
            raise ValueError("Every candidate must predict exactly the graph nodes")
        if any(type(value) not in (bool, int, str, type(None)) for value in row.values()):
            raise ValueError("Predictions must be typed scalar answers or None for invalid output")


def ranking_bounds(graph, predictions, anchors=None, *, order=None):
    """Min/max accuracy gaps over every graph-consistent answer completion.

    The objective is additive across nodes and components are independent.
    Minima and maxima therefore decompose exactly; the Cartesian product of
    complete assignments is never materialized. Invalid outputs earn no credit.
    """
    components = feasible_components(graph, anchors)
    order = list(predictions) if order is None else list(order)
    _validate_predictions(graph, predictions, order)
    total = len(graph["nodes"])
    pairs = []
    for left, right in combinations(order, 2):
        rows = []
        for component in components:
            gaps = [sum(int(_equal(predictions[left][node], assignment[node]))
                        - int(_equal(predictions[right][node], assignment[node]))
                        for node in component["node_ids"])
                    for assignment in component["assignments"]]
            rows.append({"root": component["root"], "node_ids": component["node_ids"],
                         "lower_count": min(gaps), "upper_count": max(gaps),
                         "width_count": max(gaps) - min(gaps),
                         "solution_count": component["solution_count"]})
        lower, upper = sum(row["lower_count"] for row in rows), sum(row["upper_count"] for row in rows)
        pairs.append({"left": left, "right": right, "lower_count": lower,
                      "upper_count": upper, "lower_accuracy_gap": lower / total,
                      "upper_accuracy_gap": upper / total, "components": rows})
    return {"node_count": total, "candidate_order": order, "pairs": pairs,
            "component_count": len(components),
            "remaining_assignments": audit_graph(graph, anchors)["solutions_after_anchors"]}


def _gap(bounds, left, right):
    if left == right:
        return 0, 0
    for pair in bounds["pairs"]:
        if (pair["left"], pair["right"]) == (left, right):
            return pair["lower_count"], pair["upper_count"]
        if (pair["left"], pair["right"]) == (right, left):
            return -pair["upper_count"], -pair["lower_count"]
    raise KeyError((left, right))


def certified_winner(bounds):
    """Find the same accuracy maximizer under every feasible completion.

    Earlier candidates win accuracy ties. This explicit rule keeps Raw first
    when the caller supplies it first and makes the certificate checkable.
    """
    order = bounds["candidate_order"]
    for index, candidate in enumerate(order):
        if all(_gap(bounds, candidate, other)[0] >= (1 if other_index < index else 0)
               for other_index, other in enumerate(order) if other != candidate):
            return candidate
    return None


def select_instruction(graph, predictions, anchor_oracle, *, budget, raw="raw",
                       order=None, initial_anchors=None, acquisition="width", seed=0,
                       available_node_ids=None):
    """Query anchor information only while it can change instruction ranking.

    All candidates have already run on the same generated pool. This procedure
    requests checked labels, not more student responses or remote teacher calls.
    Width-based acquisition is a heuristic. Random and disagreement policies
    are supplied as mechanism controls with identical stopping certificates.
    """
    import random
    if type(budget) is not int or budget < 0:
        raise ValueError("Anchor budget must be a nonnegative integer")
    if acquisition not in {"width", "random", "disagreement"}:
        raise ValueError("Unknown anchor acquisition policy")
    order = ([raw] + [key for key in predictions if key != raw]) if order is None else list(order)
    if raw not in predictions or not order or order[0] != raw:
        raise ValueError("Raw must be an available candidate and first in the tie order")
    anchors = [dict(anchor) for anchor in (initial_anchors or [])]
    available = {node["id"] for node in graph["nodes"]} if available_node_ids is None else set(available_node_ids)
    if not available <= {node["id"] for node in graph["nodes"]}:
        raise ValueError("Available anchors refer to unknown graph nodes")
    trace, generator = [], random.Random(seed)
    status, winner = "budget_exhausted", None
    while True:
        bounds = ranking_bounds(graph, predictions, anchors, order=order)
        winner = certified_winner(bounds)
        if winner is not None:
            status = "ranking_certified"
            break
        if len(trace) >= budget:
            break
        components = feasible_components(graph, anchors)
        widths = {component["root"]: 0 for component in components}
        for pair in bounds["pairs"]:
            for component in pair["components"]:
                widths[component["root"]] += component["width_count"]
        eligible = [component for component in components if component["solution_count"] > 1
                    and widths[component["root"]] > 0
                    and available.intersection(component["node_ids"])]
        if not eligible:
            status = "anchors_unavailable"
            break
        if acquisition == "random":
            chosen = generator.choice(sorted(eligible, key=lambda component: component["root"]))
        elif acquisition == "disagreement":
            def disagreement(component):
                return sum(len({(type(predictions[candidate][node]).__name__, predictions[candidate][node])
                                for candidate in order}) - 1 for node in component["node_ids"])
            chosen = min(eligible, key=lambda component: (-disagreement(component), component["root"]))
        else:
            chosen = min(eligible, key=lambda component: (-widths[component["root"]], component["root"]))
        node = min(available.intersection(chosen["node_ids"]))
        attempt = {"query_index": len(trace), "node_id": node, "component_root": chosen["root"],
                   "ranking_width_count": widths[chosen["root"]], "status": "attempted"}
        trace.append(attempt)
        try:
            anchor = anchor_oracle(node)
            try:
                attempt["returned_anchor"] = json.loads(json.dumps(anchor, allow_nan=False))
            except (TypeError, ValueError):
                attempt["returned_anchor_repr"] = repr(anchor)[:1024]
            if not isinstance(anchor, dict) or anchor.get("node_id") != node:
                raise ValueError("Anchor oracle must return the requested node's value and provenance")
            feasible_components(graph, [*anchors, anchor])
        except InconsistentGraphError as exc:
            attempt.update(status="inconsistent", error=str(exc))
            raise AnchorVerificationError("Checked anchor contradicts the supplied relation graph", trace) from exc
        except Exception as exc:
            attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise AnchorVerificationError("Anchor verification failed", trace) from exc
        anchors.append(dict(anchor))
        attempt.update(status="returned", anchor=dict(anchor))
    selected = winner if winner is not None else raw
    regret_upper_count = max((_gap(bounds, other, selected)[1] for other in order), default=0)
    return {"selected_id": selected, "status": status, "anchors": anchors, "queries": trace,
            "verified_anchor_queries": len(trace), "anchor_budget": budget, "acquisition": acquisition,
            "initial_anchor_count": len(initial_anchors or []),
            "bounds": bounds, "worst_case_regret_upper": regret_upper_count / bounds["node_count"],
            "conditional_certificate": {
                "certified": winner is not None, "scope": "Fixed retained predictions on this generated graph only",
                "sound_relations_and_anchors_assumed": True, "private_query_accuracy_certified": False,
                "minimum_anchor_cost_guaranteed": False,
                "saved_teacher_generation_cost_claimed": False}}
