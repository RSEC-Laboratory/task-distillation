"""Finite identifiability checks for an explicitly bijective relation language.

These checks establish only what the supplied domains, relations and answer
anchors imply. They do not establish that natural-language inputs obey a
relation, or that a supplied relation/anchor describes the intended task.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import re

MAX_NODES = 256
MAX_DOMAIN_SIZE = 512
MAX_EDGES = 4096
MAX_ANCHORS = 4096
MAX_ABS_INTEGER = 10**12
MAX_LABEL_LENGTH = 256
MAX_IDENTIFIER_LENGTH = 128
MAX_SOURCE_LENGTH = 1024


class GraphValidationError(ValueError):
    """The graph is outside the supported finite, typed language."""


class InconsistentGraphError(ValueError):
    def __init__(self, audit):
        super().__init__("The supplied graph and anchors have no satisfying assignment")
        self.audit = audit


class AmbiguousGraphError(ValueError):
    def __init__(self, audit):
        super().__init__("The supplied graph and anchors do not identify unique labels")
        self.audit = audit


def _require(condition, message):
    if not condition:
        raise GraphValidationError(message)


def _object(value, keys, name):
    _require(type(value) is dict and set(value) == set(keys),
             f"{name} must contain exactly {sorted(keys)}")


def _list(value, maximum, name, *, minimum=0):
    _require(type(value) is list and minimum <= len(value) <= maximum,
             f"{name} must be a list with {minimum}..{maximum} items")


def _identifier(value):
    _require(type(value) is str and len(value) <= MAX_IDENTIFIER_LENGTH
             and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", value) is not None,
             "Node IDs must be bounded, nonempty ASCII identifiers")


def _value(value, name):
    _require(type(value) in (bool, int, str), f"{name} must be a bool, integer or string")
    if type(value) is int:
        _require(abs(value) <= MAX_ABS_INTEGER, f"{name} exceeds the integer bound")
    if type(value) is str:
        _require(len(value) <= MAX_LABEL_LENGTH, f"{name} exceeds the string bound")


def _key(value):
    # Python otherwise treats True and 1 as equal and as the same dictionary key.
    return type(value).__name__, value


def _hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    kind: str
    delta: int = 0
    mapping: dict | None = None
    inverse: dict | None = None

    def transform(self, value, forward):
        if self.kind == "identity":
            return value
        if self.kind == "boolean_not":
            return not value
        if self.kind == "integer_shift":
            return value + self.delta if forward else value - self.delta
        return (self.mapping if forward else self.inverse)[value]


def _prepare(graph, anchors):
    _object(graph, {"nodes", "edges"}, "graph")
    _list(graph["nodes"], MAX_NODES, "nodes", minimum=1)
    _list(graph["edges"], MAX_EDGES, "edges")
    domains, kinds = {}, {}
    for node in graph["nodes"]:
        _object(node, {"id", "domain"}, "node")
        name, domain = node["id"], node["domain"]
        _identifier(name)
        _require(name not in domains, "Duplicate node ID")
        _list(domain, MAX_DOMAIN_SIZE, f"domain of {name}", minimum=1)
        for value in domain:
            _value(value, f"domain value of {name}")
        _require(len({type(value) for value in domain}) == 1, "Domains must be homogeneous")
        _require(len({_key(value) for value in domain}) == len(domain), "Duplicate domain value")
        domains[name] = tuple(domain)
        kinds[name] = type(domain[0])
    edges = []
    for raw in graph["edges"]:
        _object(raw, {"source", "target", "operation"}, "edge")
        source, target, operation = raw["source"], raw["target"], raw["operation"]
        _identifier(source)
        _identifier(target)
        _require(source in domains and target in domains, "Edge references an unknown node")
        _require(type(operation) is dict and type(operation.get("kind")) is str,
                 "Operation must declare a supported kind")
        kind = operation["kind"]
        _require(kind in {"identity", "boolean_not", "integer_shift", "label_permutation"},
                 "Unknown operation kind; arbitrary code is not supported")
        fields = {"kind", "delta"} if kind == "integer_shift" else (
            {"kind", "mapping"} if kind == "label_permutation" else {"kind"})
        _object(operation, fields, "operation")
        _require(kinds[source] is kinds[target], "Related node domains must have the same type")
        if kind == "identity":
            edge = _Edge(source, target, kind)
        elif kind == "boolean_not":
            _require(kinds[source] is bool, "boolean_not requires Boolean domains")
            edge = _Edge(source, target, kind)
        elif kind == "integer_shift":
            _require(kinds[source] is int and type(operation["delta"]) is int,
                     "integer_shift requires integer domains and a non-Boolean integer delta")
            _require(abs(operation["delta"]) <= MAX_ABS_INTEGER, "Shift exceeds the integer bound")
            edge = _Edge(source, target, kind, delta=operation["delta"])
        else:
            _require(kinds[source] is str, "label_permutation requires string domains")
            mapping = operation["mapping"]
            _require(type(mapping) is dict and 1 <= len(mapping) <= MAX_DOMAIN_SIZE,
                     "Permutation must be a bounded mapping")
            for key, value in mapping.items():
                _require(type(key) is str and type(value) is str, "Permutation labels must be strings")
                _value(key, "permutation key")
                _value(value, "permutation value")
            _require(set(mapping) == set(mapping.values()) and len(set(mapping.values())) == len(mapping),
                     "label_permutation must be a bijection of its string label set")
            _require(set(domains[source]) <= set(mapping) and set(domains[target]) <= set(mapping),
                     "Permutation must cover both related node domains")
            edge = _Edge(source, target, kind, mapping=dict(mapping),
                         inverse={value: key for key, value in mapping.items()})
        edges.append(edge)
    anchors = [] if anchors is None else anchors
    _list(anchors, MAX_ANCHORS, "anchors")
    normalized = []
    for anchor in anchors:
        _object(anchor, {"node_id", "value", "source"}, "anchor")
        node, value, source = anchor["node_id"], anchor["value"], anchor["source"]
        _identifier(node)
        _require(node in domains, "Anchor references an unknown node")
        _value(value, "anchor value")
        _require(type(value) is kinds[node], "Anchor type differs from its node domain")
        _require(_key(value) in {_key(item) for item in domains[node]}, "Anchor is outside its node domain")
        _require(type(source) is str and bool(source.strip()) and len(source) <= MAX_SOURCE_LENGTH,
                 "Anchor source must be bounded, nonempty provenance text")
        normalized.append(dict(anchor))
    return domains, edges, normalized


def _analyze(graph, anchors):
    domains, edges, anchors = _prepare(graph, anchors)
    domain_keys = {node: {_key(value) for value in values} for node, values in domains.items()}
    adjacency = {node: [] for node in domains}
    for index, edge in enumerate(edges):
        adjacency[edge.source].append((edge.target, index, True))
        adjacency[edge.target].append((edge.source, index, False))
    components, unseen = [], set(domains)
    while unseen:
        seed = min(unseen)
        nodes, queue = set(), deque([seed])
        while queue:
            node = queue.popleft()
            if node in nodes:
                continue
            nodes.add(node)
            queue.extend(neighbor for neighbor, _, _ in adjacency[node] if neighbor not in nodes)
        unseen -= nodes
        root = min(nodes, key=lambda node: (len(domains[node]), node))
        relevant_anchors = [anchor for anchor in anchors if anchor["node_id"] in nodes]
        before, after = [], []
        for root_value in domains[root]:
            assignment, pending, consistent = {root: root_value}, deque([root]), True
            while pending and consistent:
                node = pending.popleft()
                for neighbor, index, forward in adjacency[node]:
                    value = edges[index].transform(assignment[node], forward)
                    if (_key(value) not in domain_keys[neighbor]
                            or (neighbor in assignment and _key(assignment[neighbor]) != _key(value))):
                        consistent = False
                        break
                    if neighbor not in assignment:
                        assignment[neighbor] = value
                        pending.append(neighbor)
            if consistent:
                before.append(assignment)
                if all(_key(assignment[a["node_id"]]) == _key(a["value"]) for a in relevant_anchors):
                    after.append(assignment)
        components.append({"root": root, "nodes": sorted(nodes), "before": before,
                           "after": after, "anchors": relevant_anchors})
    before_count = math.prod(len(component["before"]) for component in components)
    after_count = math.prod(len(component["after"]) for component in components)
    anchored_nodes = sorted({anchor["node_id"] for anchor in anchors})
    component_rows = [{
        "root": component["root"], "node_ids": component["nodes"],
        "solutions_before_anchors": len(component["before"]),
        "solutions_after_anchors": len(component["after"]),
        "anchored_node_ids": sorted({a["node_id"] for a in component["anchors"]}),
        "minimum_additional_anchors": (None if not component["after"] else int(len(component["after"]) > 1)),
    } for component in components]
    report = {
        "version": 1, "language": "finite_bijective_relations_v1",
        "status": "inconsistent" if after_count == 0 else "identified" if after_count == 1 else "ambiguous",
        "graph_hash": _hash(graph), "anchors_hash": _hash(anchors),
        "node_count": len(domains), "edge_count": len(edges), "component_count": len(components),
        "solutions_before_anchors": before_count, "solutions_after_anchors": after_count,
        "components": component_rows,
        "anchor_coverage": {
            "supplied_anchors": len(anchors), "directly_anchored_node_ids": anchored_nodes,
            "directly_anchored_nodes": len(anchored_nodes), "node_count": len(domains),
            "directly_anchored_fraction": len(anchored_nodes) / len(domains),
            "components_with_anchors": sum(bool(c["anchors"]) for c in components),
            "uniquely_identified_nodes_after_anchors": sum(len(c["nodes"]) for c in components if len(c["after"]) == 1),
            "unique_node_count_scope": "component_local_not_global_feasibility",
        },
        "minimum_additional_anchors": (None if not after_count else sum(len(c["after"]) > 1 for c in components)),
        "additional_anchor_interpretation": "One correct answer anchor at any node of each remaining ambiguous component is sufficient and necessary within this bijective DSL. An inconsistent system cannot be repaired by adding anchors.",
        "conditional_scope": "Assignment identifiability is conditional on the declared domains, relations and anchors. Relation soundness for natural-language inputs and anchor correctness are assumed, not proved.",
    }
    return report, components, domains, edges, anchors


def audit_graph(graph: dict, anchors: list[dict] | None = None) -> dict:
    """Count all satisfying assignments before and after partial answer supervision."""
    return _analyze(graph, anchors)[0]


def feasible_components(graph: dict, anchors: list[dict] | None = None) -> list[dict]:
    """Expose post-anchor component assignments without their Cartesian product.

    Each connected component has at most MAX_DOMAIN_SIZE assignments. Callers
    may minimize or maximize additive ranking contributions component by
    component. An inconsistent graph raises instead of exposing an empty set
    that might accidentally yield a vacuous ranking certificate. Returned
    dictionaries are independent copies and contain only immutable scalar values.
    """
    report, components, _, _, _ = _analyze(graph, anchors)
    if report["solutions_after_anchors"] == 0:
        raise InconsistentGraphError(report)
    return [{"root": component["root"], "node_ids": list(component["nodes"]),
             "solutions_before_anchors": len(component["before"]),
             "solution_count": len(component["after"]),
             "assignments": [dict(sorted(assignment.items())) for assignment in component["after"]],
             "anchored_node_ids": sorted({anchor["node_id"] for anchor in component["anchors"]})}
            for component in components]


def compile_graph(graph: dict, anchors: list[dict] | None = None) -> dict:
    """Return uniquely implied labels, or reject inconsistency/remaining ambiguity."""
    report, components, _, _, _ = _analyze(graph, anchors)
    if report["solutions_after_anchors"] == 0:
        raise InconsistentGraphError(report)
    if report["solutions_after_anchors"] != 1:
        raise AmbiguousGraphError(report)
    labels = {node: assignment[node] for component in components
              for assignment in component["after"] for node in component["nodes"]}
    labels = dict(sorted(labels.items()))
    return {
        "inferred_labels": labels,
        "conditional_certificate": {
            "kind": "unique_finite_assignment", "language": report["language"],
            "graph_hash": report["graph_hash"], "anchors_hash": report["anchors_hash"],
            "inferred_labels_hash": _hash(labels), "solution_count": 1,
            "soundness_assumed": True, "natural_language_input_semantics_proved": False,
            "scope": report["conditional_scope"],
        },
        "audit": report,
    }


certify_graph = compile_graph


def score_outputs(graph: dict, outputs: dict, anchors: list[dict] | None = None) -> dict:
    """Score declared relations and anchors separately on a full typed assignment.

    A perfect relation score alone is not evidence of identifiable or correct
    task labels. Missing, extra, wrong-type or out-of-domain outputs are errors.
    """
    domains, edges, anchors = _prepare(graph, anchors)
    _require(type(outputs) is dict and set(outputs) == set(domains),
             "Outputs must contain exactly one value for every declared node")
    for node, value in outputs.items():
        _value(value, "output value")
        _require(type(value) is type(domains[node][0]), "Output type differs from node domain")
        _require(_key(value) in {_key(item) for item in domains[node]}, "Output is outside its node domain")
    edge_rows = [{"index": index, "source": edge.source, "target": edge.target,
                  "passed": _key(edge.transform(outputs[edge.source], True)) == _key(outputs[edge.target])}
                 for index, edge in enumerate(edges)]
    anchor_rows = [{"index": index, "node_id": anchor["node_id"], "source": anchor["source"],
                    "passed": _key(outputs[anchor["node_id"]]) == _key(anchor["value"])}
                   for index, anchor in enumerate(anchors)]
    return {
        "edges": {"passed": sum(row["passed"] for row in edge_rows), "total": len(edge_rows), "rows": edge_rows},
        "anchors": {"passed": sum(row["passed"] for row in anchor_rows), "total": len(anchor_rows), "rows": anchor_rows},
        "all_satisfied": all(row["passed"] for row in edge_rows + anchor_rows),
        "scope": "Constraint satisfaction only; this does not prove assignment uniqueness or natural-language task correctness.",
    }
