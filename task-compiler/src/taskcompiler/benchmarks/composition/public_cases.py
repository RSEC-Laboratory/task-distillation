"""Controlled public dialogues from declared slots and synthetic value seeds.

No model, filesystem, benchmark-input, or benchmark-label access occurs here.
The surface renderer expresses each event literally. Its agreement with the
event annotations covers this restricted language only, not arbitrary dialogue.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import json
import unicodedata

from taskcompiler.benchmarks.interventions.program import (
    execute_events, validate_group, validate_program, variants,
)
from .response_model import content_hash


SPLITS = ("optimization", "selection", "audit")
PATTERNS = ("assignment", "correction", "unaccepted_offer", "carry_over",
            "clear", "no_preference", "offer_correction", "clear_restore")
RENDERER_VERSION = "explicit_slot_events_v1"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _literal(value):
    return json.dumps(value, ensure_ascii=False)


def _valid_value(value):
    return (isinstance(value, str) and 0 < len(value) <= 160
            and value == value.strip() and value.casefold() != "dontcare"
            and all(not unicodedata.category(c).startswith("C") for c in value))


def _seed_pools(pack, seed):
    _require(isinstance(pack, dict) and isinstance(pack.get("groups"), list),
             "Require a public teacher pack with program and groups")
    _require(isinstance(pack.get("provenance"), dict) and pack["provenance"].get("kind") == "teacher_generated",
             "Input must declare description-only teacher generation provenance")
    program = deepcopy(validate_program(pack.get("program")))
    _require(set(program) == {"version", "rules", "slots"}
             and all(set(s) == {"name", "description", "is_categorical", "possible_values"}
                     and isinstance(s["description"], str) for s in program["slots"]),
             "Require only the declared public program and slot fields")
    _require(len(program["slots"]) <= 64 and all(len(s["name"]) <= 80 for s in program["slots"]),
             "Controlled renderer supports at most 64 slots with bounded names")
    slots = {s["name"]: s for s in program["slots"]}
    observed = {name: set() for name in slots}
    rejected = 0
    # Free utterances, teacher references, clauses, and student feedback are not
    # consulted. Existing event annotations supply candidate values, not labels.
    for group in pack["groups"]:
        events = group.get("events", []) if isinstance(group, dict) else []
        for event in events if isinstance(events, list) else []:
            if not isinstance(event, dict) or event.get("kind") != "set":
                continue
            name, values = event.get("slot"), event.get("values")
            if not isinstance(name, str) or name not in slots or not isinstance(values, list) or len(values) != 1 or not _valid_value(values[0]):
                rejected += 1
                continue
            if slots[name]["is_categorical"] and values[0] not in slots[name]["possible_values"]:
                rejected += 1
                continue
            observed[name].add(values[0])
    records, pools = [], {}
    for name, slot in sorted(slots.items()):
        candidates = set(observed[name])
        if slot["is_categorical"]:
            candidates.update(v for v in slot["possible_values"] if _valid_value(v))
        # Case/normalization-only variants cannot manufacture correction diversity.
        distinct = {}
        for value in sorted(candidates):
            distinct.setdefault(unicodedata.normalize("NFC", value).casefold(), value)
        values = sorted(distinct.values(), key=lambda value: (content_hash([seed, name, value]), value))[:8]
        _require(values, f"No usable public value seed for declared slot: {name}")
        pools[name] = sorted(values)
        records.append({"slot": name, "is_categorical": slot["is_categorical"],
                        "available_distinct_values": len(distinct), "retained_value_cap": 8,
                        "sampled_values": [{"value": value,
                            "sources": (["teacher_synthetic_set"] if value in observed[name] else [])
                                       + (["public_schema_enum"] if slot["is_categorical"] else [])}
                            for value in values]})
    seed_material = {"program": program, "observed_set_values": {k: sorted(v) for k, v in observed.items()}}
    return program, pools, records, content_hash(seed_material), rejected


def _recipes(scenario):
    primary, first, second = scenario["primary_slot"], scenario["first"], scenario["second"]
    a = ("USER", primary, "set", first)
    b = ("USER", primary, "set", second)
    offer = ("SYSTEM", primary, "set", first)
    clear = ("USER", primary, "clear", None)
    other = (("USER", scenario["secondary_slot"], "set", scenario["secondary_value"])
             if scenario["secondary_slot"] is not None else ("SYSTEM", None, "noop", None))
    return {"assignment": [a], "correction": [a, b],
            "unaccepted_offer": [offer, ("USER", None, "noop", None)],
            "carry_over": [a, other], "clear": [a, clear],
            "no_preference": [a, ("USER", primary, "dontcare", None)],
            "offer_correction": [a, b, offer], "clear_restore": [a, clear, b]}


def _utterance(action, surface):
    speaker, name, kind, value = action
    if kind == "noop":
        return ("I make no changes to any slot." if speaker == "USER"
                else "I have no new offers or information.")
    slot = _literal(name)
    if kind == "set" and speaker == "SYSTEM":
        return ((f"I offer {_literal(value)} for slot {slot}. This offer is unaccepted.") if surface == 0
                else (f"For slot {slot}, my unaccepted offer is {_literal(value)}."))
    if kind == "set":
        return (f"I set my value for slot {slot} to {_literal(value)}." if surface == 0
                else f"My specified value for slot {slot} is {_literal(value)}.")
    if kind == "clear":
        return (f"I withdraw my value for slot {slot}; leave that slot unassigned." if surface == 0
                else f"Remove my previously specified value for slot {slot}. I leave it unassigned.")
    _require(kind == "dontcare" and speaker == "USER", "Unsupported controlled event")
    return (f"I have no preference for slot {slot}." if surface == 0
            else f"Any value is acceptable to me for slot {slot}; I have no preference.")


def render_group(program, scenario, pattern, *, surface=0):
    """Return one exact surface sibling with explicit role/operation provenance."""
    _require(pattern in PATTERNS and type(surface) is int and surface in (0, 1),
             "Unsupported controlled pattern or surface")
    recipes = _recipes(scenario)
    # Every operation sibling shares this identity, independent of surface text.
    root = content_hash({"renderer": RENDERER_VERSION, "program": program,
                         "role_operation_recipes": {k: [list(a) for a in v] for k, v in recipes.items()}})
    actions = recipes[pattern]
    turns = [{"speaker": action[0], "utterance": _utterance(action, surface)} for action in actions]
    events = [{"turn": index, "slot": name, "kind": kind, "values": [value] if kind == "set" else []}
              for index, (_, name, kind, value) in enumerate(actions) if kind != "noop"]
    group = {"id": f"r{root[:24]}-{pattern}-s{surface}", "root_id": root, "family": pattern,
             "turns": turns, "events": events,
             "renderer": {"version": RENDERER_VERSION, "surface": surface, "scenario": deepcopy(scenario)}}
    validate_group(group, program)
    return group


def validate_controlled_group(group, program):
    """Reject changed surfaces/events; this is a renderer check, not an NLP judge."""
    recipe = group["renderer"]
    _require(recipe["version"] == RENDERER_VERSION, "Wrong controlled renderer version")
    expected = render_group(program, recipe["scenario"], group["family"], surface=recipe["surface"])
    _require(group == expected, "Controlled group differs from its complete rendering recipe")
    return group


def build_public_cases(teacher_pack, *, seed, roots_per_split=1, surface_variants=1):
    """Prepare grouped public scenarios without consulting model outcomes.

    The caller must retain provenance proving the input pack was generated only
    from the public description. A dict marker is not a privacy certificate.
    """
    _require(type(seed) is int and type(roots_per_split) is int and 1 <= roots_per_split <= 32,
             "Require integer seed and 1..32 roots per split")
    _require(type(surface_variants) is int and surface_variants in (1, 2), "Use one or two fixed surfaces")
    program, pools, value_records, seed_hash, rejected = _seed_pools(teacher_pack, seed)
    names, roots = sorted(pools), {}
    for position, name in enumerate(names):
        secondary = names[(position + 1) % len(names)] if len(names) > 1 else None
        for first, second in combinations(pools[name], 2):
            for other in pools[secondary] if secondary is not None else (None,):
                scenario = {"primary_slot": name, "first": first, "second": second,
                            "secondary_slot": secondary, "secondary_value": other}
                exemplar = render_group(program, scenario, "assignment")
                roots[exemplar["root_id"]] = scenario
    count = len(SPLITS) * roots_per_split
    _require(len(roots) >= count, f"Insufficient distinct public scenarios: need {count}, have {len(roots)}")
    chosen, occupied, collisions = [], set(), 0
    for root in sorted(roots, key=lambda root: (content_hash([seed, root]), root)):
        fingerprints = {content_hash(render_group(program, roots[root], pattern, surface=surface)["turns"])
                        for pattern in PATTERNS for surface in range(surface_variants)}
        if fingerprints & occupied:
            collisions += 1
            continue
        chosen.append(root)
        occupied.update(fingerprints)
        if len(chosen) == count:
            break
    _require(len(chosen) == count,
             f"Insufficient non-overlapping scenarios after frozen deterministic scan: need {count}, found {len(chosen)}")
    partitions, inventory = {key: [] for key in SPLITS}, []
    for index, root in enumerate(chosen):
        phase, scenario = SPLITS[index % len(SPLITS)], roots[root]
        groups = [render_group(program, scenario, pattern, surface=surface)
                  for pattern in PATTERNS for surface in range(surface_variants)]
        partitions[phase].extend(groups)
        inventory.append({"root_id": root, "split": phase, "scenario": deepcopy(scenario),
                          "group_ids": [group["id"] for group in groups]})
    # Any identical rendered dialogue is a sibling even when it was reached by
    # two different scenario recipes. Such a collision must never cross splits.
    seen = {}
    for phase, groups in partitions.items():
        for group in groups:
            fingerprint = content_hash(group["turns"])
            if fingerprint in seen:
                _require(seen[fingerprint] == phase, "Internal public scenario collision across splits")
            seen[fingerprint] = phase
    return {"version": 1, "kind": "controlled_public_dialogues", "program": program,
            "generation": {"renderer": RENDERER_VERSION, "seed": seed, "roots_per_split": roots_per_split,
                           "surface_variants": surface_variants, "model_calls": 0,
                           "candidate_roots": len(roots), "rejected_overlapping_roots": collisions,
                           "root_selection": "seeded hash order; retain complete roots with no reused rendered dialogue; round-robin split assignment"},
            "seed_provenance": {"public_seed_sha256": seed_hash, "rejected_set_seed_records": rejected,
                                "used_fields": ["program", "groups[].events[set].slot", "groups[].events[set].values"],
                                "free_teacher_utterances_reused": False},
            "value_pools": value_records, "roots": inventory, "splits": partitions,
            "references": {group["id"]: {state: execute_events(program, group, rules) for state, rules in variants().items()}
                           for groups in partitions.values() for group in groups},
            "coverage": {phase: {"roots": roots_per_split, "groups": len(groups),
                                  "slots_with_events": sorted({e["slot"] for g in groups for e in g["events"]})}
                         for phase, groups in partitions.items()},
            "limitations": ["The same deterministic language templates recur across all splits.",
                            "Distinct roots are slot/value scenarios, not independently sampled natural-language styles or human gold.",
                            "References are correct for the restricted rendered events; this does not validate free-form dialogue semantics.",
                            "A set seed is an opaque public string; domain plausibility beyond categorical membership is not inferred.",
                            "The generator never parses a private query or contacts the remote teacher."]}
