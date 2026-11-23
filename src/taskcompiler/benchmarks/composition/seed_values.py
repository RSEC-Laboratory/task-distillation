"""Description-only synthetic values for the controlled composition grammar.

Validation is structural and categorical membership only. Values are opaque
public seeds, never annotations or independent gold for natural language.
"""
from __future__ import annotations

from copy import deepcopy
import json
import unicodedata

from taskcompiler.benchmarks.dialogue.dataset import validate_schemas
from taskcompiler.benchmarks.interventions.program import ORIGINAL, validate_group
from taskcompiler.procedural.schema import content_hash, strict_json_loads


PROTOCOL = "description_only_composition_values_v1"
SYSTEM = """Generate synthetic public values from ONLY the supplied public service description and slot schema.
Return one JSON object with exactly the key values. values maps EVERY declared slot
name to a list of strings; do not omit, rename, or add a slot. Do not return dialogue,
instructions, annotations, references, or other fields. For each noncategorical slot,
invent exactly FOUR distinct plausible values. For each categorical slot, choose
between TWO and FOUR distinct values copied exactly from its possible_values; if
the public enumeration contains only ONE value, return that one value. Strings must
be 1-160 characters, have no leading/trailing whitespace or control/format characters,
and must not equal dontcare, ignoring case. Distinct values must also differ after
Unicode NFC normalization and case folding. No previous examples, student outputs,
failures, or private queries are supplied. These are synthetic seeds, not NLP gold.
"""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def public_schema(schema):
    """Copy only public description and slot declarations; ignore extra inputs."""
    public = {key: deepcopy(schema[key]) for key in ("service_name", "description", "slots")}
    validate_schemas([{**public, "intents": []}])
    _require(len(public["slots"]) <= 64, "At most 64 public slots are supported")
    for slot in public["slots"]:
        _require(set(slot) == {"name", "description", "is_categorical", "possible_values"},
                 "Public slot declarations require exactly the supported fields")
        _require(len(slot["name"]) <= 80, "Public slot name exceeds the renderer bound")
        if slot["is_categorical"]:
            _require(slot["possible_values"], "Categorical slot has no public values")
            _require(all(valid_value(value) for value in slot["possible_values"]),
                     "Categorical enumeration contains an unusable seed value")
            distinct = {normalized(value) for value in slot["possible_values"]}
            _require(len(distinct) == len(slot["possible_values"]),
                     "Categorical enumeration contains normalized duplicates")
    return public


def normalized(value):
    return unicodedata.normalize("NFC", value).casefold()


def valid_value(value):
    return (isinstance(value, str) and 0 < len(value) <= 160 and value == value.strip()
            and value.casefold() != "dontcare"
            and all(not unicodedata.category(character).startswith("C") for character in value))


def teacher_request(schema, seed):
    _require(type(seed) is int, "Generation seed must be an integer")
    return SYSTEM, json.dumps({"protocol": PROTOCOL, "generation_seed": seed,
                               "public_service": public_schema(schema)},
                              ensure_ascii=False, sort_keys=True, allow_nan=False)


def response_schema(schema):
    """Common exact JSON contract, independent of outputs and previous failures."""
    public = public_schema(schema)
    slots = {}
    for slot in public["slots"]:
        categorical = slot["is_categorical"]
        count = len(slot["possible_values"])
        item = {"type": "string", "minLength": 1, "maxLength": 160}
        if categorical:
            item["enum"] = deepcopy(slot["possible_values"])
        slots[slot["name"]] = {"type": "array", "items": item, "uniqueItems": True,
                              "minItems": (1 if count == 1 else 2) if categorical else 4,
                              "maxItems": min(4, count) if categorical else 4}
    return {"type": "object", "properties": {"values": {"type": "object", "properties": slots,
            "required": list(slots), "additionalProperties": False}},
            "required": ["values"], "additionalProperties": False}


def validate_values(text, schema):
    public = public_schema(schema)
    body = strict_json_loads(text)
    _require(isinstance(body, dict) and set(body) == {"values"}, "Expected exactly the values field")
    values = body["values"]
    names = {slot["name"] for slot in public["slots"]}
    _require(isinstance(values, dict) and set(values) == names, "Values must cover every declared public slot exactly")
    for slot in public["slots"]:
        candidates = values[slot["name"]]
        _require(isinstance(candidates, list) and all(valid_value(value) for value in candidates),
                 "Invalid bounded public seed string for slot: " + slot["name"])
        _require(len({normalized(value) for value in candidates}) == len(candidates),
                 "Normalized duplicate public seed for slot: " + slot["name"])
        if slot["is_categorical"]:
            count = len(slot["possible_values"])
            minimum = 1 if count == 1 else 2
            _require(minimum <= len(candidates) <= min(4, count)
                     and all(value in slot["possible_values"] for value in candidates),
                     "Categorical seeds must use the declared public enumeration: " + slot["name"])
        else:
            _require(len(candidates) == 4, "Noncategorical slots require exactly four values: " + slot["name"])
    return deepcopy(values)


def make_seed_pack(values, schema, *, seed, teacher_identity, request_sha256, response_sha256):
    """Turn already validated declarative strings into explicit, typed set events."""
    public = public_schema(schema)
    checked = validate_values(json.dumps({"values": values}, ensure_ascii=False), public)
    _require(type(seed) is int, "Generation seed must be an integer")
    program = {"version": 1, "rules": deepcopy(ORIGINAL), "slots": deepcopy(public["slots"])}
    groups = []
    for slot_index, slot in enumerate(public["slots"]):
        for value_index, value in enumerate(checked[slot["name"]]):
            group = {"id": f"seed-{slot_index}-{value_index}", "family": "declarative_value_seed",
                     "turns": [{"speaker": "USER", "utterance": "I explicitly set slot "
                                + json.dumps(slot["name"], ensure_ascii=False) + " to "
                                + json.dumps(value, ensure_ascii=False) + "."}],
                     "events": [{"turn": 0, "slot": slot["name"], "kind": "set", "values": [value]}]}
            validate_group(group, program)
            groups.append(group)
    return {"program": program, "groups": groups,
            "provenance": {"kind": "teacher_generated", "protocol": PROTOCOL,
                           "material": "declarative_values", "seed": seed,
                           "teacher_identity": deepcopy(teacher_identity),
                           "public_description_sha256": content_hash(public),
                           "teacher_request_sha256": request_sha256,
                           "teacher_response_sha256": response_sha256,
                           "event_origin": "Literal code conversion of validated public value strings",
                           "semantic_status": "Opaque synthetic seeds; not independent NLP annotations or gold"}}
