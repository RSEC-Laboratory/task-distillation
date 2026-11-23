"""Validation for description-only procedural task packs.

The procedure language is a DAG of instructions for a language model. Validating
it proves structural consistency, not that its natural-language steps are true,
executable by this module, or sufficient to solve the task.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

_MAX_FILE_BYTES = 32 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_SCHEMA_KEYS = {"type", "enum", "properties", "required", "additionalProperties", "items"}
_PACK_KEYS = {
    "version", "task_id", "description", "output_schema", "requirements",
    "procedures", "teacher_instruction", "wording_candidates", "cases", "provenance",
}


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _json_value(value: Any, path: str = "$", depth: int = 0) -> None:
    if depth > 40:
        _fail(path, "JSON nesting exceeds 40 levels")
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) in (int, float):
        if isinstance(value, float) and not math.isfinite(value):
            _fail(path, "non-finite numbers are not JSON values")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "JSON object keys must be strings")
            _json_value(item, f"{path}.{key}", depth + 1)
        return
    _fail(path, f"unsupported JSON value {type(value).__name__}")


def content_hash(value: Any) -> str:
    """SHA-256 of canonical JSON; description provenance uses raw UTF-8 instead."""
    _json_value(value)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: Any, path: str, keys: set[str] | None = None) -> dict:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    if keys is not None and set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        _fail(path, f"incorrect keys (missing={missing}, unexpected={extra})")
    return value


def _text(value: Any, path: str, maximum: int = 20_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        _fail(path, f"expected nonempty text of at most {maximum} characters")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        _fail(path, "expected a 1-128 character identifier using letters, digits, _, ., or -")
    return value


def _list(value: Any, path: str, minimum: int, maximum: int) -> list:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        _fail(path, f"expected a list with {minimum}-{maximum} entries")
    return value


def _types(schema: dict) -> list[str]:
    value = schema["type"]
    return [value] if isinstance(value, str) else value


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return left == right


def _validate_schema(schema: Any, path: str = "$schema", depth: int = 0) -> None:
    if depth > 20:
        _fail(path, "schema nesting exceeds 20 levels")
    _object(schema, path)
    unknown = set(schema) - _SCHEMA_KEYS
    if unknown:
        _fail(path, f"unsupported schema keywords: {sorted(unknown)}")
    if "type" not in schema:
        _fail(path, "type is required")
    types = schema["type"]
    if isinstance(types, str):
        types = [types]
    if (
        not isinstance(types, list)
        or not types
        or any(not isinstance(item, str) or item not in _TYPES for item in types)
        or len(set(types)) != len(types)
    ):
        _fail(path, "type must be a supported type or a list of distinct supported types")
    if "object" in types:
        if "properties" not in schema or "required" not in schema:
            _fail(path, "object schemas require properties and required")
        properties = _object(schema["properties"], f"{path}.properties")
        if len(properties) > 128:
            _fail(path, "at most 128 object properties are supported")
        required = _list(schema["required"], f"{path}.required", 0, 128)
        if any(not isinstance(key, str) for key in required) or len(set(required)) != len(required):
            _fail(path, "required must contain distinct property names")
        if set(required) - set(properties):
            _fail(path, "required contains an unknown property")
        if schema.get("additionalProperties") is not False:
            _fail(path, "object schemas must set additionalProperties to false")
        for key, child in properties.items():
            _text(key, f"{path}.properties key", maximum=128)
            _validate_schema(child, f"{path}.properties.{key}", depth + 1)
    elif set(schema) & {"properties", "required", "additionalProperties"}:
        _fail(path, "object keywords require object in type")
    if "array" in types:
        if "items" not in schema:
            _fail(path, "array schemas require items")
        _validate_schema(schema["items"], f"{path}.items", depth + 1)
    elif "items" in schema:
        _fail(path, "items requires array in type")
    if "enum" in schema:
        values = _list(schema["enum"], f"{path}.enum", 1, 256)
        without_enum = {key: value for key, value in schema.items() if key != "enum"}
        for index, value in enumerate(values):
            _json_value(value, f"{path}.enum[{index}]")
            _check_output(value, without_enum, f"{path}.enum[{index}]")
            if any(_json_equal(value, earlier) for earlier in values[:index]):
                _fail(path, "enum contains duplicate values")


def _has_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return type(value) is bool
    if type_name == "integer":
        return type(value) is int or (
            type(value) is float and math.isfinite(value) and value.is_integer()
        )
    if type_name == "number":
        return type(value) in (int, float) and (
            type(value) is int or math.isfinite(value)
        )
    return isinstance(value, {"object": dict, "array": list, "string": str}[type_name])


def _check_output(value: Any, schema: dict, path: str) -> None:
    if not any(_has_type(value, kind) for kind in _types(schema)):
        _fail(path, f"value does not match type {schema['type']!r}")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        _fail(path, "value is not in enum")
    if isinstance(value, dict):
        properties = schema["properties"]
        missing = set(schema["required"]) - set(value)
        extra = set(value) - set(properties)
        if missing or extra:
            _fail(path, f"incorrect output keys (missing={sorted(missing)}, unexpected={sorted(extra)})")
        for key, item in value.items():
            _check_output(item, properties[key], f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_output(item, schema["items"], f"{path}[{index}]")


def validate_output(value: Any, schema: dict) -> None:
    """Validate shape and primitive constraints, never semantic answer accuracy.

    Supported keywords are type (including unions), enum, properties, required,
    additionalProperties:false, and items. Every object is closed explicitly.
    """
    _json_value(schema, "$schema")
    _validate_schema(schema)
    _json_value(value)
    _check_output(value, schema, "$")


def _validate_cases(cases: Any, schema: dict | None = None) -> None:
    cases = _list(cases, "cases", 3, 10_000)
    ids: set[str] = set()
    groups: set[str] = set()
    normalized_inputs: set[str] = set()
    for index, case in enumerate(cases):
        path = f"cases[{index}]"
        _object(case, path, {"id", "group_id", "input", "reference"})
        case_id = _identifier(case["id"], f"{path}.id")
        if case_id in ids:
            _fail(path, f"duplicate case id {case_id!r}")
        ids.add(case_id)
        group = _identifier(case["group_id"], f"{path}.group_id")
        groups.add(group)
        source = _text(case["input"], f"{path}.input", maximum=200_000)
        normalized = " ".join(unicodedata.normalize("NFC", source).split())
        if normalized in normalized_inputs:
            _fail(path, "normalized duplicate input would distort case weighting or conflict in references")
        normalized_inputs.add(normalized)
        _json_value(case["reference"], f"{path}.reference")
        if schema is not None:
            _check_output(case["reference"], schema, f"{path}.reference")
    if len(groups) < 3:
        _fail("cases", "at least three distinct scenario groups are required")


def validate_task_pack(pack: dict) -> None:
    """Reject malformed or inconsistent packs without claiming semantic verification."""
    _json_value(pack)
    _object(pack, "task_pack", _PACK_KEYS)
    if type(pack["version"]) is not int or pack["version"] != 1:
        _fail("version", "expected integer 1")
    _identifier(pack["task_id"], "task_id")
    description = _text(pack["description"], "description", maximum=200_000)
    _validate_schema(pack["output_schema"])
    requirements = _list(pack["requirements"], "requirements", 1, 128)
    requirement_ids: set[str] = set()
    for index, requirement in enumerate(requirements):
        path = f"requirements[{index}]"
        _object(requirement, path, {"id", "text"})
        identifier = _identifier(requirement["id"], f"{path}.id")
        if identifier in requirement_ids:
            _fail(path, f"duplicate requirement id {identifier!r}")
        requirement_ids.add(identifier)
        _text(requirement["text"], f"{path}.text")

    procedures = _list(pack["procedures"], "procedures", 2, 16)
    procedure_ids: set[str] = set()
    for index, procedure in enumerate(procedures):
        path = f"procedures[{index}]"
        _object(procedure, path, {"id", "steps"})
        identifier = _identifier(procedure["id"], f"{path}.id")
        if identifier in procedure_ids:
            _fail(path, f"duplicate procedure id {identifier!r}")
        procedure_ids.add(identifier)
        steps = _list(procedure["steps"], f"{path}.steps", 1, 128)
        earlier: set[str] = set()
        covered: set[str] = set()
        for step_index, step in enumerate(steps):
            step_path = f"{path}.steps[{step_index}]"
            _object(step, step_path, {"id", "instruction", "requires", "covers"})
            step_id = _identifier(step["id"], f"{step_path}.id")
            if step_id in earlier:
                _fail(step_path, f"duplicate step id {step_id!r}")
            _text(step["instruction"], f"{step_path}.instruction")
            requires = _list(step["requires"], f"{step_path}.requires", 0, 128)
            covers = _list(step["covers"], f"{step_path}.covers", 0, 128)
            for field, values in (("requires", requires), ("covers", covers)):
                for item in values:
                    _identifier(item, f"{step_path}.{field}")
                if len(set(values)) != len(values):
                    _fail(step_path, f"duplicate {field} identifiers")
            if set(requires) - earlier:
                _fail(step_path, "dependencies must refer to earlier steps; forward, cyclic, and unknown references are invalid")
            if set(covers) - requirement_ids:
                _fail(step_path, "covers contains an unknown requirement")
            earlier.add(step_id)
            covered.update(covers)
        if covered != requirement_ids:
            _fail(path, f"procedure does not cover every requirement: {sorted(requirement_ids - covered)}")

    _text(pack["teacher_instruction"], "teacher_instruction", maximum=100_000)
    wordings = _list(pack["wording_candidates"], "wording_candidates", 2, 16)
    if len(wordings) != len(procedures):
        _fail("wording_candidates", "must have the same candidate count as procedures")
    for index, wording in enumerate(wordings):
        _text(wording, f"wording_candidates[{index}]", maximum=100_000)
    if len(set(wordings)) != len(wordings):
        _fail("wording_candidates", "duplicate candidates are not permitted")
    _validate_cases(pack["cases"], pack["output_schema"])
    provenance = _object(
        pack["provenance"], "provenance",
        {"kind", "description_sha256", "teacher_model", "generation_seed"},
    )
    if provenance["kind"] not in {"teacher_generated", "fixture"}:
        _fail("provenance.kind", "expected teacher_generated or fixture")
    expected_hash = hashlib.sha256(description.encode("utf-8")).hexdigest()
    if provenance["description_sha256"] != expected_hash:
        _fail("provenance.description_sha256", "does not match the exact UTF-8 task description")
    _text(provenance["teacher_model"], "provenance.teacher_model", maximum=256)
    if type(provenance["generation_seed"]) is not int:
        _fail("provenance.generation_seed", "expected an integer")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def strict_json_loads(text: str | bytes | bytearray) -> Any:
    """Parse bounded JSON without duplicate keys, non-finite values, or deep nesting.

    This is shared by task-pack ingestion and model-output scoring so malformed
    output cannot be accepted through a more permissive parser in another stage.
    """
    if not isinstance(text, (str, bytes, bytearray)):
        raise ValueError("JSON source must be text or bytes")
    if len(text) > _MAX_FILE_BYTES:
        raise ValueError("JSON source exceeds 32 MiB/characters")
    try:
        value = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds supported limits") from exc
    _json_value(value)
    return value


def load_task_pack(path: Path) -> dict:
    """Read an immutable input pack; duplicate JSON keys are rejected."""
    path = Path(path)
    if path.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError("task pack exceeds 32 MiB")
    pack = strict_json_loads(path.read_text(encoding="utf-8"))
    validate_task_pack(pack)
    return pack


def split_cases(cases: list[dict], seed: int) -> dict[str, list[dict]]:
    """Deterministic approximately 60/20/20 split of entire scenario groups.

    Membership is invariant to case order. At least one group is reserved for
    each partition; proportions concern groups rather than individual examples.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    _validate_cases(cases)
    groups = sorted({case["group_id"] for case in cases})
    random.Random(seed).shuffle(groups)
    quotas = [len(groups) * fraction for fraction in (0.6, 0.2, 0.2)]
    sizes = [max(1, math.floor(quota)) for quota in quotas]
    while sum(sizes) < len(groups):
        chosen = max(range(3), key=lambda index: quotas[index] - sizes[index])
        sizes[chosen] += 1
    while sum(sizes) > len(groups):
        chosen = max(
            (index for index in range(3) if sizes[index] > 1),
            key=lambda index: sizes[index] - quotas[index],
        )
        sizes[chosen] -= 1
    partitions = {}
    offset = 0
    for name, size in zip(("optimization", "selection", "synthetic_holdout"), sizes):
        members = set(groups[offset:offset + size])
        partitions[name] = [case for case in cases if case["group_id"] in members]
        offset += size
    return partitions


def validate_prepared_splits(cases: list[dict], partitions: dict) -> dict[str, list[dict]]:
    """Validate externally prepared partitions without changing their membership.

    Every original case must appear exactly once, unchanged, and scenario groups
    must remain intact. This supports protocols with different declared splits;
    it does not establish independence or correctness of generated references.
    """
    _validate_cases(cases)
    names = {"optimization", "selection", "synthetic_holdout"}
    if not isinstance(partitions, dict) or set(partitions) != names:
        raise ValueError("Prepared splits require optimization, selection, and synthetic_holdout")
    original = {case["id"]: content_hash(case) for case in cases}
    seen: set[str] = set()
    group_partition: dict[str, str] = {}
    result = {}
    for name in ("optimization", "selection", "synthetic_holdout"):
        rows = partitions[name]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Prepared split {name} must be a nonempty list")
        result[name] = []
        for case in rows:
            if not isinstance(case, dict) or case.get("id") not in original:
                raise ValueError("Prepared splits contain an unknown case")
            identifier = case["id"]
            if identifier in seen:
                raise ValueError("Prepared splits contain a repeated case")
            if content_hash(case) != original[identifier]:
                raise ValueError("Prepared splits changed original case content")
            seen.add(identifier)
            group = case["group_id"]
            if group in group_partition and group_partition[group] != name:
                raise ValueError("Prepared splits divide a scenario group")
            group_partition[group] = name
            result[name].append(json.loads(json.dumps(case, ensure_ascii=False)))
    if seen != set(original):
        raise ValueError("Prepared splits must cover every original case")
    return result
