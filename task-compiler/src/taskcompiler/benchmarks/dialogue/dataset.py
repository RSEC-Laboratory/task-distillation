"""Pinned SGD development data with separate public schemas and local labels.

No test dialogues are acquired. Selection hashes dialogue identities only.
The label-loading API enforces a procedural freeze record, not filesystem access
control: researchers can still open the separately retained source annotations.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

SGD_COMMIT = "e852981ae34990f4358979625854259302feaa78"
METRICS_COMMIT = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
PROTOCOL = "sgd_development_v1"
DEFAULT_SEED = 20261005
MAX_SOURCE_BYTES = 100_000_000


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def strict_json(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def bad(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def _write_new_or_equal(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"Existing immutable artifact differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def source_catalog():
    from .catalog import SOURCES
    return deepcopy(list(SOURCES))


def acquire_sources(cache_root, *, fetch=None, sources=None):
    """Download/reuse exact official bytes with a fixed total footprint bound."""
    root = Path(cache_root).resolve()
    records = source_catalog() if sources is None else sources
    if sum(record["bytes"] for record in records) > MAX_SOURCE_BYTES:
        raise ValueError("Pinned acquisition exceeds the declared disk bound")
    def download(url):
        request = Request(url, headers={"User-Agent": "taskcompiler-sgd-acquisition/1"})
        with urlopen(request, timeout=60) as response:
            return response.read(MAX_SOURCE_BYTES + 1)
    fetch = download if fetch is None else fetch
    for record in records:
        destination = (root / record["cache_path"]).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("Source cache path escapes cache root")
        data = destination.read_bytes() if destination.exists() else fetch(record["url"])
        if len(data) != record["bytes"] or sha256(data) != record["sha256"]:
            raise ValueError(f"Pinned official bytes differ: {record['path']}")
        _write_new_or_equal(destination, data)
    return records


def validate_schemas(rows):
    """Retain every declared slot; these are public definitions, not gold states."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Schemas must be a nonempty list")
    result = {}
    for service in rows:
        name = service.get("service_name")
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("Invalid or duplicate service name")
        slots, intents = service.get("slots"), service.get("intents")
        if not isinstance(service.get("description"), str) or not isinstance(slots, list) or not slots:
            raise ValueError("Service requires a description and nonempty slots")
        if not isinstance(intents, list):
            raise ValueError("Service intents must be a list")
        seen = set()
        for slot in slots:
            key = slot.get("name")
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError("Invalid or duplicate slot name")
            if (type(slot.get("is_categorical")) is not bool
                    or not isinstance(slot.get("description"), str)
                    or not isinstance(slot.get("possible_values"), list)
                    or any(not isinstance(x, str) for x in slot["possible_values"])):
                raise ValueError("Malformed slot definition")
            seen.add(key)
        intent_names = [item.get("name") for item in intents]
        if any(not isinstance(x, str) or not x for x in intent_names) or len(set(intent_names)) != len(intent_names):
            raise ValueError("Malformed intent definitions")
        result[name] = service
    return result


def select_dialogues(dialogues, *, count, seed):
    """Identity-only hash ranking; no reference state or model result is read."""
    if type(seed) is not int or type(count) is not int or not 0 < count <= len(dialogues):
        raise ValueError("Require integer seed and positive count within the source")
    identities = [row.get("dialogue_id") for row in dialogues]
    if any(not isinstance(x, str) or not x for x in identities) or len(set(identities)) != len(identities):
        raise ValueError("Invalid or duplicated dialogue identity")
    return sorted(dialogues, key=lambda row: (
        sha256(f"{seed}\0{row['dialogue_id']}".encode()), row["dialogue_id"]))[:count]


def select_services(schemas, *, count, seed):
    """Hash public schema identities; no dialogue annotation influences services."""
    if type(count) is not int or not 0 < count <= len(schemas) or type(seed) is not int:
        raise ValueError("Invalid public service selection count/seed")
    return sorted(schemas, key=lambda name: (
        sha256(f"{seed}\0service\0{name}".encode()), name))[:count]


def select_service_dialogues(dialogues, services, *, count, seed, max_history_chars):
    """Choose dialogues for each preselected service using text-only eligibility."""
    if type(count) is not int or count < 1 or type(max_history_chars) is not int or max_history_chars < 1:
        raise ValueError("Require positive dialogue/history limits")
    identities = [row.get("dialogue_id") for row in dialogues]
    if any(not isinstance(x, str) or not x for x in identities) or len(set(identities)) != len(identities):
        raise ValueError("Invalid or duplicated dialogue identity")
    eligible = {name: [] for name in services}
    for dialogue in dialogues:
        history, supported = [], set()
        for turn in dialogue["turns"]:
            history.append(f"{turn['speaker']}: {turn['utterance']}")
            if turn["speaker"] == "USER" and len("\n".join(history)) <= max_history_chars:
                supported.update(frame["service"] for frame in turn.get("frames", [])
                                 if frame["service"] in eligible)
        for name in supported:
            eligible[name].append(dialogue)
    selected, per_service = {}, {}
    for name in services:
        ranked = sorted(eligible[name], key=lambda row: (
            sha256(f"{seed}\0{name}\0{row['dialogue_id']}".encode()), row["dialogue_id"]))[:count]
        per_service[name] = {"eligible_dialogue_count": len(eligible[name]),
                             "selected_dialogue_ids": [row["dialogue_id"] for row in ranked],
                             "shortfall": max(0, count - len(ranked))}
        for row in ranked:
            selected[row["dialogue_id"]] = row
    return [selected[key] for key in sorted(selected)], per_service


def project_dialogues(dialogues, schemas):
    """Expose only dialogue text, target service and stable evaluation identity."""
    inputs, labels = [], []
    for dialogue in dialogues:
        history = []
        for turn_index, turn in enumerate(dialogue["turns"]):
            if turn.get("speaker") not in {"USER", "SYSTEM"} or not isinstance(turn.get("utterance"), str):
                raise ValueError("Malformed dialogue text")
            history.append({"speaker": turn["speaker"], "utterance": turn["utterance"]})
            if turn["speaker"] != "USER":
                continue
            services_seen = set()
            for frame in turn.get("frames", []):
                service = frame.get("service")
                if service not in schemas or service in services_seen:
                    raise ValueError("Unknown or duplicate service frame")
                services_seen.add(service)
                identifier = f"{dialogue['dialogue_id']}:{turn_index}:{service}"
                input_text = "\n".join(f"{row['speaker']}: {row['utterance']}" for row in history)
                inputs.append({"id": identifier, "benchmark": "sgd", "task": service,
                               "dialogue_id": dialogue["dialogue_id"], "turn_index": turn_index,
                               "service": service, "input": input_text,
                               "history": [dict(row) for row in history]})
                # These annotations are written only into the separate scoring artifact.
                labels.append({"id": identifier, "dialogue_id": dialogue["dialogue_id"],
                               "service": service, "state": frame["state"]})
    if not inputs:
        raise ValueError("No user service frames in the selected dialogues")
    if len({row["id"] for row in inputs}) != len(inputs):
        raise ValueError("Duplicated projected frame identity")
    return inputs, labels


def public_service_description(service):
    """The ONLY task-specific benchmark material permitted in teacher requests."""
    validate_schemas([service])
    return ("Track this service's dialogue state from the complete conversation. "
            "Use the latest user-confirmed constraints, including corrections; carry "
            "earlier constraints unless changed. Return only a JSON object with key "
            "slot_values. slot_values maps assigned slot names to one-element lists "
            "of strings. Omit unassigned slots. Use dontcare for an explicit lack of "
            "preference. Do not treat system suggestions as accepted constraints.\n"
            "Public service schema:\n" + json.dumps(service, ensure_ascii=False, sort_keys=True))


def _jsonl(rows):
    return b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for row in rows)


def prepare_development(cache_root, output_dir, *, service_count=4, dialogues_per_service=8,
                        max_history_chars=12000, seed=DEFAULT_SEED, fetch=None):
    """Prepare a local dev subset; no official test dialogue is acquired or read."""
    root, out = Path(cache_root).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError("Prepared dialogue output must be a new directory")
    if out == root or not out.is_relative_to(root) or out.is_relative_to(root / "sources"):
        raise ValueError("Output must be a new child outside the source cache")
    records = acquire_sources(root / "sources", fetch=fetch)
    by_path = {record["path"]: record for record in records if record["repository"].endswith("dstc8-schema-guided-dialogue")}
    read = lambda path: strict_json((root / "sources" / by_path[path]["cache_path"]).read_bytes())
    schemas = validate_schemas(read("dev/schema.json"))
    train_services = set(validate_schemas(read("train/schema.json")))
    dialogues = []
    for path in sorted(by_path):
        if path.startswith("dev/dialogues_"):
            dialogues.extend(read(path))
    selected_services = select_services(schemas, count=service_count, seed=seed)
    chosen, per_service = select_service_dialogues(dialogues, selected_services,
        count=dialogues_per_service, seed=seed, max_history_chars=max_history_chars)
    all_inputs, all_labels = project_dialogues(chosen, schemas)
    eligible_ids = {row["id"] for row in all_inputs if row["service"] in selected_services
                    and len(row["input"]) <= max_history_chars}
    inputs = [row for row in all_inputs if row["id"] in eligible_ids]
    labels = [row for row in all_labels if row["id"] in eligible_ids]
    if not inputs:
        raise ValueError("No text-eligible frames for the selected public services")
    # Validate reference state types outside any candidate optimization interface.
    from .metrics import validate_reference
    for row in labels:
        validate_reference(row["state"], schemas[row["service"]])
    used = Counter(row["service"] for row in inputs)
    slot_types = Counter("categorical" if slot["is_categorical"] else "noncategorical"
                         for name in used for slot in schemas[name]["slots"])
    outputs = {"public_schemas.json": json_bytes({"protocol": PROTOCOL, "schemas": list(schemas.values())}),
               "inputs.jsonl": _jsonl(inputs), "sealed_labels.jsonl": _jsonl(labels)}
    coverage = {"source_dialogue_count": len(dialogues), "selected_dialogue_count": len(chosen),
                "selected_frame_count": len(inputs), "source_services": sorted(schemas),
                "selected_services": selected_services, "dialogues_per_service": per_service,
                "requested_dialogues_per_service": dialogues_per_service,
                "max_history_chars": max_history_chars, "history_truncated": False,
                "length_excluded_frame_ids": [row["id"] for row in all_inputs
                    if row["service"] in selected_services and len(row["input"]) > max_history_chars],
                "other_service_frame_count": sum(row["service"] not in selected_services for row in all_inputs),
                "selected_service_frames": dict(sorted(used.items())),
                "absent_services": sorted(set(schemas) - set(used)),
                "selected_service_slot_types": dict(slot_types),
                "selected_unseen_services": sorted(set(used) - train_services),
                "selection_uses_labels": False, "domain_whitelist": None,
                "all_declared_slots_scored": True}
    manifest = {"protocol": PROTOCOL, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "development", "official_split": "dev", "seed": seed,
                "selection": "public services: sha256(seed+NUL+service+NUL+name); dialogues per service: sha256(seed+NUL+name+NUL+dialogue_id); ascending, ties by ID",
                "selected_dialogue_ids": [row["dialogue_id"] for row in chosen],
                "sources": records, "coverage": coverage,
                "files": {name: {"sha256": sha256(data), "bytes": len(data)} for name, data in outputs.items()},
                "teacher_access": "public service definitions only; no real dialogue, state, annotation, output or score",
                "label_access": "Local scoring after instruction freeze; procedural separation, not filesystem security",
                "metric": "Native SGD exact-mode slot matching; not directly comparable with fuzzy leaderboard scores",
                "licenses": {"dataset": "CC BY-SA 4.0", "official_metric_source": "Apache-2.0"},
                "claim_scope": "Outcome-independent development subset, not an unseen final benchmark or privacy guarantee"}
    out.mkdir(parents=True)
    for name, data in outputs.items():
        _write_new_or_equal(out / name, data)
    _write_new_or_equal(out / "manifest.json", json_bytes(manifest))
    return manifest


def _prepared_file(manifest_path, name):
    path = Path(manifest_path).resolve()
    manifest = strict_json(path.read_bytes())
    if manifest.get("protocol") != PROTOCOL or manifest.get("official_split") != "dev":
        raise ValueError("Unsupported dialogue dataset protocol/split")
    candidate = path.parent / name
    data = candidate.read_bytes()
    record = manifest["files"][name]
    if sha256(data) != record["sha256"] or len(data) != record["bytes"]:
        raise ValueError(f"Prepared artifact checksum mismatch: {name}")
    return manifest, data


def load_inputs(manifest_path):
    """Load schema and dialogue text WITHOUT opening the sealed label artifact."""
    manifest, data = _prepared_file(manifest_path, "inputs.jsonl")
    _, schema_data = _prepared_file(manifest_path, "public_schemas.json")
    rows = [strict_json(line) for line in data.splitlines() if line]
    schemas = validate_schemas(strict_json(schema_data)["schemas"])
    if len(rows) != manifest["coverage"]["selected_frame_count"] or len({x["id"] for x in rows}) != len(rows):
        raise ValueError("Dialogue input denominator mismatch")
    allowed = {"id", "benchmark", "task", "dialogue_id", "turn_index", "service", "input", "history"}
    for row in rows:
        if set(row) != allowed or row["service"] not in schemas:
            raise ValueError("Unexpected field or service in dialogue input")
        if (row["benchmark"] != "sgd" or row["task"] != row["service"]
                or type(row["turn_index"]) is not int or row["turn_index"] < 0
                or not isinstance(row["dialogue_id"], str)
                or row["id"] != f"{row['dialogue_id']}:{row['turn_index']}:{row['service']}"):
            raise ValueError("Dialogue input identity mismatch")
        history = row["history"]
        if not isinstance(history, list) or len(history) != row["turn_index"] + 1:
            raise ValueError("Dialogue history is incomplete")
        for turn in history:
            if (not isinstance(turn, dict) or set(turn) != {"speaker", "utterance"}
                    or turn["speaker"] not in {"USER", "SYSTEM"}
                    or not isinstance(turn["utterance"], str)):
                raise ValueError("Unexpected annotation in dialogue history")
        text = "\n".join(f"{turn['speaker']}: {turn['utterance']}" for turn in history)
        if history[-1]["speaker"] != "USER" or row["input"] != text:
            raise ValueError("Dialogue text/history mismatch")
    return manifest, schemas, rows


def load_labels(manifest_path, *, freeze):
    """Open labels only after a declared instruction commitment bound to inputs."""
    manifest = strict_json(Path(manifest_path).read_bytes())
    if not isinstance(freeze, dict):
        raise ValueError("An instruction freeze record is required")
    digest = freeze.get("instructions_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Instruction freeze requires a SHA256 commitment")
    for name in ("inputs.jsonl", "public_schemas.json"):
        if freeze.get(name + "_sha256") != manifest["files"][name]["sha256"]:
            raise ValueError("Instruction freeze is not bound to these inputs/schemas")
    _, data = _prepared_file(manifest_path, "sealed_labels.jsonl")
    rows = [strict_json(line) for line in data.splitlines() if line]
    _, schemas, inputs = load_inputs(manifest_path)
    if [row["id"] for row in rows] != [row["id"] for row in inputs]:
        raise ValueError("Dialogue label identities do not match inputs")
    from .metrics import validate_reference
    for row, input_row in zip(rows, inputs):
        if (row["dialogue_id"] != input_row["dialogue_id"]
                or row["service"] != input_row["service"]):
            raise ValueError("Dialogue label metadata differs from input")
        validate_reference(row["state"], schemas[row["service"]])
    return {row["id"]: row for row in rows}
