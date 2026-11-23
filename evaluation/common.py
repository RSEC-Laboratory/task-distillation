"""Record loading, unchanged scoring functions, and physical-call accounting."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


# Clean record locations are shared by the table and control evaluators.
RECORDS = Path("evaluation/records")
SUBMITTED_RECORDS = RECORDS / "submitted"
INSTRUCTION_RECORDS = RECORDS / "instruction_ablation"
COUNTING_RECORDS = RECORDS / "counting_control"
LOOKUP_RECORDS = RECORDS / "public_lookup"
GEPA_RECORDS = RECORDS / "matched_gepa"
RESERVED_PROTOCOL = RECORDS / "reserved_nlp/protocol.json"


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accuracy(correct, total):
    return {
        "correct": int(correct),
        "total": int(total),
        "accuracy": correct / total if total else None,
    }


def scoring_functions(root):
    """Load only unchanged, pure scoring functions, avoiding heavy module imports."""
    namespace = {"re": re}
    selections = {
        "src/taskcompiler/translate.py": {"normalise_student_label"},
        "src/taskcompiler/guidance_track/bbh.py": {
            "parse_answer",
            "_verified_ledger_sum",
        },
        "src/taskcompiler/guidance_track/superni.py": {
            "_normalise_label",
            "response_matches",
        },
    }
    hashes = {}
    for relative, names in selections.items():
        path = root / relative
        source = path.read_text(encoding="utf-8")
        functions = [
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        require(
            {node.name for node in functions} == names,
            "Required scoring functions missing: " + relative,
        )
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        hashes[relative] = digest(path)
    return namespace, hashes


def call_ledger(path):
    events = rows(path)
    starts = {r["call_id"]: r for r in events if r["event"] == "started"}
    ends = {r["call_id"]: r for r in events if r["event"] in {"completed", "failed"}}
    require(
        len(starts) == sum(r["event"] == "started" for r in events),
        "Duplicate call start",
    )
    require(
        len(ends) == sum(r["event"] in {"completed", "failed"} for r in events),
        "Duplicate terminal call",
    )
    require(starts.keys() == ends.keys(), "Unresolved physical calls")
    return starts, ends


def attempt_accounting(path, allow_unfinished=False):
    events = rows(path)
    starts = {r["call_id"]: r for r in events if r["event"] == "started"}
    ends = {r["call_id"]: r for r in events if r["event"] in {"completed", "failed"}}
    require(
        len(starts) == sum(r["event"] == "started" for r in events),
        "Duplicate budget-ledger start",
    )
    require(
        len(ends) == sum(r["event"] in {"completed", "failed"} for r in events),
        "Duplicate budget-ledger terminal",
    )
    require(ends.keys() <= starts.keys(), "Terminal event has no dispatch")
    unfinished = len(starts.keys() - ends.keys())
    require(
        allow_unfinished or unfinished == 0, "Unresolved attempt in a completed panel"
    )
    return {
        "attempted": len(starts),
        "completed": sum(r["event"] == "completed" for r in ends.values()),
        "failed": sum(r["event"] == "failed" for r in ends.values()),
        "unfinished_outcome_unknown": unfinished,
        "attempts_by_role": dict(Counter(r["role"] for r in starts.values())),
        "known_input_tokens": sum(
            r.get("response", {}).get("input_tokens", 0)
            for r in ends.values()
            if r["event"] == "completed"
        ),
        "known_output_tokens": sum(
            r.get("response", {}).get("output_tokens", 0)
            for r in ends.values()
            if r["event"] == "completed"
        ),
        "scope": "Known usage only. Interrupted outstanding attempts are retained, not assigned zero usage.",
    }
