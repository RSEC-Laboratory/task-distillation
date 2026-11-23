"""Run fresh local student inference against the archived, frozen preparation.

No teacher key, Python model packages, or model download is used. An Ollama
server must already have the recorded checkpoint. Every selected logical case
receives a new request; archived responses and request caches are never reused.
This reruns the student, not teacher preparation or independent-task validation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import ipaddress
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib import error, parse, request


MODELS = {
    "llama": ("llama3.2:3b", "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72"),
    "gemma": ("gemma2:9b", "ff02c3702f322b9e075e9568332d96c0a7028002f1a5a056e0a6784320a4db0b"),
}
ARMS = ("raw_original", "raw_labels", "initial_prose", "initial_list",
        "final_prose", "final_list", "selected_full", "selected_routed")
BASE = Path("reports/reviewer1_20261120/original_ablation")


class RunError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise RunError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def append(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        # Preserve a charged request even if the process stops during transport.
        import os
        os.fsync(handle.fileno())


def read_json(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "An input path escapes the artifact root")
    require(path.is_file(), f"Missing archived input: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalise_student_label(text: str, labels: set[str]) -> str | None:
    """Extract one declared label from a model response."""
    candidate = text.strip().lower().strip("`'\".,:;()[]{} ")
    abstentions = {"uncertain", "unknown", "abstain", "none"}
    if candidate in abstentions:
        return None
    if candidate in labels:
        return candidate
    first = candidate.split()[0].strip("`'\".,:;()[]{} ") if candidate else ""
    if first in abstentions:
        return None
    if first in labels:
        return first

    final_choices = sorted(labels | abstentions, key=len, reverse=True)
    final_choice_pattern = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(item) for item in final_choices) + r")(?![\w-])",
        re.IGNORECASE,
    )
    mention_choices = sorted(labels | {"uncertain"}, key=len, reverse=True)
    mention_pattern = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(item) for item in mention_choices) + r")(?![\w-])",
        re.IGNORECASE,
    )
    final_marker = re.compile(
        r"\b(?:final|answer|therefore|thus|hence|conclusion|"
        r"classification|declared\s+label|(?:the\s+)?output)\b|"
        r"^\s*(?:label|prediction)\s*:",
        re.IGNORECASE,
    )
    for line in reversed(text.splitlines()):
        if not final_marker.search(line):
            continue
        matches = {match.casefold() for match in final_choice_pattern.findall(line)}
        if len(matches) == 1:
            selected = matches.pop()
            return None if selected in abstentions else selected
        if len(matches) > 1:
            return f"__INVALID__:{candidate[:80]}"

    matches = {match.casefold() for match in mention_pattern.findall(text)}
    if len(matches) == 1:
        selected = matches.pop()
        return None if selected in abstentions else selected
    return f"__INVALID__:{candidate[:80]}"


def check_scorer(root):
    """Check this dependency-free scorer against the supplied implementation."""
    source = root / "src/taskcompiler/translate.py"
    require(source.is_file(), "Missing supplied scoring source: src/taskcompiler/translate.py")
    original = next((node for node in ast.parse(source.read_text(encoding="utf-8")).body
                     if isinstance(node, ast.FunctionDef) and node.name == "normalise_student_label"), None)
    copied = ast.parse(inspect.getsource(normalise_student_label)).body[0]
    require(original is not None and ast.dump(original) == ast.dump(copied),
            "The local scorer differs from the supplied original implementation")
    return sha(source)


def select_cases(root, cohort, arms, limit):
    prepared = read_json(root, BASE / "prepared_inputs.json")
    tasks = {(t["cohort"], t["task_id"]): t for t in prepared}
    path = root / BASE / "predictions.jsonl"
    require(path.is_file(), "Missing archived primary prediction index")
    selected = []
    available = 0
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["cohort"] != cohort or row["arm"] not in arms or row["sentinel"]:
                continue
            key = (row["task_id"], row["case_index"], row["arm"])
            require(key not in seen, "Duplicate archived primary condition")
            seen.add(key)
            available += 1
            if limit is not None and len(selected) >= limit:
                continue
            task = tasks[(cohort, row["task_id"])]
            case = task["cases"][row["case_index"]]
            require(case["index"] == row["case_index"] and case["gold"] == row["gold"],
                    "Archived case identity or expected label is inconsistent")
            selected.append({"task_id": row["task_id"], "case_index": row["case_index"],
                "arm": row["arm"], "gold": case["gold"], "labels": task["labels"],
                "source_call_id": row["call_id"], "source_request_hash": row["request_hash"],
                "expected_prompt": f"Input: {case['record']}"})
    require(bool(selected), "No archived cases match the selected cohort and arms")
    calls = root / BASE / "calls.jsonl"
    require(calls.is_file(), "Missing archived started-request ledger")
    wanted = {row["source_call_id"] for row in selected}
    starts = {}
    with calls.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["event"] == "started" and row["call_id"] in wanted:
                require(row["call_id"] not in starts, "Duplicate archived started request")
                starts[row["call_id"]] = row
    for row in selected:
        require(row["source_call_id"] in starts, "A selected case lacks its original started request")
        archived = starts[row["source_call_id"]]
        identity = archived["model_identity"]
        require(archived["prompt"] == row.pop("expected_prompt") and
                archived["request_hash"] == row["source_request_hash"] and not archived["sentinel"],
                "Archived prompt or request binding differs from its primary case")
        require(identity["model"] == MODELS[cohort][0] and identity["digest"] == MODELS[cohort][1]
                and identity["backend"] == "ollama" and not identity.get("fallback_used"),
                "Archived model identity differs from the selected checkpoint")
        require(identity["temperature"] == 0 and identity["seed"] == 0
                and identity["context_tokens"] == 8192 and identity["thinking"] is False
                and archived["max_answer_tokens"] == 16,
                "Archived decoding settings differ from this registered comparison")
        messages = ([{"role": "system", "content": archived["system"]}]
                    if archived["system"] else [])
        messages.append({"role": "user", "content": archived["prompt"]})
        row["request"] = {"model": MODELS[cohort][0], "messages": messages,
            "stream": False, "think": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": 16, "num_ctx": 8192}}
    return selected, available, {"prepared_inputs": sha(root / BASE / "prepared_inputs.json"),
        "primary_index": sha(path), "started_request_ledger": sha(calls)}


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LocalServer:
    def __init__(self, server, timeout):
        parsed = parse.urlsplit(server)
        require(parsed.scheme in {"http", "https"} and parsed.hostname and
                not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
                "--server must be an HTTP(S) address without credentials, query or fragment")
        if parsed.hostname != "localhost":
            try:
                address = ipaddress.ip_address(parsed.hostname)
            except ValueError as exc:
                raise RunError("Use localhost or a literal private/loopback IP for the local model server") from exc
            require((address.is_loopback or address.is_private) and not address.is_unspecified,
                    "Public model endpoints are not supported by this local rerun")
        self.url = server.rstrip("/")
        self.timeout = timeout
        self.opener = request.build_opener(request.ProxyHandler({}), NoRedirect())

    def json(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(self.url + path, data=body,
                              headers={"Content-Type": "application/json"} if body else {})
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (error.URLError, OSError, ValueError) as exc:
            raise RunError(f"Local server request failed ({path}): {exc}") from exc
        require(isinstance(result, dict), "Local server returned a non-object JSON response")
        return result

    def identity(self, model, expected_digest):
        version = self.json("/api/version")
        require(isinstance(version.get("version"), str), "Ollama version response is missing")
        tags = self.json("/api/tags")
        matches = [row for row in tags.get("models", [])
                   if row.get("name", row.get("model")) == model]
        require(len(matches) == 1, f"The local server must already contain exactly one {model} tag")
        digest = str(matches[0].get("digest", "")).removeprefix("sha256:")
        require(digest == expected_digest, "Local checkpoint digest does not match the retained student")
        return {"server": self.url, "ollama_version": version["version"],
                "model": model, "digest": digest}


def run(options):
    root = options.root.resolve()
    output = options.output.resolve()
    require(not output.exists(), "--output already exists; choose a new directory")
    require(output != root and not any(output.is_relative_to(root / name)
        for name in ("src", "runs", "reports", "reviewer_artifact", "frozen", "data", "protocols")),
        "--output must not be an archived source or evidence directory")
    scorer_hash = check_scorer(root)
    cohort = options.cohort or next((name for name, value in MODELS.items()
                                  if value[0] == options.model), "llama")
    model, expected_digest = MODELS[cohort]
    require(options.model is None or options.model == model, "--model and --cohort refer to different students")
    arms = list(ARMS) if "all" in (options.arm or []) else list(dict.fromkeys(options.arm or ["selected_routed"]))
    limit = None if options.all else options.limit or 5
    selected, available, hashes = select_cases(root, cohort, arms, limit)
    server = LocalServer(options.server, options.timeout)
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at_utc": now(), "scope": "Fresh local student inference using frozen historical preparation; not fresh teacher generation or independent confirmation.",
        "cohort": cohort, "model": model, "required_digest": expected_digest,
        "arms": arms, "available_primary_rows": available, "planned_requests": len(selected),
        "coverage": "all_selected_primary_rows" if options.all else "first_selected_rows_in_archived_order",
        "decoding": {"temperature": 0, "seed": 0, "num_predict": 16, "num_ctx": 8192, "think": False},
        "request_policy": "One fresh call per selected logical row; no cached response, automatic retry, download, teacher call, or remote fallback.",
        "archived_input_sha256": hashes, "scoring_source_sha256": scorer_hash,
        "cases": [{k: v for k, v in row.items() if k != "request"} for row in selected]}
    write_json(output / "plan.json", plan)
    if options.plan_only:
        print(json.dumps({"status": "planned", "planned_requests": len(selected), "network_requests": 0}, indent=2))
        return 0
    started = time.monotonic()
    predictions = []
    status = "complete"
    failure = None
    try:
        before = server.identity(model, expected_digest)
        write_json(output / "server_before.json", before)
        for index, row in enumerate(selected, 1):
            binding = {key: row[key] for key in ("task_id", "case_index", "arm", "source_call_id", "source_request_hash")}
            append(output / "requests.jsonl", {"event": "started", "call_id": index,
                "at_utc": now(), **binding, "request": row["request"]})
            call_started = time.monotonic()
            try:
                response = server.json("/api/chat", row["request"])
                require(response.get("model") == model and response.get("done") is True,
                        "The local response is incomplete or names a different model")
                message = response.get("message")
                require(isinstance(message, dict) and isinstance(message.get("content"), str),
                        "The local response has no text content")
                raw = message["content"].strip()
                predicted = normalise_student_label(raw, set(row["labels"]))
                record = {"event": "completed", "call_id": index, "at_utc": now(),
                    **binding, "raw_output": raw, "predicted": predicted,
                    "gold": row["gold"], "correct": predicted == row["gold"],
                    "invalid": isinstance(predicted, str) and predicted.startswith("__INVALID__"),
                    "input_tokens": response.get("prompt_eval_count"),
                    "output_tokens": response.get("eval_count"),
                    "latency_seconds": time.monotonic() - call_started}
            except (RunError, OSError, ValueError) as exc:
                record = {"event": "failed", "call_id": index, "at_utc": now(),
                    **binding, "gold": row["gold"], "predicted": "__INVALID__:request_failure",
                    "correct": False, "invalid": True, "error": str(exc),
                    "latency_seconds": time.monotonic() - call_started}
                append(output / "requests.jsonl", record)
                append(output / "predictions.jsonl", record)
                predictions.append(record)
                raise
            append(output / "requests.jsonl", record)
            append(output / "predictions.jsonl", record)
            predictions.append(record)
            print(f"{index}/{len(selected)} {row['task_id']} {row['arm']} correct={record['correct']}", flush=True)
        write_json(output / "server_after.json", server.identity(model, expected_digest))
    except (RunError, OSError, ValueError) as exc:
        status, failure = "incomplete", str(exc)
    counts = Counter(row["arm"] for row in predictions)
    conditions = {}
    for arm, total in counts.items():
        group = [row for row in predictions if row["arm"] == arm]
        correct = sum(row["correct"] for row in group)
        conditions[arm] = {"correct": correct, "attempted": total,
            "accuracy_on_attempted_rows": correct / total,
            "invalid": sum(row["invalid"] for row in group),
            "failed_requests": sum(row["event"] == "failed" for row in group)}
    result = {"status": status, "completed_at_utc": now(), "cohort": cohort,
        "model": model, "planned_requests": len(selected), "attempted_requests": len(predictions),
        "elapsed_seconds": time.monotonic() - started, "conditions": conditions,
        "error": failure, "teacher_calls": 0, "archived_responses_reused": 0,
        "scope": plan["scope"], "coverage": plan["coverage"],
        "limits": "Results may differ across Ollama/runtime versions. A limited run is not the full benchmark. Failed attempts remain incorrect; unattempted rows are not silently scored."}
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0 if status == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Example: python -S rerun_student.py --root task-compiler --cohort llama --arm raw_labels --limit 5 --output fresh-llama")
    default = Path(__file__).resolve().parent
    if (default / "task-compiler").is_dir():
        default = default / "task-compiler"
    parser.add_argument("--root", type=Path, default=default, help="Materialized project/evidence root")
    parser.add_argument("--server", default="http://127.0.0.1:11434", help="Existing loopback/private-IP Ollama server")
    parser.add_argument("--cohort", choices=tuple(MODELS), help="Default: infer from --model, otherwise llama")
    parser.add_argument("--model", choices=[value[0] for value in MODELS.values()], help="Optional explicit student tag")
    parser.add_argument("--arm", choices=(*ARMS, "all"), action="append", help="Repeat for multiple arms; default selected_routed")
    extent = parser.add_mutually_exclusive_group()
    extent.add_argument("--limit", type=int, help="Maximum fresh requests; default 5")
    extent.add_argument("--all", action="store_true", help="Run all primary rows for the selected cohort and arms")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside archived source/evidence folders")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds; no retries")
    parser.add_argument("--plan-only", action="store_true", help="Write selection/settings without contacting any server")
    options = parser.parse_args(argv)
    if options.limit is not None and options.limit <= 0:
        parser.error("--limit must be positive")
    if not math.isfinite(options.timeout) or options.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    try:
        return run(options)
    except (RunError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
