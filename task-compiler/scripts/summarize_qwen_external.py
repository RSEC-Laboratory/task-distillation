"""Audit completed Qwen external runs without dispatching model calls.

Run only after all four runs finish. Incomplete or ambiguous panels fail before
any prediction file is read. Saved predictions never select a run or condition.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taskcompiler.statistics import exact_mcnemar, task_bootstrap_difference, wilson_accuracy

METHODS = {
    "nonparent_verified_scratchpad": "counting_adapter",
    "textgrad_tgd": "counting_textgrad",
    "textgrad_published_prompt": "counting_transfer",
    "superni_compiler_matrix": "superni",
}
DISPLAY = {
    "counting_adapter": "Counting adapter",
    "counting_textgrad": "Local TextGrad optimization",
    "counting_transfer": "Published TextGrad prompt transfer",
}
CONDITIONS = ("raw_definition", "positive_examples", "parent_one_shot", "validated_full", "validated_routed")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    require(path.is_file(), f"Missing durable evidence: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(request: dict) -> tuple[str, str]:
    return (request.get("system") or "", request.get("prompt") or "")


def config_for(path: Path) -> dict:
    require(path.is_dir(), f"Run directory missing: {path}")
    value = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Invalid saved configuration: {path}")
    require(value.get("student_model") == "qwen2.5:3b", f"Not a Qwen2.5 3B run: {path}")
    require("20261006" in str(value.get("run_label", "")), f"Not the declared 20261006 extension: {path}")
    require(value.get("baseline") in METHODS, f"Unsupported external method: {path}")
    return value


def completed_run(path: Path, config: dict) -> dict:
    # This gate precedes prediction/history/ledger reads, including for explicit
    # paths. A finalized failure summary does not constitute a completed run.
    log = (path / "log.txt").read_text(encoding="utf-8")
    lines = [line for line in log.splitlines() if line.strip()]
    require(bool(lines) and lines[-1].endswith("run finished"),
            f"Run is incomplete or failed; predictions remain unread: {path}")
    summary = read_json(path / "summary.json")
    metrics = summary.get("task_metrics", {})
    require(metrics.get("baseline") == config["baseline"], f"Summary/config method mismatch: {path}")
    require(bool(metrics), f"Missing completed metrics: {path}")
    return summary


def choose_runs(args: argparse.Namespace) -> dict[str, tuple[Path, dict, dict]]:
    explicit = {name: getattr(args, name) for name in ("counting_adapter", "counting_textgrad", "counting_transfer", "superni")
                if getattr(args, name) is not None}
    require(not (args.runs and explicit), "Use --runs or named run arguments, not both")
    candidates: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    if args.runs:
        for path in args.runs:
            path = path.resolve()
            config = config_for(path)
            candidates[METHODS[config["baseline"]]].append((path, config))
    elif explicit:
        for method, path in explicit.items():
            path = path.resolve()
            config = config_for(path)
            require(METHODS[config["baseline"]] == method, f"Named argument has wrong method: {path}")
            candidates[method].append((path, config))
    else:
        # Discovery examines labels/configuration only. It never chooses the
        # latest or highest-scoring repeated run.
        for path in sorted(args.runs_root.resolve().iterdir()):
            if not path.is_dir() or "qwen" not in path.name.casefold() or "20261006" not in path.name:
                continue
            config = config_for(path)
            candidates[METHODS[config["baseline"]]].append((path, config))
    require(set(candidates) == set(METHODS.values()),
            "Provide one completed run for each of counting adapter, TextGrad optimization, prompt transfer, and SuperNI")
    for method, rows in candidates.items():
        require(len(rows) == 1, f"Ambiguous {method} runs; specify the intended path explicitly")
    # Gate the entire panel before inspecting any final answers.
    result = {}
    for method in METHODS.values():
        path, config = candidates[method][0]
        result[method] = (path, config, completed_run(path, config))
    return result


def ledger_costs(events: list[dict], classifier) -> tuple[dict, list[dict]]:
    starts = {}
    terminals = {}
    for event in events:
        kind = event.get("event")
        require(kind in {"started", "completed", "failed"}, f"Unknown ledger event: {kind}")
        call_id = event["call_id"]
        if kind == "started":
            require(call_id not in starts and call_id not in terminals, f"Duplicate/reordered request {call_id}")
            starts[call_id] = event
        else:
            require(call_id in starts and call_id not in terminals, f"Unmatched/duplicate terminal event {call_id}")
            require(starts[call_id]["role"] == event["role"], f"Role changed for request {call_id}")
            require(datetime.fromisoformat(event["at_utc"]) >= datetime.fromisoformat(starts[call_id]["at_utc"]), f"Terminal predates request {call_id}")
            terminals[call_id] = event
    groups: dict[str, dict] = {}
    calls = []
    for call_id, start in starts.items():
        role = start["role"]
        require(role in {"student", "teacher"}, f"Unknown request role: {role}")
        phase = classifier(start)
        require(phase in {"preparation", "final"}, f"Unknown phase for request {call_id}")
        key = f"{phase}_{role}"
        group = groups.setdefault(key, {
            "attempted_calls": 0, "completed_calls": 0, "failed_calls": 0,
            "unterminated_calls": 0, "known_input_tokens": 0, "known_output_tokens": 0,
            "known_model_latency_seconds": 0.0, "known_request_elapsed_seconds": 0.0,
            "unknown_input_usage_calls": 0, "unknown_output_usage_calls": 0,
            "unknown_model_latency_calls": 0, "unknown_request_elapsed_calls": 0,
        })
        group["attempted_calls"] += 1
        terminal = terminals.get(call_id)
        state = terminal["event"] if terminal else "unterminated"
        group[{"completed": "completed_calls", "failed": "failed_calls", "unterminated": "unterminated_calls"}[state]] += 1
        response = terminal.get("response", {}) if terminal and state == "completed" else {}
        for field, known, unknown in (
            ("input_tokens", "known_input_tokens", "unknown_input_usage_calls"),
            ("output_tokens", "known_output_tokens", "unknown_output_usage_calls"),
            ("latency_seconds", "known_model_latency_seconds", "unknown_model_latency_calls"),
        ):
            value = response.get(field)
            if value is None:
                group[unknown] += 1
            else:
                require(isinstance(value, (int, float)) and value >= 0, f"Invalid {field} for request {call_id}")
                group[known] += value
        elapsed = terminal.get("elapsed_seconds") if terminal else None
        if elapsed is None:
            group["unknown_request_elapsed_calls"] += 1
        else:
            require(elapsed >= 0, f"Invalid elapsed time for request {call_id}")
            group["known_request_elapsed_seconds"] += elapsed
        calls.append({"start": start, "terminal": terminal, "phase": phase, "state": state})
    return {
        "groups": groups,
        "attempted_calls": len(starts),
        "completed_calls": sum(x["completed_calls"] for x in groups.values()),
        "failed_calls": sum(x["failed_calls"] for x in groups.values()),
        "unterminated_calls": sum(x["unterminated_calls"] for x in groups.values()),
        "scope": "Physical inference attempts from durable started events; retries are disabled by the audited runner. Unknown usage remains unknown. Model latency excludes routing/local compilation and differs from request elapsed and run wall time.",
        "observed_teacher_output_token_limits": sorted({x["start"]["request"]["max_tokens"] for x in calls
                                                        if x["start"]["role"] == "teacher" and x["start"]["request"].get("max_tokens") is not None}),
    }, calls


def verify_teacher_routing(identity: dict, config: dict) -> dict:
    """Verify the endpoint while preserving mutable resolved-provider metadata.

    Frozen config and client source establish the requested routing policy.
    TeacherClient overwrites identity.provider with the response provider, so
    neither later started events nor completed events repeat the request policy.
    Provider display-name aliases cannot be audited as routing violations here.
    """
    require(identity.get("base_url") == str(config.get("openrouter_base_url", "")).rstrip("/"),
            "Teacher endpoint differs from the saved configuration")
    requested = config.get("teacher_openrouter_provider")
    require(requested is None or isinstance(requested, str), "Invalid requested teacher provider policy")
    requested = requested.strip() if requested else None
    reported = identity.get("provider")
    require(reported is None or isinstance(reported, str), "Invalid reported teacher provider metadata")
    return {
        "requested_provider": requested,
        "requested_routing_policy": "provider order fixed; allow_fallbacks false" if requested else "unrestricted provider selection",
        "reported_provider": reported,
        "reported_label_matches_request": (reported.strip().casefold() == requested.casefold())
        if requested and reported else None,
        "scope": "Hash-checked config and frozen client source establish the request policy. Response provider labels remain descriptive; missing labels or noncanonical aliases do not establish a routing violation.",
    }


def verify_frozen_commitments(protocol: dict, config: dict, *, summary: dict | None = None,
                              events: list[dict] | None = None) -> None:
    """Check all frozen data/config files and every recorded model identity."""
    for mapping in (protocol.get("source_sha256", {}), protocol.get("counting_data_sha256", {}),
                    protocol.get("config_sha256", {}), protocol.get("superni", {}).get("source_sha256", {})):
        for relative, expected in mapping.items():
            require(sha(ROOT / relative) == expected, f"Frozen source/config/data commitment changed: {relative}")
    matching_configs = []
    for relative in protocol.get("config_sha256", {}):
        candidate = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        if candidate.get("run_label") == config["run_label"]:
            matching_configs.append(candidate)
    require(len(matching_configs) == 1, "Saved run lacks a unique frozen configuration")
    require(all(config.get(key) == value for key, value in matching_configs[0].items()),
            "Saved run settings differ from the frozen configuration")
    for entry in protocol.get("superni", {}).get("data_and_saved_artifacts", []):
        data_path = ROOT / entry["data_path"]
        require(sha(data_path) == entry["data_sha256"], f"Frozen SuperNI data changed: {entry['task_id']}")
        data = read_json(data_path)
        first20 = [{"item_id": str(x.get("id") or f"{entry['task_id']}-instance-{i:04d}"),
                    "input": x["input"], "output": x["output"]}
                   for i, x in enumerate(data["Instances"][:20])]
        digest = hashlib.sha256(json.dumps(first20, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        require(digest == entry["first20_content_sha256"], f"Frozen SuperNI first20 instances changed: {entry['task_id']}")
        require(sha(ROOT / entry["saved_artifact_path"]) == entry["saved_artifact_file_sha256"],
                f"Frozen historical artifact provenance changed: {entry['task_id']}")
    student = protocol["student"]
    teacher = protocol["teacher"]
    for field, expected in (("student_model", student["model"]), ("student_backend", student["backend"]),
                            ("student_seed", student["seed"]), ("student_context_tokens", student["context_tokens"]),
                            ("student_thinking", student["thinking"]), ("ollama_base_url", student["base_url"]),
                            ("teacher_model", teacher["model"]), ("teacher_backend", teacher["backend"]),
                            ("teacher_selection", teacher["selection"])):
        require(config.get(field) == expected, f"Saved setting differs from protocol: {field}")
    if config["dataset"] == "bbh":
        require(config["max_answer_tokens"] == protocol["counting"]["answer_tokens"],
                "Counting answer-token setting differs from protocol")
        teacher_cap = (protocol["counting"].get("adapter_teacher_output_token_cap")
                       if config["baseline"] == "nonparent_verified_scratchpad"
                       else protocol["counting"].get("textgrad_teacher_output_token_cap",
                                                     protocol["counting"].get("teacher_tokens")))
    else:
        fixed = protocol["superni"]["fixed_settings"]
        require(config["max_answer_tokens"] == fixed["max_answer_tokens"],
                "SuperNI answer-token setting differs from protocol")
        teacher_cap = fixed["teacher_output_token_cap"]
    def identity_matches(role: str, identity: dict) -> None:
        if role == "student":
            for field in ("model", "backend", "base_url", "digest", "temperature", "seed", "context_tokens", "thinking"):
                require(identity.get(field) == student[field], f"Student identity differs from protocol: {field}")
            require(identity.get("requested_backend") == student["backend"] and identity.get("fallback_used") is False,
                    "Student silently changes backend or uses a fallback")
        elif role == "teacher":
            require(identity.get("backend") == teacher["backend"] and identity.get("requested_model") == teacher["model"]
                    and identity.get("temperature") == 0, "Teacher requested model/backend/temperature differs from protocol")
            verify_teacher_routing(identity, config)
            require(identity.get("max_retries") == 0, "Teacher inference retries are not disabled")
    if summary is not None:
        models = summary.get("environment", {}).get("models", {})
        require("student" in models, "Completed run lacks a student identity")
        for role in ("student", "teacher"):
            if role in models:
                identity_matches(role, models[role])
    for event in events or []:
        if event["event"] in {"started", "completed"}:
            require(isinstance(event.get("model_identity"), dict), "Durable model event lacks an identity")
            identity_matches(event["role"], event["model_identity"])
        if event["event"] == "started" and event["role"] == "teacher":
            require(teacher_cap is not None and event["request"].get("max_tokens") == teacher_cap,
                    "Teacher request output-token cap differs from protocol")


def provenance(path: Path, config: dict, summary: dict) -> dict:
    protocol_path = path / "artifacts/frozen_protocol.json"
    require(protocol_path.is_file(), f"No retained audited protocol: {path}")
    protocol = read_json(protocol_path)
    # Frozen implementation is the scoring authority. Do not silently audit
    # with current helper functions if their source changed.
    events = read_jsonl(path / "artifacts/model_calls.jsonl")
    verify_frozen_commitments(protocol, config, summary=summary, events=events)
    teacher_identity = summary.get("environment", {}).get("models", {}).get("teacher")
    routing = verify_teacher_routing(teacher_identity, config) if teacher_identity else {
        "requested_provider": config.get("teacher_openrouter_provider"),
        "requested_routing_policy": "No fresh teacher inference in this run",
    }
    routing["completed_response_provider_counts"] = dict(Counter(
        str(event["model_identity"].get("provider") or "unreported")
        for event in events if event["event"] == "completed" and event["role"] == "teacher"))
    routing["ledger_interpretation"] = "Completed teacher-event provider labels describe that response. Started-event labels may still describe the preceding response; they are not a separately recorded requested routing policy."
    return {
        "run_path": str(path), "run_label": config["run_label"],
        "baseline": config["baseline"], "student_model": config["student_model"],
        "teacher_model": config.get("teacher_model"), "teacher_selection": config.get("teacher_selection"),
        "recorded_settings": {key: config.get(key) for key in (
            "student_backend", "student_context_tokens", "max_answer_tokens", "student_thinking",
            "seed", "student_seed", "token_budget", "max_guidance_retries",
            "max_teacher_repair_rounds", "teacher_structured_outputs", "top_k_rules",
            "retriever_model", "retriever_revision")},
        "recorded_models": summary.get("environment", {}).get("models", {}),
        "teacher_routing": routing,
        "run_wall_seconds": summary.get("total_wall_seconds"),
        "frozen_protocol_sha256": sha(protocol_path), "config_sha256": sha(path / "config.yaml"),
        "predictions_sha256": sha(path / "artifacts/predictions.jsonl"),
        "model_calls_sha256": sha(path / "artifacts/model_calls.jsonl"),
        "source_snapshot_sha256": summary.get("environment", {}).get("source", {}).get("snapshot_sha256"),
        "frozen_data_config_and_all_model_identities_verified": True,
    }


def counting_request_assignments(events: list[dict], rows: list[dict], *,
                                 verifier, parser, corrections_allowed: bool) -> dict:
    """Associate exact correction requests with their preceding final answer.

    The frozen correction request omits the original question. Its exact prompt,
    system message, computed trace value and immediate predecessor establish the
    association instead. A malformed or orphan correction fails this audit.
    """
    correction_system = "Return the locally verified answer exactly as requested. Do not add reasoning."
    correction_prefix = "A local verifier checked your structured trace "
    by_id = {row["item_id"]: row for row in rows}
    require(len(by_id) == len(rows), "Duplicate Counting prediction item IDs")
    questions = {row["question"]: row["item_id"] for row in rows}
    require(len(questions) == len(rows), "Duplicate final questions prevent request association")
    assignments = {}
    initial_items = set()
    corrected_items = set()
    previous_event = None
    last_started_id = None
    for event in events:
        if event["event"] == "started":
            call_id = event["call_id"]
            require(call_id not in assignments, f"Duplicate Counting request {call_id}")
            request = event["request"]
            prompt = request.get("prompt") or ""
            system = request.get("system") or ""
            matches = [item_id for question, item_id in questions.items() if question in prompt]
            require(len(matches) <= 1, "Ambiguous final-question matching")
            is_correction = system == correction_system or prompt.startswith(correction_prefix)
            if is_correction:
                require(corrections_allowed and event["role"] == "student" and not matches,
                        "Correction request occurs outside the frozen Counting adapter")
                require(last_started_id is not None and previous_event is not None,
                        "Correction lacks a preceding final answer")
                preceding = assignments[last_started_id]
                require(preceding["phase"] == "final" and preceding["kind"] == "initial"
                        and previous_event["event"] == "completed"
                        and previous_event["call_id"] == last_started_id
                        and previous_event["role"] == "student",
                        "Correction must immediately follow its completed initial final answer")
                item_id = preceding["item_id"]
                row = by_id[item_id]
                require(bool(row.get("local_correction_used")) and item_id not in corrected_items,
                        f"Unexpected or duplicate correction for {item_id}")
                initial_text = previous_event["response"]["text"]
                require(initial_text == row["initial_output"], f"Correction predecessor text differs: {item_id}")
                checked = verifier("object_counting", initial_text)
                require(checked is not None and checked == row.get("verified_ledger")
                        and checked != parser("object_counting", initial_text),
                        f"Correction does not match the frozen local-verifier decision: {item_id}")
                expected_prompt = ("A local verifier checked your structured trace "
                                   f"and obtained {checked}. Return exactly "
                                   f"`Answer: {checked}`.")
                require(prompt == expected_prompt and system == correction_system
                        and request.get("response_schema") is None,
                        f"Correction prompt/system/schema differs from the frozen method: {item_id}")
                assignments[call_id] = {"phase": "final", "kind": "correction", "item_id": item_id}
                corrected_items.add(item_id)
            elif event["role"] == "student" and matches:
                item_id = matches[0]
                require(item_id not in initial_items, f"Repeated initial final request: {item_id}")
                assignments[call_id] = {"phase": "final", "kind": "initial", "item_id": item_id}
                initial_items.add(item_id)
            else:
                assignments[call_id] = {"phase": "preparation", "kind": "preparation", "item_id": None}
            last_started_id = call_id
        previous_event = event
    require(initial_items == set(by_id), "Counting ledger lacks a unique initial final request for every case")
    require(corrected_items == {row["item_id"] for row in rows if row.get("local_correction_used")},
            "Counting ledger/prediction correction sets differ")
    return assignments


def counting_correction_effects(rows: list[dict], *, parser) -> dict:
    """Describe initial answers and correction outcomes without new selection."""
    outcomes = {"incorrect_to_correct": 0, "correct_to_incorrect": 0,
                "incorrect_to_incorrect": 0, "correct_to_correct": 0}
    initial_by_id = {}
    corrected_items = []
    for row in rows:
        initial = parser("object_counting", row["initial_output"]) == row["gold"]
        final = bool(row["correct"])
        initial_by_id[row["item_id"]] = initial
        if row.get("local_correction_used"):
            outcome = ("correct" if initial else "incorrect") + "_to_" + ("correct" if final else "incorrect")
            outcomes[outcome] += 1
            corrected_items.append({"item_id": row["item_id"], "outcome": outcome})
        else:
            require(initial == final, f"Uncorrected Counting answer changes score: {row['item_id']}")
    initial_correct = sum(initial_by_id.values())
    final_correct = sum(bool(row["correct"]) for row in rows)
    require(len(initial_by_id) == len(rows), "Duplicate item IDs in correction diagnostic")
    require(final_correct - initial_correct == outcomes["incorrect_to_correct"] - outcomes["correct_to_incorrect"],
            "Counting correction outcomes do not reconcile the initial/final gain")
    return {
        "initial_correct": initial_correct, "total": len(rows),
        "initial_accuracy": initial_correct / len(rows),
        "initial_correctness_by_id": initial_by_id,
        "corrected_cases": len(corrected_items), "outcomes": outcomes,
        "net_correct_answer_change": final_correct - initial_correct,
        "corrected_case_outcomes": corrected_items,
        "scope": "Initial accuracy and correction transitions are descriptive pipeline diagnostics, not additional selected primary conditions. The frozen local checker verifies arithmetic of the item/quantity ledger supplied by the student; it does not independently verify category membership or whether the selected items faithfully represent the input.",
    }


def audit_counting(runs: dict) -> dict:
    from taskcompiler.guidance_track.bbh import parse_answer, _verified_scratchpad_answer

    audited = {}
    reference_ids = None
    reference_cases = None
    reference_manifest = None
    for method in ("counting_adapter", "counting_textgrad", "counting_transfer"):
        path, config, summary = runs[method]
        prov = provenance(path, config, summary)
        rows = read_jsonl(path / "artifacts/predictions.jsonl")
        metrics = summary["task_metrics"]
        require(config.get("benchmark_task") == "object_counting" and config.get("benchmark_split") == "test", f"Wrong Counting panel: {path}")
        ids = [x["item_id"] for x in rows]
        require(len(rows) == len(set(ids)) == 100, f"Counting requires100 distinct cases: {path}")
        manifest = read_json(path / "artifacts/split_manifest.json")
        require(ids == manifest["splits"]["test"], f"Counting final IDs do not match locked100-case split: {path}")
        cases = {x["item_id"]: (x["question"], str(x["reference"])) for x in rows}
        data_path = ROOT / "data/bbh_official/bbh/object_counting.json"
        require(sha(data_path) == manifest["data_sha256"], "Official Counting data commitment changed")
        official = read_json(data_path)["examples"][150:250]
        official_cases = {
            f"object_counting-{index:03d}-{hashlib.sha256(str(x['input']).encode('utf-8')).hexdigest()[:10]}":
            (str(x["input"]), str(x["target"]))
            for index, x in enumerate(official, start=150)
        }
        require(cases == official_cases, f"Counting rows differ from the committed official positional test split: {path}")
        if reference_ids is None:
            reference_ids, reference_cases, reference_manifest = ids, cases, manifest["manifest_sha256"]
        require(ids == reference_ids and cases == reference_cases, f"Counting methods have different cases/reference answers: {path}")
        require(manifest["manifest_sha256"] == reference_manifest, f"Counting split manifest differs: {path}")
        for row in rows:
            gold = str(int(row["reference"]))
            parsed = parse_answer("object_counting", row["raw_output"])
            require(row["gold"] == gold and row["parsed_output"] == parsed, f"Counting parser/reference mismatch: {row['item_id']}")
            require(row["correct"] == (parsed == gold), f"Counting score mismatch: {row['item_id']}")
        correct = sum(x["correct"] for x in rows)
        correction_effects = counting_correction_effects(rows, parser=parse_answer)
        require(metrics["correct"] == correct and metrics["num_examples"] == 100, f"Counting summary differs: {path}")
        events = read_jsonl(path / "artifacts/model_calls.jsonl")
        assignments = counting_request_assignments(
            events, rows, verifier=_verified_scratchpad_answer, parser=parse_answer,
            corrections_allowed=(method == "counting_adapter"))
        costs, calls = ledger_costs(events, lambda start: assignments[start["call_id"]]["phase"])
        costs["classification"] = "Initial final calls match exact final question text; correction calls require the immediately preceding completed initial answer, the computed verified-ledger value, and the exact frozen correction prompt/system."
        final_calls = [x for x in calls if x["phase"] == "final" and x["start"]["role"] == "student"]
        expected_calls = 100 + sum(bool(x.get("local_correction_used")) for x in rows)
        require(len(final_calls) == expected_calls, f"Counting final/correction request count differs: {path}")
        require(all(x["state"] == "completed" for x in final_calls), f"Unobserved final answers cannot enter complete Counting panel: {path}")
        by_id = defaultdict(list)
        for call in final_calls:
            item_id = assignments[call["start"]["call_id"]]["item_id"]
            by_id[item_id].append(call["terminal"]["response"])
        for row in rows:
            responses = by_id[row["item_id"]]
            require(len(responses) == 1 + bool(row.get("local_correction_used")), f"Wrong final attempts for {row['item_id']}")
            require(responses[0]["text"] == row["initial_output"] and responses[-1]["text"] == row["raw_output"], f"Counting ledger/prediction text mismatch: {row['item_id']}")
            for token in ("input_tokens", "output_tokens"):
                require(sum(x[token] for x in responses) == row[token], f"Counting final token mismatch: {row['item_id']}")
        audited[method] = {
            "correct": correct, "total": 100, "accuracy": correct / 100,
            "wilson_95_descriptive": asdict(wilson_accuracy(correct, 100)),
            "parse_failures": sum(x["parsed_output"] is None for x in rows),
            "local_corrections": sum(bool(x.get("local_correction_used")) for x in rows),
            "initial_answers_and_correction_effects": correction_effects,
            "final_student_attempts": len(final_calls),
            "final_finish_reasons": {
                kind: dict(Counter(str(x["terminal"]["response"].get("finish_reason") or "unrecorded")
                                   for x in final_calls if assignments[x["start"]["call_id"]]["kind"] == kind))
                for kind in ("initial", "correction")},
            "correctness_by_id": {x["item_id"]: x["correct"] for x in rows},
            "costs": costs, "provenance": prov,
            "preparation_information": "Public task text and fresh structured teacher artifact; local scratchpad checks/corrections" if method == "counting_adapter" else "50 labeled training and100 labeled validation cases;12 TextGrad updates" if method == "counting_textgrad" else "Published optimized prompt transferred to Qwen; no fresh local optimization",
            "optimizer_protocol": metrics.get("optimizer_protocol"),
            "scoring_qualification": "Final scoring uses the shared BBH parser for all three methods. TextGrad optimization/validation retains the official last-number equality function, which maps a parse failure to zero; this differs from the final parser and matches the retained Llama protocol. The analysis changes neither parser nor optimizer decisions.",
        }
    comparisons = []
    a = audited["counting_adapter"]["correctness_by_id"]
    for method in ("counting_textgrad", "counting_transfer"):
        b = audited[method]["correctness_by_id"]
        va, vb = [a[k] for k in reference_ids], [b[k] for k in reference_ids]
        comparisons.append({
            "method": "counting_adapter", "comparator": method,
            "mean_accuracy_difference": sum(x - y for x, y in zip(va, vb)) / 100,
            "gains": sum(x and not y for x, y in zip(va, vb)),
            "losses": sum(y and not x for x, y in zip(va, vb)),
            "ties": sum(x == y for x, y in zip(va, vb)),
            "exact_two_sided_mcnemar_p": exact_mcnemar(va, vb),
            "multiple_comparison_adjustment": "None; both paired tests are exploratory and unadjusted",
            "scope": "Paired descriptive comparison on the same100 previously examined BBH test cases; different preparation information and actual costs.",
        })
    return {"same100_case_ids_and_references": True, "manifest_sha256": reference_manifest,
            "case_ids": reference_ids, "methods": audited, "paired_comparisons": comparisons}


def audit_interrupted_runs(args: argparse.Namespace, selected: dict) -> list[dict]:
    """Charge terminated predecessor launches without importing their answers."""
    if args.interrupted_runs:
        paths = [p.resolve() for p in args.interrupted_runs]
    else:
        paths = [p.resolve() for p in args.runs_root.resolve().iterdir()
                 if p.is_dir() and "qwen" in p.name.casefold() and "20261006" in p.name
                 and (p / "artifacts/termination.json").is_file()]
    require(len(paths) == len(set(paths)), "Duplicate interrupted-run path would double-charge overhead")
    selected_paths = {record[0] for record in selected.values()}
    data = read_json(ROOT / "data/bbh_official/bbh/object_counting.json")["examples"]
    public_questions = {str(x["input"]) for x in data[:150]}
    final_questions = {str(x["input"]) for x in data[150:250]}
    result = []
    for path in sorted(paths):
        require(path not in selected_paths, "Completed algorithm run also listed as interrupted overhead")
        config = config_for(path)
        require(config["dataset"] == "bbh" and config.get("benchmark_task") == "object_counting", "Interrupted-cost reader currently supports Counting predecessors only")
        termination_path = path / "artifacts/termination.json"
        termination = read_json(termination_path)
        require(termination.get("status") == "externally_terminated", f"No explicit termination boundary: {path}")
        events = read_jsonl(path / "artifacts/model_calls.jsonl")
        verify_frozen_commitments(read_json(path / "artifacts/frozen_protocol.json"), config, events=events)
        def classify(start):
            if start["role"] == "teacher":
                return "preparation"
            prompt = start["request"].get("prompt") or ""
            require(not any(q in prompt for q in final_questions), f"Terminated predecessor accessed final cases; separate final-cost handling required: {path}")
            require(prompt in public_questions, f"Cannot establish public preparation phase for terminated request: {path}")
            return "preparation"
        costs, _ = ledger_costs(events, classify)
        require(not (path / "artifacts/predictions.jsonl").exists(), f"Unexpected final predictions in explicitly pre-final terminated predecessor: {path}")
        declared = termination.get("cost", {})
        for key, actual in (("started", costs["attempted_calls"]), ("completed", costs["completed_calls"]),
                            ("teacher_attempts", costs["groups"].get("preparation_teacher", {}).get("attempted_calls", 0))):
            if key in declared:
                require(declared[key] == actual, f"Termination cost declaration differs for {key}: {path}")
        result.append({"run_path": str(path), "run_label": config["run_label"],
                       "status": termination["status"], "termination_reason": termination.get("reason"),
                       "termination_sha256": sha(termination_path),
                       "model_calls_sha256": sha(path / "artifacts/model_calls.jsonl"),
                       "costs": costs,
                       "scope": "Additional project effort outside the complete algorithm run. All started attempts remain charged; unanswered requests have unknown completion, tokens and model duration. No prediction or optimizer history is read."})
    return result


def aggregate_costs(records: list[dict]) -> dict:
    groups = {}
    for record in records:
        for phase, values in record["groups"].items():
            group = groups.setdefault(phase, {key: 0 for key in values})
            for key, value in values.items():
                group[key] += value
    return {"groups": groups,
            "attempted_calls": sum(r["attempted_calls"] for r in records),
            "completed_calls": sum(r["completed_calls"] for r in records),
            "failed_calls": sum(r["failed_calls"] for r in records),
            "unterminated_calls": sum(r["unterminated_calls"] for r in records)}


def audit_superni(run: tuple[Path, dict, dict]) -> dict:
    from taskcompiler.guidance_track import superni as sn

    path, config, summary = run
    prov = provenance(path, config, summary)
    metrics = summary["task_metrics"]
    rows = read_jsonl(path / "artifacts/predictions.jsonl")
    require(config.get("benchmark_split") == "development" and config.get("num_tasks") == 4 and config.get("num_examples") == 20, "SuperNI requires the declared4-task/20-instance development panel")
    require(tuple(metrics["conditions"]) == CONDITIONS, "SuperNI condition set/order changed")
    require(len(rows) == 400, "SuperNI requires80 rows for each of five conditions")
    manifest = read_json(path / "artifacts/suite_manifest.json")
    require([t["task_id"] for t in manifest["tasks"]] == list(sn.DEVELOPMENT_TASKS), "SuperNI development task set/order differs")
    require(metrics["manifest_sha256"] == manifest["manifest_sha256"], "SuperNI manifest/summary mismatch")
    by_key = {}
    for row in rows:
        key = (row["task_id"], row["item_id"], row["condition"])
        require(key not in by_key, f"Duplicate SuperNI row: {key}")
        by_key[key] = row
    final_signatures = set()
    expected_responses = Counter()
    task_metrics = []
    fallbacks = []
    cached_rows = Counter()
    candidate_metadata = []
    for entry in manifest["tasks"]:
        task = sn.load_task(entry["task_id"])
        require(task.source_sha256 == entry["source_sha256"], f"SuperNI task data changed: {task.task_id}")
        artifact_path = path / "artifacts" / f"{task.task_id}.json"
        artifact = read_json(artifact_path)
        require(artifact["task_id"] == task.task_id and artifact["source_sha256"] == task.source_sha256,
                f"SuperNI artifact/source identity differs: {task.task_id}")
        require(artifact.get("parent_received_instances") is False and artifact.get("student_validation_used_instances") is False,
                f"SuperNI artifact records official-instance exposure during preparation: {task.task_id}")
        artifact_copy = dict(artifact)
        internal = artifact_copy.pop("artifact_sha256")
        require(hashlib.sha256(json.dumps(artifact_copy, sort_keys=True).encode("utf-8")).hexdigest() == internal, f"SuperNI artifact hash mismatch: {task.task_id}")
        initial = sn.parse_representation(json.dumps(artifact["initial_representation"])) if artifact["initial_representation"] is not None else None
        final = sn.parse_representation(json.dumps(artifact["final_representation"])) if artifact["final_representation"] is not None else None
        fallback = artifact.get("failure") is not None or final is None
        if not fallback:
            validation = artifact.get("final_validation", [])
            require(len(validation) == len(task.positive_examples) and all(x["correct"] for x in validation),
                    f"SuperNI deployable artifact lacks a complete public certificate: {task.task_id}")
        if fallback:
            fallbacks.append(task.task_id)
        scores = {condition: {"correct": 0, "total": 20} for condition in CONDITIONS}
        for example in task.instances[:20]:
            cache = {}
            for condition in CONDITIONS:
                key = (task.task_id, example.item_id, condition)
                require(key in by_key, f"Missing official SuperNI case/condition: {key}")
                row = by_key[key]
                require(row["references"] == list(example.outputs), f"SuperNI references differ: {key}")
                try:
                    correct = sn.response_matches(row["response"], example.outputs)
                except IndexError:
                    correct = False
                require(row["correct"] == correct, f"SuperNI exact-match score differs: {key}")
                public_examples = task.positive_examples if condition != "raw_definition" else ()
                instructions = None
                if condition == "parent_one_shot" and initial is not None:
                    instructions = sn.render_representation(initial)
                elif condition == "validated_full" and not fallback:
                    instructions = sn.render_representation(final)
                elif condition == "validated_routed" and not fallback:
                    routing = row.get("routing") or {}
                    indices = routing.get("selected_optional_indices")
                    optional = sn._optional_lines(final)
                    require(isinstance(indices, list) and indices == sorted(set(indices)) and all(isinstance(i, int) and 0 <= i < len(optional) for i in indices), f"Invalid routed instruction indices: {key}")
                    require(routing.get("optional_count") == len(optional) and routing.get("top_k") == config["top_k_rules"], f"Routing configuration differs: {key}")
                    instructions = sn._mandatory_lines(final) + [optional[i] for i in indices]
                prompt = sn._student_prompt(task.definition, example.input_text, instructions, public_examples)
                sig = (sn.STUDENT_SYSTEM_PROMPT, prompt)
                reused = sig in cache
                require(row.get("response_reused_for_identical_prompt") is reused, f"SuperNI fallback/cache flag differs: {key}")
                if reused:
                    previous = cache[sig]
                    require(all(row[field] == previous[field] for field in ("response", "input_tokens", "output_tokens")), f"SuperNI identical prompt receives inconsistent cached response: {key}")
                    cached_rows[condition] += 1
                else:
                    cache[sig] = row
                    final_signatures.add(sig)
                    expected_responses[(sig, row["response"])] += 1
                if condition in {"validated_full", "validated_routed"} and fallback:
                    require((row.get("routing") or {}).get("fallback") == "positive_examples", f"SuperNI failed compilation does not retain positive-example fallback: {key}")
                scores[condition]["correct"] += int(correct)
        for score in scores.values():
            score["accuracy"] = score["correct"] / 20
        task_metrics.append({"task_id": task.task_id, "conditions": scores})
        candidate_metadata.append({"task_id": task.task_id, "artifact_sha256": sha(artifact_path),
                                   "failure": artifact.get("failure"), "teacher_calls": artifact["teacher_calls"]})
    require(task_metrics == metrics["task_metrics"], "SuperNI independently recomputed task metrics differ")
    condition_results = {}
    for condition in CONDITIONS:
        correct = sum(t["conditions"][condition]["correct"] for t in task_metrics)
        accuracy = correct / 80
        require(abs(metrics["macro_task_accuracy"][condition] - accuracy) < 1e-12, f"SuperNI macro differs: {condition}")
        condition_results[condition] = {"correct": correct, "total": 80, "macro_task_accuracy": accuracy,
                                        "response_reused_rows": cached_rows[condition]}
    bootstrap = {}
    scores_a = {t["task_id"]: t["conditions"]["validated_routed"]["accuracy"] for t in task_metrics}
    for comparator in ("raw_definition", "positive_examples"):
        scores_b = {t["task_id"]: t["conditions"][comparator]["accuracy"] for t in task_metrics}
        value = asdict(task_bootstrap_difference(scores_a, scores_b, samples=10000, seed=20260727))
        recorded = metrics["paired_task_bootstrap"][comparator]
        require(all(abs(value[k] - recorded[k]) < 1e-12 for k in value), f"SuperNI task-bootstrap mismatch: {comparator}")
        bootstrap[comparator] = value
    def classify(start):
        return "final" if start["role"] == "student" and signature(start["request"]) in final_signatures else "preparation"
    costs, calls = ledger_costs(read_jsonl(path / "artifacts/model_calls.jsonl"), classify)
    final_calls = [x for x in calls if x["phase"] == "final" and x["start"]["role"] == "student"]
    require(all(x["state"] == "completed" for x in final_calls), "SuperNI complete comparison contains unobserved final responses")
    observed = Counter((signature(x["start"]["request"]), x["terminal"]["response"]["text"]) for x in final_calls)
    require(observed == expected_responses, "SuperNI final ledger does not match distinct uncached prompts and saved responses")
    first_final = min(datetime.fromisoformat(x["start"]["at_utc"]) for x in final_calls)
    preparation_calls = [x for x in calls if x["phase"] == "preparation"]
    require(all(datetime.fromisoformat(x["start"]["at_utc"]) < first_final for x in preparation_calls), "SuperNI preparation occurs after final scoring starts")
    artifact_times = [datetime.fromtimestamp((path / "artifacts" / f"{t}.json").stat().st_mtime, timezone.utc) for t in sn.DEVELOPMENT_TASKS]
    require(max(artifact_times) <= first_final, "SuperNI artifact file modification occurs after final scoring starts")
    require(costs["groups"].get("preparation_teacher", {}).get("attempted_calls", 0) == metrics["teacher_calls"], "SuperNI teacher accounting differs")
    return {
        "same80_official_case_ids_and_references_per_condition": True,
        "all_four_artifacts_fixed_before_final_scoring": True,
        "case_ids": [{"task_id": t, "item_id": i} for t, i, condition in by_key if condition == "positive_examples"],
        "conditions": condition_results, "primary_comparison": "validated_routed minus positive_examples",
        "descriptive_four_task_bootstrap_95": bootstrap, "bootstrap_resamples": 10000, "bootstrap_seed": 20260727,
        "task_metrics": task_metrics, "fallback_tasks": fallbacks, "artifacts": candidate_metadata,
        "costs": costs, "provenance": prov,
        "scope": "Fresh Qwen-specific compilation, public-positive-example validation/repair and routed deployment with a manually fixed Llama70B teacher. Other conditions and their shared costs remain retained. This extends a previously examined four-task development panel, not the unopened confirmation panel.",
    }


def readable(summary: dict) -> str:
    lines = ["# Qwen2.5 3B external development extension", "",
             "This exploratory extension uses a frozen local Qwen2.5 3B student and a manually fixed Llama 70B teacher. It uses previously examined tasks and supplies no fresh confirmation or general-transfer claim. Counting methods receive different preparation information and spend different amounts of compute. The Counting adapter uses a fresh teacher scaffold; the transferred published prompt receives no fresh local optimization.", "",
             "## BBH Object Counting", "", "All three methods score the same 100 locked test cases. Wilson intervals describe example-level accuracy within this single task.", "",
             "| Method | Initial correct / total | Final correct / total | Final accuracy | Final student calls | Descriptive Wilson 95% interval |", "|---|---:|---:|---:|---:|---:|"]
    for method, row in summary["counting"]["methods"].items():
        interval = row["wilson_95_descriptive"]
        initial = row["initial_answers_and_correction_effects"]
        lines.append(f"| {DISPLAY[method]} | {initial['initial_correct']}/100 | {row['correct']}/100 | {100*row['accuracy']:.2f}% | {row['final_student_attempts']} | [{100*interval['low']:.2f}, {100*interval['high']:.2f}]% |")
    adapter = summary["counting"]["methods"]["counting_adapter"]
    effects = adapter["initial_answers_and_correction_effects"]
    outcome = effects["outcomes"]
    lines += ["", f"The adapter initially answers {effects['initial_correct']}/100 cases correctly. Its {effects['corrected_cases']} correction calls change {outcome['incorrect_to_correct']} incorrect answers to correct and {outcome['correct_to_incorrect']} correct answers to incorrect; {outcome['incorrect_to_incorrect']} remain incorrect and {outcome['correct_to_correct']} remain correct. The net change is {effects['net_correct_answer_change']:+d} correct answers. Its final {adapter['correct']}/100 result therefore measures instruction delivery, student trace generation and local arithmetic correction together. The checker verifies the arithmetic of the ledger supplied by the student; category membership and faithful item selection remain student decisions. The adapter uses {adapter['final_student_attempts']} final student calls, including corrections; TextGrad optimization and published-prompt transfer each use 100 final calls."]
    lines += ["", "| Adapter comparison | Difference (pp) | Gains / losses | Exact two-sided McNemar p |", "|---|---:|---:|---:|"]
    for row in summary["counting"]["paired_comparisons"]:
        lines.append(f"| {DISPLAY[row['comparator']]} | {100*row['mean_accuracy_difference']:+.2f} | {row['gains']} / {row['losses']} | {row['exact_two_sided_mcnemar_p']:.6g} |")
    lines += ["", "The adapter uses public task text and a newly generated structured artifact with local scratchpad checks. Local TextGrad uses 50 labeled training cases and 100 labeled validation cases over 12 updates. Prompt transfer reuses the published optimized instruction. These are paired final-case comparisons with unequal preparation inputs and costs. Both McNemar tests are exploratory and unadjusted for multiple comparisons.", "",
              "All final predictions use the shared BBH parser. TextGrad optimization and validation retain its official last-number equality function, which maps a parse failure to zero; this differs from the final parser and follows the retained Llama protocol. The analysis changes neither parser nor optimizer decisions.", "",
              "## Super-NaturalInstructions", "", "Each condition scores the same 80 cases: 20 instances for each of four source-distinct development tasks. All four artifacts are fixed before final scoring. Invalid outputs and compilation fallbacks remain in the denominators.", "",
              "| Condition | Correct / total | Macro task accuracy | Cached response rows |", "|---|---:|---:|---:|"]
    superni = summary["superni"]
    for condition, row in superni["conditions"].items():
        lines.append(f"| {condition.replace('_', ' ')} | {row['correct']}/80 | {100*row['macro_task_accuracy']:.2f}% | {row['response_reused_rows']} |")
    for comparator, row in superni["descriptive_four_task_bootstrap_95"].items():
        lines += ["", f"Routed adapter minus {comparator.replace('_', ' ')}: {100*row['estimate']:+.2f} pp, descriptive whole-task 95% interval [{100*row['low']:.2f}, {100*row['high']:.2f}] pp (10,000 resamples over four tasks)."]
    lines += ["", "Compilation fallback tasks: " + (", ".join(superni["fallback_tasks"]) or "none") + ". Byte-identical prompts reuse the comparator response; cached rows add no model attempt or token cost.", "",
              "## Actual call accounting", "", "Counts charge every durable started request, including failed or unterminated requests. Known usage sums remain lower bounds when usage is missing. Preparation and final calls are classified from final questions, exact correction prompts tied to their preceding verified answers, or reconstructed final prompt signatures. Summed model latency excludes routing and local computation; it is not total compute or run wall time.", "",
              "| Run | Phase / role | Attempts | Completed | Failed | Unfinished | Known input / output tokens | Known model latency (s) | Unknown usage / latency calls |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    records = [(DISPLAY[k], v) for k, v in summary["counting"]["methods"].items()] + [("SuperNI, all five conditions", superni)]
    records += [("Interrupted launch overhead", v) for v in summary["interrupted_launches"]]
    for label, record in records:
        for phase, row in record["costs"]["groups"].items():
            unknown = f"{row['unknown_input_usage_calls']}/{row['unknown_output_usage_calls']}/{row['unknown_model_latency_calls']}"
            lines.append(f"| {label} | {phase.replace('_', ' / ')} | {row['attempted_calls']} | {row['completed_calls']} | {row['failed_calls']} | {row['unterminated_calls']} | {row['known_input_tokens']:,} / {row['known_output_tokens']:,} | {row['known_model_latency_seconds']:.2f} | {unknown} |")
    project = summary["project_effort"]
    complete = project["completed_algorithm_run_costs"]
    interrupted = project["interrupted_launch_overhead_costs"]
    overall = project["all_actual_project_attempts_and_known_usage"]
    lines += ["", f"Completed algorithm runs use {complete['attempted_calls']:,} attempts. Interrupted-launch overhead adds {interrupted['attempted_calls']:,} attempts, giving {overall['attempted_calls']:,} actual project attempts. Overhead is retained separately and also included in the structured project totals; unknown attempts are never assigned zero actual cost."]
    for row in summary["interrupted_launches"]:
        lines += ["", f"Interrupted predecessor `{Path(row['run_path']).name}`: {row['termination_reason']}"]
    lines += ["", "SuperNI final costs cover all five retained conditions, including shared/cached responses; they are not a cost estimate for only the two primary columns. Cross-student comparisons retain the manually fixed teacher model and disclose current runtime identities. Historical Llama SuperNI fallback rows require its published identical-prompt correction; no earlier result is overwritten by this report.", "",
              "The structured companion summary retains every task, primary comparison, other condition, model identity, run path, evidence hash and unknown-cost count. This analysis dispatches no model calls and selects no favorable run or condition.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, help="Explicit paths to all four completed runs, in any order")
    parser.add_argument("--counting-adapter", type=Path)
    parser.add_argument("--counting-textgrad", type=Path)
    parser.add_argument("--counting-transfer", type=Path)
    parser.add_argument("--superni", type=Path)
    parser.add_argument("--interrupted-runs", nargs="+", type=Path,
                        help="Terminated predecessor launches to charge separately; defaults to all declared Qwen20261006 termination artifacts")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/qwen_external_20261006")
    args = parser.parse_args()
    runs = choose_runs(args)
    summary = {
        "analysis_id": "qwen25-3b-external-development-20261006-v1",
        "status": "completed_panel_audited",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": sha(Path(__file__)),
        "new_model_calls_for_analysis": 0,
        "counting": audit_counting(runs), "superni": audit_superni(runs["superni"]),
        "claim_limits": ["Exploratory extension on previously examined panels, not fresh confirmation.",
                         "Same manually fixed teacher model is not a verified Qwen distillation-lineage claim or a guarantee of identical remote provider execution across dates.",
                         "Counting information, preparation calls and actual compute differ.",
                         "Four SuperNI tasks, not80 independent tasks; whole-task intervals are descriptive.",
                         "Fresh scaffolding and Qwen validation/repair are distinguished from published-prompt transfer and historic artifact reuse."],
    }
    summary["interrupted_launches"] = audit_interrupted_runs(args, runs)
    complete_costs = [x["costs"] for x in summary["counting"]["methods"].values()] + [summary["superni"]["costs"]]
    overhead_costs = [x["costs"] for x in summary["interrupted_launches"]]
    summary["project_effort"] = {
        "completed_algorithm_run_costs": aggregate_costs(complete_costs),
        "interrupted_launch_overhead_costs": aggregate_costs(overhead_costs),
        "all_actual_project_attempts_and_known_usage": aggregate_costs(complete_costs + overhead_costs),
        "scope": "Completed algorithm costs and independently charged interrupted launches are separate views of the same project ledger totals. Missing completion/usage remains unknown. Calendar interruption gaps and unmeasured local/routing computation are not inferred from summed model latency.",
    }
    output = args.output_dir.resolve()
    require(output == (ROOT / "reports/qwen_external_20261006").resolve(), "Write analysis only under reports/qwen_external_20261006")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "RESULTS.md").write_text(readable(summary), encoding="utf-8")
    print(f"Audited completed panel; wrote {output / 'summary.json'} and {output / 'RESULTS.md'}. No model calls.")


if __name__ == "__main__":
    main()
