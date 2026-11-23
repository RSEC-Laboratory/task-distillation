"""Audit and summarize a completed frozen matched original-pipeline study."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[2]
# ISO timestamps retain microseconds; raw ledger Unix floats can retain finer
# fractions. Allow only that serialization resolution when comparing the two.
ISO_RESOLUTION_SECONDS = 0.000001


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def audit_calls(path: Path, stored: dict, protocol: dict, *, final: bool, opening: dict) -> list[dict]:
    """Recompute attempt costs and verify charged requests against transport."""
    from matched_original import EventLedger
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if [row["event_index"] for row in events] != list(range(len(events))):
        raise ValueError("Missing or reordered call-ledger events")
    if any(before["recorded_at_unix"] > after["recorded_at_unix"] for before, after in zip(events, events[1:])):
        raise ValueError("Call-ledger timestamps are out of order")
    ledger = EventLedger()
    ledger.events = events
    if ledger.summary() != stored:
        raise ValueError("Call summary differs from retained attempts and usage")
    requests = [row for row in events if row["event"] == "model_request"]
    identifiers = [row["attempt_id"] for row in requests]
    if len(set(identifiers)) != len(identifiers) or stored["unfinished_calls"]:
        raise ValueError("Duplicate or unfinished model attempt")
    for request in requests:
        attempt = [row for row in events if row.get("attempt_id") == request["attempt_id"]]
        terminal = [row for row in attempt if row["event"] in {"model_response", "model_failure"}]
        transports = [row for row in attempt if row["event"] == "transport_request"]
        payloads = [row for row in attempt if row["event"] == "transport_response"]
        if len(terminal) != 1 or len(transports) != 1 or len(payloads) > 1:
            raise ValueError("Attempt must have one dispatched transport and one terminal outcome")
        if request["event_index"] >= transports[0]["event_index"] or terminal[0]["event_index"] <= transports[0]["event_index"]:
            raise ValueError("Request was not recorded before transport dispatch")
        body = transports[0]["body"]
        identity = request["model_identity"]
        if body["model"] != identity["model"] or body["messages"] != [
            {"role": "system", "content": request["system"]}, {"role": "user", "content": request["prompt"]}]:
            raise ValueError("Transport differs from the charged model request")
        is_teacher = request["role"] == "teacher_optimization"
        expected_cap = protocol["teacher_max_output_tokens"] if is_teacher else protocol["student_max_output_tokens"]
        if (body["options"]["num_predict"] != expected_cap or body["options"]["num_ctx"] != protocol["context_tokens"]
                or body["options"]["temperature"] != protocol["temperature"] or body["think"] is not False):
            raise ValueError("Dispatched model options differ from shared protocol")
        if body["options"]["seed"] != identity["seed"]:
            raise ValueError("Charged and actual model seed differ")
        if body.get("format", {}) != request["output_schema"]:
            raise ValueError("Actual decoding differs from charged output schema")
        if not is_teacher and (request["output_schema"] or body["options"]["seed"] != protocol["seed"]):
            raise ValueError("Solver decoding or seed differs across conditions")
        if terminal[0]["event"] == "model_response":
            if len(payloads) != 1:
                raise ValueError("Completed attempt lacks original transport response")
            raw = payloads[0]["payload"]
            if not raw.get("done") or raw.get("error") or raw.get("done_reason") == "length":
                raise ValueError("Incomplete, erroneous or truncated transport cannot be a successful response")
            if (raw["message"]["content"] != terminal[0]["text"]
                or raw.get("prompt_eval_count") != terminal[0]["input_tokens"]
                or raw.get("eval_count") != terminal[0]["output_tokens"]):
                raise ValueError("Reported output or token use differs from original response")
        if final and request["recorded_at_unix"] + ISO_RESOLUTION_SECONDS < timestamp(opening["opened_at"]).timestamp():
            raise ValueError("Final model request predates global instruction lock")
    for role, costs in stored["roles"].items():
        ceiling = (protocol["final_cases_per_task"] if final else
                   protocol["optimization_teacher_call_ceiling_per_task_arm"] if role == "teacher_optimization" else
                   protocol["optimization_solver_call_ceiling_per_task_arm"])
        if costs["attempted_calls"] > ceiling:
            raise ValueError("Attempted model-call ceiling exceeded")
    return events


def interval(differences: list[float], seed: int, resamples: int = 10000) -> list[float]:
    generator = random.Random(seed)
    draws = sorted(statistics.mean(generator.choices(differences, k=len(differences)))
                   for _ in range(resamples))
    return [draws[int(.025 * resamples)], draws[int(.975 * resamples)]]


def summarize(output: Path) -> dict:
    from matched_original import (PublicTask, PreparedMethod, parse_label,
                                  solver_messages, make_router, content_hash, MODEL_FAILURE_TEXT)
    from taskcompiler.dsl.parser import parse_program
    from taskcompiler.dsl.interpreter import run as execute_dsl
    protocol = read(output / "protocol.json")
    completion = read(output / "completion.json")
    if completion["status"] != "complete" or (output / "interruption.json").exists():
        raise ValueError("Incomplete study cannot be summarized as complete")
    source = read(output / "source_snapshot.json")
    for name, expected in source["source_hashes"].items():
        if sha(output / "source" / name) != expected:
            raise ValueError(f"Copied source mismatch: {name}")
    package = read(output / "native_package_snapshot.json")
    if package["version"] != protocol["gepa"]["version"]:
        raise ValueError("Native package version mismatch")
    for name, expected_hash in package["files_sha256"].items():
        if sha(output / "dependencies" / name) != expected_hash:
            raise ValueError(f"Native package source mismatch: {name}")
    if sha(output / "protocol.json") != source["source_hashes"]["experiments/protocols/matched_original/development_20261005_v1.json"]:
        raise ValueError("Executed protocol differs from the committed source snapshot")
    seal = read(output / "case_commitment.json")
    if sha(output / "public_tasks.json") != seal["public_tasks_sha256"]:
        raise ValueError("Public data commitment mismatch")
    if sha(output / "sealed/final_cases.json") != seal["final_cases_sha256"]:
        raise ValueError("Final data commitment mismatch")
    lock = read(output / "prepared_lock.json")
    opening = read(output / "final_opening.json")
    expected = len(protocol["task_ids"]) * len(protocol["solver_models"]) * 2
    if lock["expected_artifacts"] != expected or len(lock["artifacts"]) != expected:
        raise ValueError("Incomplete global prepared-artifact lock")
    lock_hash = sha(output / "prepared_lock.json")
    if opening["prepared_lock_sha256"] != lock_hash or completion["prepared_lock_sha256"] != lock_hash:
        raise ValueError("Instruction-lock commitment mismatch")
    if not (timestamp(seal["committed_at"]) <= timestamp(lock["locked_at"]) <= timestamp(opening["opened_at"])):
        raise ValueError("Preparation/final opening chronology mismatch")
    if not timestamp(read(output / "resource_scope.json")["declared_at"]) <= timestamp(seal["committed_at"]):
        raise ValueError("Resource declaration was not prospective")
    public = {row["task_id"]: row for row in read(output / "public_tasks.json")}
    if set(public) != {f"policy-{number:03d}" for number in protocol["task_ids"]}:
        raise ValueError("Public task inventory differs from fixed protocol")
    tasks = {key: PublicTask.from_dict(value) for key, value in public.items()}
    if any(len(task.build_cases) != 27 or len(task.audit_cases) != 27 for task in tasks.values()):
        raise ValueError("Public 27/27 case split differs from protocol")
    finals = read(output / "sealed/final_cases.json")
    if set(finals) != set(public):
        raise ValueError("Final task inventory differs from public task inventory")
    identities = read(output / "model_identities.json")
    router_identity = read(output / "router_identity.json")
    snapshot = Path(router_identity["identity"]["local_snapshot"])
    for name, expected_hash in router_identity["files_sha256"].items():
        if sha(snapshot / name) != expected_hash:
            raise ValueError("Pinned routing model changed before audit replay")
    router = make_router(snapshot)
    prepared, preparation_costs = {}, []
    evidence = {name: sha(output / name) for name in (
        "protocol.json", "source_snapshot.json", "public_tasks.json", "sealed/final_cases.json",
        "case_commitment.json", "prepared_lock.json", "final_opening.json", "completion.json",
        "model_identities.json", "backend_identity.json", "router_identity.json", "native_package_snapshot.json", "resource_scope.json")}
    for name, expected_hash in lock["artifacts"].items():
        path = output / name
        if path.resolve().parent != (output / "prepared").resolve() or sha(path) != expected_hash:
            raise ValueError(f"Prepared artifact mismatch: {name}")
        row = read(path)
        key = (row["solver_model"], row["task_id"], row["arm"])
        if key in prepared or timestamp(row["prepared_at"]) > timestamp(lock["locked_at"]):
            raise ValueError("Duplicate or late prepared artifact")
        prepared[key] = row
        evidence[name] = expected_hash
        preparation_costs.append({"solver_model": key[0], "task_id": key[1], "arm": key[2],
            "wall_seconds": row["preparation_wall_seconds"], "calls": row["call_summary"]})
        events = audit_calls(output / "calls" / f"{key[0].replace(':', '_')}-{key[1]}-{key[2]}.jsonl",
                             row["call_summary"], protocol, final=False, opening=opening)
        if any(event["recorded_at_unix"] > timestamp(row["prepared_at"]).timestamp() + ISO_RESOLUTION_SECONDS for event in events):
            raise ValueError("Optimization events continued after prepared artifact was stored")
        for event in events:
            if event["event"] == "model_request":
                model_name = protocol["teacher_model"] if event["role"] == "teacher_optimization" else key[0]
                if event["model_identity"]["digest"] != identities[model_name]["digest"]:
                    raise ValueError("Optimization model differs from frozen model identity")
        if key[2] == "gepa":
            trace = row["method"]["trace"]
            if trace["metric_calls"] > protocol["optimization_solver_call_ceiling_per_task_arm"]:
                raise ValueError("Native GEPA logical metric ceiling exceeded")
            logical_calls = sum(len(event["case_ids"]) for event in trace["native_history"]
                                if event["event"] == "evaluation_requested")
            if trace["metric_calls"] != logical_calls:
                raise ValueError("Native GEPA metric accounting differs from its retained evaluation history")
            if trace["solver_attempts"] != row["call_summary"]["roles"].get("solver_optimization", {}).get("attempted_calls", 0):
                raise ValueError("Native GEPA attempted model-call trace differs from call ledger")
        elif "behavioral" in row["method"]["trace"]:
            if row["method"]["trace"]["behavioral"]["student_calls"] != row["call_summary"]["roles"].get("solver_optimization", {}).get("attempted_calls", 0):
                raise ValueError("Full behavioral call accounting differs from its call ledger")
    expected_keys = {(model, f"policy-{number:03d}", arm) for model in protocol["solver_models"]
                     for number in protocol["task_ids"] for arm in ("full", "gepa")}
    if set(prepared) != expected_keys or len(list((output / "prepared").glob("*.json"))) != expected:
        raise ValueError("Unexpected prepared artifact inventory")
    all_task_rows, final_costs = [], []
    for model in protocol["solver_models"]:
        for number in protocol["task_ids"]:
            task_id = f"policy-{number:03d}"
            public_records = {json.dumps(row["input_record"], sort_keys=True)
                              for split in ("build_cases", "audit_cases") for row in public[task_id][split]}
            references = {row["case_id"]: row for row in finals[task_id]}
            if len(references) != protocol["final_cases_per_task"]:
                raise ValueError("Unexpected final denominator")
            if any(json.dumps(row["input_record"], sort_keys=True) in public_records for row in references.values()):
                raise ValueError("Final/public input overlap")
            scores, failures, deploy_status = {}, {}, {}
            full_method = PreparedMethod.from_dict(prepared[(model, task_id, "full")]["method"])
            full_source = full_method.to_dict()["dsl_source"]
            learned_program = parse_program(full_source, task_spec=tasks[task_id].spec) if full_source else None
            for arm in ("raw", "full", "gepa", "direct_dsl"):
                if arm in {"full", "gepa"} and (model, task_id, arm) not in prepared:
                    raise ValueError("Missing frozen algorithm preparation")
                path = output / "final" / f"{model.replace(':', '_')}-{task_id}-{arm}.json"
                payload = read(path)
                if (payload["solver_model"], payload["task_id"], payload["arm"]) != (model, task_id, arm):
                    raise ValueError("Final cell metadata mismatch")
                if timestamp(payload["completed_at"]) < timestamp(opening["opened_at"]):
                    raise ValueError("Final outcome predates instruction lock")
                rows = payload["rows"]
                if len(rows) != len(references) or {row["case_id"] for row in rows} != set(references):
                    raise ValueError("Missing or duplicate final predictions")
                for row in rows:
                    expected_label = references[row["case_id"]]["expected_label"]
                    if row["expected_label"] != expected_label:
                        raise ValueError("Final reference label mismatch")
                    if row["correct"] != (row["predicted_label"] == expected_label):
                        raise ValueError("Stored score differs from shared canonical scoring")
                scores[arm] = sum(row["correct"] for row in rows)
                failures[arm] = sum(isinstance(row["predicted_label"], str) and row["predicted_label"].startswith("__") for row in rows)
                deploy_status[arm] = (prepared[(model, task_id, arm)]["method"].get("status")
                                      if arm in {"full", "gepa"} else None)
                if arm != "direct_dsl":
                    events = audit_calls(output / "calls" / f"{model.replace(':', '_')}-{task_id}-{arm}-final.jsonl",
                                         payload["call_summary"], protocol, final=True, opening=opening)
                    final_requests = {row["metadata"]["case_id"]: row for row in events if row["event"] == "model_request"}
                    if set(final_requests) != set(references):
                        raise ValueError("Final ledger does not match the retained prediction denominator")
                    if any(event["recorded_at_unix"] > timestamp(payload["completed_at"]).timestamp() + ISO_RESOLUTION_SECONDS for event in events):
                        raise ValueError("Final call continued after its recorded cell completion")
                    if timestamp(payload["completed_at"]) > timestamp(completion["completed_at"]):
                        raise ValueError("Final cell completed after run completion")
                    for row in rows:
                        request = final_requests[row["case_id"]]
                        if request["system"] != row["system"] or request["prompt"] != row["prompt"]:
                            raise ValueError("Final prediction uses different messages from its call ledger")
                        if request["model_identity"]["digest"] != identities[model]["digest"]:
                            raise ValueError("Final solver differs from frozen model identity")
                        if row["input_record"] != references[row["case_id"]]["input_record"]:
                            raise ValueError("Retained final input differs from committed case")
                        method = (PreparedMethod.raw(tasks[task_id]) if arm == "raw" else
                                  PreparedMethod.from_dict(prepared[(model, task_id, arm)]["method"]))
                        instruction, certificate = method.instructions_for(row["input_record"], router)
                        system, prompt = solver_messages(tasks[task_id], instruction, row["input_record"])
                        if row["system"] != system or row["prompt"] != prompt or row["routing_certificate"] != certificate:
                            raise ValueError("Final request differs from frozen prepared instructions or routing replay")
                        if (request["metadata"]["prepared_hash"] != method.prepared_hash
                            or request["metadata"]["instruction_hash"] != content_hash(instruction)):
                            raise ValueError("Final prepared/instruction hash differs from frozen artifact")
                        terminal = next(event for event in events if event.get("attempt_id") == request["attempt_id"]
                                        and event["event"] in {"model_response", "model_failure"})
                        raw_text = terminal["text"] if terminal["event"] == "model_response" else MODEL_FAILURE_TEXT
                        if row["student_text"] != raw_text or parse_label(raw_text, tasks[task_id]) != row["predicted_label"]:
                            raise ValueError("Final stored label differs from reparsed original response")
                    final_costs.append({"solver_model": model, "task_id": task_id, "arm": arm,
                         "wall_seconds": payload["final_wall_seconds"], "calls": payload["call_summary"]})
                else:
                    for row in rows:
                        prediction = (execute_dsl(learned_program, references[row["case_id"]]["input_record"]).output_label
                                      if learned_program else "__NO_VERIFIED_PROGRAM__")
                        if row["predicted_label"] != prediction:
                            raise ValueError("Direct control differs from frozen teacher-generated program")
                evidence[str(path.relative_to(output))] = sha(path)
            all_task_rows.append({"solver_model": model, "task_id": task_id, "family": number % 4,
                "denominator": len(references), "correct": scores, "invalid": failures,
                "deployment_status": deploy_status,
                "full_minus_gepa": (scores["full"] - scores["gepa"]) / len(references)})
    expected_final_files = len(protocol["task_ids"]) * len(protocol["solver_models"]) * 4
    if len(list((output / "final").glob("*.json"))) != expected_final_files:
        raise ValueError("Unexpected extra final artifacts")
    # Include every raw model request/response/failure ledger, not only successes.
    expected_ledgers = {
        f"{model.replace(':', '_')}-policy-{number:03d}-{arm}{suffix}.jsonl"
        for model in protocol["solver_models"] for number in protocol["task_ids"]
        for arm, suffix in (("full", ""), ("gepa", ""), ("raw", "-final"), ("full", "-final"), ("gepa", "-final"))}
    if {path.name for path in (output / "calls").glob("*.jsonl")} != expected_ledgers:
        raise ValueError("Unexpected call-ledger inventory; no paid attempts may be omitted")
    actual_student_attempts = actual_backend_failures = 0
    for path in sorted((output / "calls").glob("*.jsonl")):
        evidence[str(path.relative_to(output))] = sha(path)
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        actual_student_attempts += sum(row["event"] == "model_request" and row.get("role", "").startswith("solver_") for row in events)
        actual_backend_failures += sum(row["event"] == "model_failure" and row.get("role", "").startswith("solver_")
                                      and row.get("failure_kind") == "backend" for row in events)
    if completion["total_student_attempts"] != actual_student_attempts or completion["backend_student_failures"] != actual_backend_failures:
        raise ValueError("Run totals differ from retained student attempts/backend errors")
    per_model = []
    for model in protocol["solver_models"]:
        rows = [row for row in all_task_rows if row["solver_model"] == model]
        differences = [row["full_minus_gepa"] for row in rows]
        per_model.append({"solver_model": model, "tasks": len(rows),
            "total_cases": sum(row["denominator"] for row in rows),
            "correct": {arm: sum(row["correct"][arm] for row in rows) for arm in ("raw", "full", "gepa", "direct_dsl")},
            "macro_accuracy": {arm: statistics.mean(row["correct"][arm] / row["denominator"] for row in rows)
                               for arm in ("raw", "full", "gepa", "direct_dsl")},
            "paired_full_minus_gepa": statistics.mean(differences),
            "descriptive_task_bootstrap_95_interval": interval(differences, protocol["seed"]),
            "wins": sum(value > 0 for value in differences), "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
            "family_mean_difference": {str(family): statistics.mean(row["full_minus_gepa"] for row in rows if row["family"] == family)
                                       for family in sorted({row["family"] for row in rows})}})
    costs = []
    for model in protocol["solver_models"]:
        for arm in ("full", "gepa"):
            cells = [row for row in preparation_costs if row["solver_model"] == model and row["arm"] == arm]
            roles = {role: {field: sum(row["calls"]["roles"].get(role, {}).get(field, 0) for row in cells)
                           for field in ("attempted_calls", "failed_calls", "input_tokens_known", "output_tokens_known",
                                         "unknown_input_usage_calls", "unknown_output_usage_calls", "latency_seconds")}
                     for role in ("solver_optimization", "teacher_optimization")}
            methods = [prepared[(model, f"policy-{number:03d}", arm)]["method"] for number in protocol["task_ids"]]
            costs.append({"solver_model": model, "arm": arm, "preparation_wall_seconds": sum(row["wall_seconds"] for row in cells),
                          "roles": roles, "fallback_tasks": sum(method["status"] == "raw_fallback" for method in methods),
                          "native_edited_candidates": sum(method["trace"].get("edited_candidate_count", 0) for method in methods)})
    return {"status": "complete_audited_development_comparison",
            "protocol_id": protocol["protocol_id"], "models": per_model,
            "tasks": all_task_rows, "preparation_costs": preparation_costs,
            "aggregate_preparation_costs": costs, "final_costs": final_costs, "evidence_sha256": evidence,
            "limitations": protocol["claim_limits"],
            "independence": "Eight related synthetic policy tasks per model; four reused families. Bootstrap is descriptive, not evidence of new-domain or independent-family superiority.",
            "budget_match": "Same public information and model-call ceilings; actual calls, tokens, routing computation and latency can differ."}


def markdown(summary: dict) -> str:
    lines = ["# Matched original-pipeline comparison", "",
             "Audited development study: eight fixed policies and two frozen small models. All instructions were locked before final cases were scored.", "",
             "| Student | Raw | Full DSL pipeline | Native GEPA | Direct teacher DSL | Full − GEPA |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary["models"]:
        cells = [f"{row['correct'][arm]}/{row['total_cases']}" for arm in ("raw", "full", "gepa", "direct_dsl")]
        lines.append(f"| {row['solver_model']} | " + " | ".join(cells) + f" | {100*row['paired_full_minus_gepa']:.2f} pp |")
    lines += ["", "Both algorithms had the same public description/schema and 54 labeled practice cases, the same local teacher and student, the same 216 student-call and four teacher-call ceilings, and the same scoring interface. Actual expenditure differs; the Full method also uses MiniLM routing.", "",
              "| Student | Full task wins / ties / losses | Descriptive paired task bootstrap interval |",
              "|---|---:|---:|"]
    for row in summary["models"]:
        lo, hi = row["descriptive_task_bootstrap_95_interval"]
        lines.append(f"| {row['solver_model']} | {row['wins']} / {row['ties']} / {row['losses']} | [{100*lo:.2f}, {100*hi:.2f}] pp |")
    lines += ["", "This is a controlled development comparison on four reused synthetic families. It does not establish novelty, fresh confirmation, general NLP transfer, or publication readiness. The direct-DSL column executes the generated teacher program, never the hidden reference program. All failed compilations and model attempts remain in the evidence.", "",
              "| Student | Method | Student preparation calls | Teacher calls | Preparation wall seconds | Raw fallbacks |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summary["aggregate_preparation_costs"]:
        roles = row["roles"]
        lines.append(f"| {row['solver_model']} | {row['arm']} | {roles['solver_optimization']['attempted_calls']} | {roles['teacher_optimization']['attempted_calls']} | {row['preparation_wall_seconds']:.1f} | {row['fallback_tasks']} |")
    lines += ["",
              "Detailed task rows, actual costs, statuses and artifact hashes are in `summary.json`. The runner's protocol, source snapshot, instruction lock and raw call ledgers are retained alongside it.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.output)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    report = markdown(summary)
    for name, content in (("summary.json", rendered), ("results.md", report)):
        target = args.output / name
        if target.exists():
            if target.read_text(encoding="utf-8") != content:
                raise ValueError(f"Existing report differs from deterministic replay: {name}")
        else:
            target.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "models": summary["models"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
