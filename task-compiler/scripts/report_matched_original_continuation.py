"""Audit the retained original evidence plus the prospectively frozen recovery."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from report_matched_original import summarize, markdown, sha, read, timestamp, ISO_RESOLUTION_SECONDS


def audit_recovery(output: Path):
    amendment = read(output / "continuation_amendment.json")
    preflight = read(output / "continuation_preflight.json")
    snapshot = read(output / "continuation_source_snapshot.json")
    start = read(output / "continuation_start.json")
    predecessor = Path(start["predecessor_path"])
    if not (timestamp(snapshot["frozen_at"]) <= timestamp(start["started_at"])
            <= timestamp(read(output/"prepared_lock.json")["locked_at"])):
        raise ValueError("Recovery freeze/start/lock chronology differs")
    if read(output/"completion.json")["continuation_amendment_sha256"] != sha(output/"continuation_amendment.json"):
        raise ValueError("Recovery completion amendment commitment differs")
    backend=read(output/"continuation_backend_identity.json")
    if backend["server_version"] != read(output/"backend_identity.json")["server_version"] or backend["model_identities"] != read(output/"model_identities.json"):
        raise ValueError("Resumed runtime/model identities differ")
    if sha(output / "protocol.json") != amendment["original_protocol_sha256"]:
        raise ValueError("Original protocol changed during recovery")
    if read(output/"source_snapshot.json")["commit"] != amendment["original_source_commit"]:
        raise ValueError("Original source identity changed")
    for name, expected in snapshot["source_hashes"].items():
        if sha(output/"continuation_source"/name) != expected:
            raise ValueError("Recovery source snapshot changed")
    if sha(output/"continuation_amendment.json") != snapshot["source_hashes"]["protocols/matched_original/continuation_20261006_v1.json"]:
        raise ValueError("Executed amendment differs from committed recovery snapshot")
    affected = amendment["affected_cell"]
    partial_name = f"calls/{affected}.jsonl"
    for name, expected in preflight["predecessor_files_sha256"].items():
        original = predecessor/name; copied = output/name
        if sha(original) != expected:
            raise ValueError("Immutable predecessor changed")
        if name == partial_name:
            if not copied.read_bytes().startswith(original.read_bytes()):
                raise ValueError("Interrupted ledger prefix changed")
        elif sha(copied) != expected:
            raise ValueError("Completed predecessor evidence changed in continuation")
    if preflight["model_inference_calls"] != 0 or preflight["replayed_solver_responses"] != 57:
        raise ValueError("Exact zero-inference preflight missing")
    events = [json.loads(line) for line in (output/partial_name).read_text(encoding="utf-8").splitlines()]
    start_time=timestamp(start["started_at"]).timestamp()
    for path in (output/"calls").glob("*.jsonl"):
        name=str(path.relative_to(output)).replace("\\","/")
        if name in preflight["predecessor_files_sha256"]:
            if name!=partial_name: continue
            prefix_count=len((predecessor/name).read_text(encoding="utf-8").splitlines())
        else: prefix_count=0
        new_events=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][prefix_count:]
        if any(r["recorded_at_unix"]+ISO_RESOLUTION_SECONDS<start_time for r in new_events):
            raise ValueError("New model events predate prospective recovery freeze/start")
    lost = [r for r in events if r["event"] == "model_failure" and r.get("failure_kind") == "external_interruption"]
    if len(lost) != 1 or lost[0]["attempt_id"] != "teacher_optimization-0001" or not lost[0].get("completion_unknown") or not lost[0].get("latency_unknown"):
        raise ValueError("Unobserved attempt is not retained honestly")
    if lost[0].get("input_tokens") is not None or lost[0].get("output_tokens") is not None or "latency_seconds" in lost[0]:
        raise ValueError("Unknown cost was fabricated")
    requests = [r for r in events if r["event"] == "model_request" and r["role"] == "teacher_optimization"]
    if len(requests) < 2:
        raise ValueError("Interrupted request was not continued")
    fields = ("system", "prompt", "output_schema", "metadata", "model_identity")
    if any(requests[0][key] != requests[1][key] for key in fields):
        raise ValueError("Repeated unanswered request differs from the original")
    row = read(output/"prepared"/f"{affected}.json")
    if row["recovery"]["replayed_solver_responses"] != 57 or row["recovery"]["unknown_teacher_attempts"] != 1:
        raise ValueError("Recovery counters differ")
    return {"amendment_id":amendment["amendment_id"],"affected_cell":affected,
        "completed_artifacts_reused_unchanged":24,"replayed_solver_responses_without_dispatch":57,
        "unobserved_teacher_attempts_charged":1,"unknown_token_usage_calls":1,"unknown_latency_calls":1,
        "predecessor_files_sha256":preflight["predecessor_files_sha256"],"recovery_source":snapshot,
        "limitations":amendment["claim_limits"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("output",type=Path); args=parser.parse_args()
    recovery = audit_recovery(args.output)
    summary = summarize(args.output)  # All original cost, chronology, model, parser and frozen-method audits.
    sensitivity = [r for r in summary["tasks"] if r["solver_model"]=="llama3.2:3b" and r["task_id"]!="policy-004"]
    if len(sensitivity)!=7: raise ValueError("Unexpected affected-task sensitivity inventory")
    recovery["amendment_secondary_sensitivity"]={"solver_model":"llama3.2:3b","omitted_task":"policy-004",
        "tasks":7,"mean_full_minus_gepa":statistics.mean(r["full_minus_gepa"] for r in sensitivity),
        "scope":"Descriptive diagnostic specified after interruption but before final opening; primary remains all eight tasks."}
    summary["status"]="complete_audited_amended_development_comparison"
    summary["interruption_recovery"]=recovery
    summary["limitations"] += recovery["limitations"]
    report=markdown(summary)+"\n## Process-interruption amendment\n\nThe original process stopped before final opening. All 24 completed preparations were reused unchanged; 57 saved student responses reconstructed the affected GEPA cell without new dispatch. Its unanswered teacher attempt remains charged within the four-attempt ceiling. Token use and duration for that attempt are unknown, so reported token/latency sums are known amounts, and its preparation wall time is a lower bound. The prospective amendment grants six additional hours for continuation. This is an amended development comparison, not an uninterrupted replication.\n"
    for name,content in (("summary.json",json.dumps(summary,indent=2,ensure_ascii=False)+"\n"),("results.md",report)):
        target=args.output/name
        if target.exists():
            if target.read_text(encoding="utf-8") != content: raise ValueError("Deterministic recovery report changed")
        else: target.write_text(content,encoding="utf-8",newline="\n")
    print(json.dumps({"status":summary["status"],"models":summary["models"]},indent=2))


if __name__=="__main__":main()
