"""Prospective recovery of a process-interrupted, pre-final matched study.

Completed responses are replayed locally, never sampled again. The one
unanswered parent attempt remains charged. No algorithm or prompt is edited.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from matched_original import (EventLedger, ModelSession, PublicTask, PublicCase,
    PreparedMethod, Limits, compile_full, compile_gepa, evaluate_prepared,
    make_router, canonical_json)
from run_matched_original import (ROOT, now, digest, write_json, verify_sources,
    verify_native_package, lock_prepared, verify_prepared_lock, open_final_cases)

AMENDMENT = ROOT / "protocols/matched_original/continuation_20261006_v1.json"
PREDECESSOR = ROOT / "reports/matched_original/development_20261005_v1"
RECOVERED_CELL = "llama3.2_3b-policy-004-gepa"


class RecoveryBoundary(RuntimeError):
    """Offline reconstruction reached the first request requiring inference."""


class ReplaySession(ModelSession):
    def __init__(self, model, ledger, role, limit, replay_events, orphan_event=None, allow_fresh=False):
        super().__init__(model, ledger, role, limit)
        self.allow_fresh = allow_fresh
        self.orphan_event = copy.deepcopy(orphan_event)
        self.orphan_recovered = False
        self.replay_consumed = 0
        self.cached = []
        requests = [r for r in replay_events if r["event"] == "model_request" and r.get("role") == role]
        if len({r["attempt_id"] for r in requests}) != len(requests):
            raise ValueError("Duplicate historical request")
        for request in requests:
            terminals = [r for r in replay_events if r.get("attempt_id") == request["attempt_id"]
                         and r["event"] in {"model_response", "model_failure"}]
            if request == orphan_event:
                if terminals or request != requests[-1]:
                    raise ValueError("Orphan must be the last unanswered request")
                continue
            if len(terminals) != 1 or terminals[0]["event"] != "model_response":
                raise ValueError("Recovery requires exactly one saved successful response")
            self.cached.append((copy.deepcopy(request), copy.deepcopy(terminals[0])))
        if orphan_event is not None:
            if orphan_event not in requests:
                raise ValueError("Orphan is absent from historical requests")
            self.attempts = 1  # Consumed even though its outcome is unobserved.

    def _match(self, request, system, prompt, schema, metadata, seed):
        effective_seed = self.identity.get("seed") if seed is None else seed
        actual = (system, prompt, schema or {}, metadata or {}, effective_seed)
        expected = (request["system"], request["prompt"], request["output_schema"],
                    request.get("metadata", {}), request["model_identity"].get("seed"))
        if canonical_json(actual) != canonical_json(expected):
            raise ValueError("Offline recovery request differs from the retained request")

    def generate(self, system, prompt, schema=None, *, metadata=None, seed=None, failure_is_invalid=True):
        from taskcompiler.procedural.experiment import Response
        if self.replay_consumed < len(self.cached):
            request, terminal = self.cached[self.replay_consumed]
            self._match(request, system, prompt, schema, metadata, seed)
            self.reserve_batch(1)
            self.attempts += 1
            self.replay_consumed += 1
            self.last_failure_kind = None
            return Response(terminal["text"], terminal["input_tokens"],
                            terminal["output_tokens"], terminal["latency_seconds"])
        if self.orphan_event is not None and not self.orphan_recovered:
            self._match(self.orphan_event, system, prompt, schema, metadata, seed)
        if not self.allow_fresh:
            raise RecoveryBoundary("Exact replay reached the unanswered parent request; zero new inference")
        if self.orphan_event is not None and not self.orphan_recovered:
            self.ledger.append("model_failure", role=self.role,
                attempt_id=self.orphan_event["attempt_id"], error_type="ExternalSessionInterruption",
                error="Process disappeared; original response, token usage and duration are unobserved.",
                failure_kind="external_interruption", input_tokens=None, output_tokens=None,
                latency_unknown=True, completion_unknown=True)
            self.orphan_recovered = True
        return super().generate(system, prompt, schema, metadata=metadata, seed=seed,
                                failure_is_invalid=failure_is_invalid)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def limits_for(protocol):
    return Limits(solver_calls=protocol["optimization_solver_call_ceiling_per_task_arm"],
        teacher_calls=protocol["optimization_teacher_call_ceiling_per_task_arm"],
        teacher_tokens=protocol["teacher_max_output_tokens"], max_instruction_words=protocol["max_instruction_words"],
        top_k=protocol["full"]["router_top_k"], seed=protocol["seed"])


def inspect_predecessor():
    if any((PREDECESSOR / name).exists() for name in
           ("completion.json", "prepared_lock.json", "final_opening.json", "interruption.json")):
        raise ValueError("Unexpected predecessor stage; this amendment is specific to pre-final process loss")
    if len(list((PREDECESSOR / "prepared").glob("*.json"))) != 24:
        raise ValueError("Expected exactly 24 completed predecessor preparations")
    if (PREDECESSOR / "final").exists():
        raise ValueError("Final outcomes must remain unopened")
    source = read(PREDECESSOR / "source_snapshot.json")
    amendment = read(AMENDMENT)
    if source["commit"] != amendment["original_source_commit"] or digest(PREDECESSOR/"protocol.json") != amendment["original_protocol_sha256"]:
        raise ValueError("Predecessor differs from the declared original study")
    verify_sources(source["source_hashes"])
    package_hashes = read(PREDECESSOR / "native_package_snapshot.json")["files_sha256"]
    verify_native_package(package_hashes)
    for name, expected in package_hashes.items():
        if digest(PREDECESSOR/"dependencies"/name) != expected: raise ValueError("Copied GEPA source changed")
    for name, expected in source["source_hashes"].items():
        if digest(PREDECESSOR / "source" / name) != expected:
            raise ValueError("Copied original source changed")
    events = [json.loads(line) for line in (PREDECESSOR / "calls" / f"{RECOVERED_CELL}.jsonl").read_text(encoding="utf-8").splitlines()]
    from report_matched_original import audit_calls
    protocol = read(PREDECESSOR/"protocol.json")
    expected_prepared = {f"{model}-policy-{number:03d}-{arm}.json"
        for model, numbers in (("qwen2.5_3b",range(8)),("llama3.2_3b",range(4)))
        for number in numbers for arm in ("full","gepa")}
    if {p.name for p in (PREDECESSOR/"prepared").glob("*.json")} != expected_prepared:
        raise ValueError("Completed predecessor cell inventory changed")
    for path in (PREDECESSOR/"prepared").glob("*.json"):
        row=read(path)
        audit_calls(PREDECESSOR/"calls"/f"{path.stem}.jsonl",row["call_summary"],protocol,final=False,opening={})
    requests = [r for r in events if r["event"] == "model_request"]
    terminal_ids = {r["attempt_id"] for r in events if r["event"] in {"model_response", "model_failure"}}
    orphans = [r for r in requests if r["attempt_id"] not in terminal_ids]
    if (len(orphans) != 1 or orphans[0]["role"] != "teacher_optimization"
        or sum(r["role"] == "solver_optimization" for r in requests) != 57
        or sum(r["role"] == "teacher_optimization" for r in requests) != 1):
        raise ValueError("Unexpected interrupted request prefix")
    inventory = {str(p.relative_to(PREDECESSOR)): digest(p) for p in sorted(PREDECESSOR.rglob("*")) if p.is_file()}
    return events, orphans[0], inventory


def offline_preflight():
    events, orphan, inventory = inspect_predecessor()
    protocol = read(PREDECESSOR / "protocol.json")
    tasks = [PublicTask.from_dict(row) for row in read(PREDECESSOR / "public_tasks.json")]
    task = next(t for t in tasks if t.task_id == "policy-004")
    class NoDispatch:
        def __init__(self, identity): self.identity = identity
        def generate(self, *args, **kwargs): raise AssertionError("Offline preflight dispatched a model")
    identities = read(PREDECESSOR / "model_identities.json")
    ledger = EventLedger(); ledger.events = copy.deepcopy(events)
    solver = ReplaySession(NoDispatch(identities["llama3.2:3b"]), ledger, "solver_optimization", 216, events)
    teacher = ReplaySession(NoDispatch(identities["qwen3:8b"]), ledger, "teacher_optimization", 4, events, orphan)
    try:
        compile_gepa(task, solver, teacher, limits_for(protocol))
    except RecoveryBoundary:
        if solver.replay_consumed != 57 or solver.calls != 57 or teacher.calls != 1 or ledger.events != events:
            raise ValueError("Incomplete or mutated offline replay")
    else:
        raise ValueError("Offline reconstruction did not reach the expected boundary")
    return {"status": "exact_prefix_replayed_without_inference", "model_inference_calls": 0,
        "replayed_solver_responses": 57, "charged_unanswered_teacher_attempts": 1,
        "remaining_solver_attempts": solver.remaining, "remaining_teacher_attempts": teacher.remaining,
        "predecessor_files_sha256": inventory}


def run(output):
    amendment = read(AMENDMENT)
    proof = offline_preflight()  # Fails before any model initialization or dispatch.
    if output.exists() or not output.resolve().is_relative_to((ROOT / "reports/matched_original").resolve()):
        raise ValueError("Continuation requires a fresh grouped output directory")
    recovery_names = ["scripts/continue_matched_original.py", "scripts/report_matched_original_continuation.py",
                      "protocols/matched_original/continuation_20261006_v1.json"]
    tracked = subprocess.check_output(["git", "ls-files", "--", *recovery_names], cwd=ROOT, text=True).splitlines()
    if set(tracked) != set(recovery_names) or subprocess.check_output(["git", "diff", "HEAD", "--", *tracked], cwd=ROOT):
        raise ValueError("Commit unchanged recovery implementation and amendment before inference")
    recovery_hashes = {name: digest(ROOT / name) for name in recovery_names}
    shutil.copytree(PREDECESSOR, output)
    shutil.copyfile(AMENDMENT, output / "continuation_amendment.json")
    write_json(output / "continuation_preflight.json", proof)
    write_json(output / "continuation_source_snapshot.json", {"frozen_at": now(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": recovery_hashes})
    for name in recovery_names:
        target = output / "continuation_source" / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    write_json(output / "continuation_start.json", {"started_at": now(), "cause_of_process_loss": "unknown",
        "predecessor_path": str(PREDECESSOR), "original_final_cases_opened": False,
        "unknown_request_duration": True, "additional_wall_allowance_seconds": amendment["additional_wall_allowance_seconds"]})
    protocol = read(output / "protocol.json"); limits = limits_for(protocol)
    started = time.monotonic()
    sources = read(output / "source_snapshot.json")["source_hashes"]
    packages = read(output / "native_package_snapshot.json")["files_sha256"]
    from taskcompiler.procedural.local_model import OllamaLocal
    models = {}
    for name in [protocol["teacher_model"], *protocol["solver_models"]]:
        model = OllamaLocal(name, endpoint=protocol["endpoint"], seed=protocol["seed"],
            max_output_tokens=protocol["teacher_max_output_tokens"] if name == protocol["teacher_model"] else protocol["student_max_output_tokens"],
            context_tokens=protocol["context_tokens"], timeout=protocol["model_timeout_seconds"])
        model.identity["structured_decoding"] = "per_request_output_schema"
        if model.identity != read(output / "model_identities.json")[name]:
            raise ValueError("Resumed model identity differs from original")
        models[name] = model
    if models[protocol["teacher_model"]]._request("/api/version") != read(output / "backend_identity.json")["server_version"]:
        raise ValueError("Resumed runtime differs from original")
    write_json(output/"continuation_backend_identity.json", {"verified_at":now(),
        "server_version":models[protocol["teacher_model"]]._request("/api/version"),
        "model_identities":{name:model.identity for name,model in models.items()},
        "models":{name:model._request("/api/show",{"model":name}) for name,model in models.items()}})
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    router_record = read(output / "router_identity.json")
    snapshot = Path(router_record["identity"]["local_snapshot"])
    for name, expected in router_record["files_sha256"].items():
        if digest(snapshot / name) != expected: raise ValueError("Pinned router changed")
    router = make_router(snapshot)
    tasks = [PublicTask.from_dict(row) for row in read(output / "public_tasks.json")]
    original_events = [json.loads(line) for p in (output/"calls").glob("*.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
    attempted_student = sum(r["event"]=="model_request" and r.get("role","").startswith("solver_") for r in original_events)
    backend_student_failures = 0

    def check_stop():
        verify_sources(sources); verify_sources(recovery_hashes); verify_native_package(packages)
        if time.monotonic() - started > amendment["additional_wall_allowance_seconds"]:
            raise RuntimeError("Continuation wall allowance reached")
        if shutil.disk_usage(ROOT).free < protocol["stop_rules"]["minimum_free_workspace_bytes"]:
            raise RuntimeError("Free-space stop rule reached")
        if attempted_student and backend_student_failures / attempted_student > .05:
            raise RuntimeError("Student backend-error stop rule reached")
        for name, expected in proof["predecessor_files_sha256"].items():
            if digest(PREDECESSOR / name) != expected: raise ValueError("Predecessor evidence changed")

    def ledger_at(path, context, events=None):
        if events is None: return EventLedger(path, context=context)
        ledger = EventLedger(context=context); ledger.path = path; ledger.events = copy.deepcopy(events)
        return ledger

    try:
        for mi, model_name in enumerate(protocol["solver_models"]):
            for ti, task in enumerate(tasks):
                for arm in (["full", "gepa"] if (mi+ti)%2 == 0 else ["gepa", "full"]):
                    check_stop(); cell = f"{model_name.replace(':', '_')}-{task.task_id}-{arm}"
                    path = output / "prepared" / f"{cell}.json"
                    if path.exists(): continue  # Exact predecessor bytes, including failures.
                    context = {"cell": cell, "task_id": task.task_id, "solver_model": model_name, "arm": arm}
                    partial_seconds = 0.0
                    if cell == RECOVERED_CELL:
                        events, orphan, _ = inspect_predecessor()
                        ledger = ledger_at(output/"calls"/f"{cell}.jsonl", context, events)
                        solver = ReplaySession(models[model_name], ledger, "solver_optimization",216,events,allow_fresh=True)
                        teacher = ReplaySession(models[protocol["teacher_model"]],ledger,"teacher_optimization",4,events,orphan,True)
                        partial_seconds = events[-1]["recorded_at_unix"]-events[0]["recorded_at_unix"]
                    else:
                        ledger = ledger_at(output/"calls"/f"{cell}.jsonl",context)
                        solver = ModelSession(models[model_name],ledger,"solver_optimization",216)
                        teacher = ModelSession(models[protocol["teacher_model"]],ledger,"teacher_optimization",4)
                    cell_started = time.monotonic()
                    method = compile_full(task,solver,teacher,router,limits) if arm=="full" else compile_gepa(task,solver,teacher,limits)
                    record = {"task_id":task.task_id,"solver_model":model_name,"arm":arm,"prepared_at":now(),
                        "method":method.to_dict(),"preparation_wall_seconds":partial_seconds+time.monotonic()-cell_started,
                        "call_summary":ledger.summary()}
                    if cell == RECOVERED_CELL:
                        record["recovery"]={"replayed_solver_responses":solver.replay_consumed,
                            "unknown_teacher_attempts":1,"wall_time_kind":"known_lower_bound; interrupted duration unknown"}
                    write_json(path,record)
                    attempted_student += solver.calls - (57 if cell == RECOVERED_CELL else 0)
                    backend_student_failures += solver.backend_failures
                    print(f"PREPARED {cell} solver={solver.calls} teacher={teacher.calls}",flush=True)
        expected = 32; paths = sorted((output/"prepared").glob("*.json"))
        check_stop(); lock_prepared(output,paths,expected); final_data = open_final_cases(output,expected)
        for mi, model_name in enumerate(protocol["solver_models"]):
            for ti, task in enumerate(tasks):
                cases = [PublicCase(**row) for row in final_data[task.task_id]]
                arms = ["raw","full","gepa"]
                if (mi+ti)%2: arms.reverse()
                for arm in arms:
                    check_stop(); verify_prepared_lock(output,expected)
                    cell=f"{model_name.replace(':','_')}-{task.task_id}-{arm}"
                    method=PreparedMethod.raw(task) if arm=="raw" else PreparedMethod.from_dict(read(output/"prepared"/f"{cell}.json")["method"])
                    ledger=EventLedger(output/"calls"/f"{cell}-final.jsonl",context={"cell":cell,"task_id":task.task_id,"solver_model":model_name,"arm":arm})
                    solver=ModelSession(models[model_name],ledger,"solver_final",27); cell_started=time.monotonic()
                    rows=evaluate_prepared(task,method,cases,solver,router)
                    write_json(output/"final"/f"{cell}.json",{"task_id":task.task_id,"solver_model":model_name,"arm":arm,
                        "completed_at":now(),"rows":rows,"final_wall_seconds":time.monotonic()-cell_started,"call_summary":ledger.summary()})
                    attempted_student += solver.calls
                    backend_student_failures += solver.backend_failures
                    print(f"FINAL {cell} correct={sum(r['correct'] for r in rows)}/27",flush=True)
                from taskcompiler.dsl.parser import parse_program
                from taskcompiler.dsl.interpreter import run as execute_dsl
                full=read(output/"prepared"/f"{model_name.replace(':','_')}-{task.task_id}-full.json")["method"]
                program=parse_program(full["dsl_source"],task_spec=task.spec) if full.get("dsl_source") else None
                rows=[]
                for case in cases:
                    predicted=execute_dsl(program,case.input_record).output_label if program else "__NO_VERIFIED_PROGRAM__"
                    rows.append({"case_id":case.case_id,"expected_label":case.expected_label,"predicted_label":predicted,"correct":predicted==case.expected_label})
                write_json(output/"final"/f"{model_name.replace(':','_')}-{task.task_id}-direct_dsl.json",{
                    "task_id":task.task_id,"solver_model":model_name,"arm":"direct_dsl","completed_at":now(),
                    "rows":rows,"program_available":bool(program),"additional_model_calls":0})
        events=[json.loads(line) for p in (output/"calls").glob("*.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
        check_stop(); verify_prepared_lock(output,expected)
        write_json(output/"completion.json",{"status":"complete","completed_at":now(),
            "elapsed_seconds":time.monotonic()-started,"elapsed_scope":"continuation only; predecessor retained separately",
            "total_student_attempts":sum(r["event"]=="model_request" and r.get("role","").startswith("solver_") for r in events),
            "backend_student_failures":sum(r["event"]=="model_failure" and r.get("role","").startswith("solver_") and r.get("failure_kind")=="backend" for r in events),
            "prepared_lock_sha256":digest(output/"prepared_lock.json"),"continuation_amendment_sha256":digest(output/"continuation_amendment.json")})
    except BaseException as exc:
        write_json(output/"interruption.json",{"status":"interrupted","recorded_at":now(),"reason":str(exc),
            "exception_type":type(exc).__name__,"final_opened":(output/"final_opening.json").exists(),"automatic_retry":False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path); parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    if args.preflight_only:
        proof=offline_preflight(); print(json.dumps({k:v for k,v in proof.items() if k!="predecessor_files_sha256"},indent=2)); return
    if args.output is None: parser.error("--output is required for inference")
    run(args.output.resolve())


if __name__=="__main__": main()
