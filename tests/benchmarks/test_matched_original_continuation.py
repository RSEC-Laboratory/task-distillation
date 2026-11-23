"""Offline checks for a declared interruption continuation, not model results.

The fixtures contain only invented public request strings and saved responses.
Completed calls must be replayed exactly without model dispatch; an interrupted
teacher attempt remains charged before any prospectively declared fresh call.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace

import pytest

from taskcompiler.procedural.experiment import Response
from test_matched_original import FakeModel, load_script, matched, runner


sys.modules.setdefault("run_matched_original", runner)
continuation = load_script("matched_original_continuation_fixture", "continue_matched_original.py")
load_script("report_matched_original", "report_matched_original.py")
recovery_reporter = load_script("matched_original_recovery_reporter_fixture", "report_matched_original_continuation.py")


@pytest.fixture(autouse=True)
def no_live_models_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Continuation software fixture attempted live model/network access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    from taskcompiler.procedural import local_model
    monkeypatch.setattr(local_model.OllamaLocal, "__init__", forbidden)


class RecordedFixtureModel(FakeModel):
    """Small fixture producing the production ledger's complete event shape."""

    def __init__(self, role, *, forbid_dispatch=False):
        super().__init__(None)
        self.seed = 20261005
        self.identity = {"mode": "fixture", "local_only": True,
                         "model": f"offline-{role}", "seed": self.seed}
        self.forbid_dispatch = forbid_dispatch
        self.dispatched = []

    def _request(self, suffix, body=None):
        assert suffix == "/api/chat"
        number = len(self.dispatched)
        return {"done": True, "done_reason": "stop",
                "message": {"role": "assistant", "content": f"saved-response-{number}"},
                "prompt_eval_count": 10 + number, "eval_count": 2 + number}

    def generate(self, system, prompt, schema):
        if self.forbid_dispatch:
            raise AssertionError("A completed historical call was dispatched again")
        self.dispatched.append((system, prompt, deepcopy(schema), self.seed))
        payload = self._request("/api/chat", {
            "model": self.identity["model"],
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "options": {"seed": self.seed},
        })
        return Response(payload["message"]["content"], payload["prompt_eval_count"],
                        payload["eval_count"], 0.125 + len(self.dispatched) / 100)


def solver_history(count=57):
    model = RecordedFixtureModel("solver")
    ledger = matched.EventLedger()
    session = matched.ModelSession(model, ledger, "solver", 216)
    requests, responses = [], []
    for index in range(count):
        request = {"system": f"public-system-{index}", "prompt": f"public-input-{index}",
                   "schema": {}, "metadata": {"phase": "gepa_public_evaluation", "case": index},
                   "seed": 20261005}
        requests.append(request)
        responses.append(session.generate(**request))
    return ledger, requests, responses


def interrupted_teacher_history(ledger):
    temporary = matched.EventLedger()
    model = RecordedFixtureModel("teacher")
    session = matched.ModelSession(model, temporary, "teacher", 4)
    request = {"system": "public-reflection-system", "prompt": "public-reflection-body",
               "schema": {}, "metadata": {"phase": "gepa_reflection"}, "seed": 20261005}
    session.generate(**request, failure_is_invalid=False)
    # An external interruption retained the attempted request and transport body,
    # but supplied neither a raw response nor a normalized terminal response.
    rows = deepcopy(temporary.events[:2])
    for row in rows:
        row["event_index"] = len(ledger.events)
        ledger.events.append(row)
    return rows[0], request


def replay_session(ledger, role="solver", *, allow_fresh=False, orphan=None, model=None):
    model = model or RecordedFixtureModel(role, forbid_dispatch=True)
    return continuation.ReplaySession(
        model, ledger, role, 216 if role == "solver" else 4,
        replay_events=deepcopy(ledger.events), orphan_event=deepcopy(orphan), allow_fresh=allow_fresh,
    ), model


def assert_same_response(actual, original):
    assert actual.text == original.text
    assert actual.input_tokens == original.input_tokens
    assert actual.output_tokens == original.output_tokens
    assert actual.latency_seconds == original.latency_seconds


def test_all_57_completed_solver_calls_replay_exactly_without_dispatch_or_duplicate_events():
    ledger, requests, responses = solver_history()
    before = deepcopy(ledger.events)
    session, model = replay_session(ledger)
    assert session.calls == 0 and session.remaining == 216
    for index, (request, original) in enumerate(zip(requests, responses, strict=True), start=1):
        assert_same_response(session.generate(**request), original)
        assert session.calls == index and session.remaining == 216 - index
    assert session.calls == 57 and session.remaining == 159
    assert model.dispatched == []
    assert ledger.events == before
    with pytest.raises(continuation.RecoveryBoundary):
        session.generate("fresh-system", "fresh-prompt")
    assert session.calls == 57 and ledger.events == before


@pytest.mark.parametrize("changed", ["system", "prompt", "schema", "metadata", "seed"])
def test_replay_rejects_any_changed_request_component_before_dispatch(changed):
    ledger, requests, _ = solver_history(1)
    session, model = replay_session(ledger)
    request = deepcopy(requests[0])
    request[changed] = {"changed": True} if changed in {"schema", "metadata"} else (
        20261006 if changed == "seed" else request[changed] + "-changed")
    before = deepcopy(ledger.events)
    with pytest.raises(ValueError):
        session.generate(**request)
    assert session.calls == 0 and model.dispatched == [] and ledger.events == before


def test_orphan_is_charged_once_at_initialization_and_dry_boundary_does_not_close_it():
    ledger, requests, responses = solver_history()
    orphan, teacher_request = interrupted_teacher_history(ledger)
    before = deepcopy(ledger.events)
    solver, solver_model = replay_session(ledger)
    teacher, teacher_model = replay_session(ledger, "teacher", orphan=orphan)
    assert teacher.calls == 1 and teacher.remaining == 3
    for request, original in zip(requests, responses, strict=True):
        assert_same_response(solver.generate(**request), original)
    with pytest.raises(continuation.RecoveryBoundary):
        teacher.generate(**teacher_request, failure_is_invalid=False)
    assert solver.calls == 57 and solver.remaining == 159
    assert teacher.calls == 1 and teacher.remaining == 3
    assert solver_model.dispatched == teacher_model.dispatched == []
    assert ledger.events == before
    assert ledger.summary()["unfinished_calls"] == 1


def test_declared_fresh_teacher_call_closes_unknown_orphan_and_uses_attempt_0002():
    ledger, _, _ = solver_history(1)
    orphan, teacher_request = interrupted_teacher_history(ledger)
    historical_events = deepcopy(ledger.events)
    model = RecordedFixtureModel("teacher")
    teacher, _ = replay_session(ledger, "teacher", orphan=orphan, allow_fresh=True, model=model)
    assert teacher.calls == 1
    response = teacher.generate(**teacher_request, failure_is_invalid=False)
    assert response.text == "saved-response-1"
    assert teacher.calls == 2 and teacher.remaining == 2
    assert len(model.dispatched) == 1
    assert ledger.events[:len(historical_events)] == historical_events
    added = ledger.events[len(historical_events):]
    assert [row["event"] for row in added] == [
        "model_failure", "model_request", "transport_request", "transport_response", "model_response"]
    closure = added[0]
    assert closure["attempt_id"] == "teacher-0001"
    assert closure["failure_kind"] == "external_interruption"
    assert closure["input_tokens"] is None and closure["output_tokens"] is None
    assert closure["latency_unknown"] is True and "latency_seconds" not in closure
    assert all(row["attempt_id"] == "teacher-0002" for row in added[1:])
    assert sum(row["event"] == "model_failure" and row["attempt_id"] == "teacher-0001"
               for row in ledger.events) == 1
    summary = ledger.summary()
    assert summary["roles"]["teacher"]["attempted_calls"] == 2
    assert summary["roles"]["teacher"]["unknown_input_usage_calls"] == 1
    assert summary["roles"]["teacher"]["unknown_output_usage_calls"] == 1
    assert teacher.backend_failures == 0


def test_orphan_is_closed_only_once_and_total_teacher_budget_includes_it():
    ledger, _, _ = solver_history(1)
    orphan, request = interrupted_teacher_history(ledger)
    model = RecordedFixtureModel("teacher")
    teacher, _ = replay_session(ledger, "teacher", orphan=orphan, allow_fresh=True, model=model)
    for index in range(3):
        teacher.generate(**request, failure_is_invalid=False)
        assert teacher.calls == index + 2
    assert teacher.remaining == 0 and len(model.dispatched) == 3
    before = deepcopy(ledger.events)
    with pytest.raises(matched.BudgetExceeded):
        teacher.generate(**request, failure_is_invalid=False)
    assert ledger.events == before and len(model.dispatched) == 3
    assert sum(row["event"] == "model_failure" and row["attempt_id"] == "teacher-0001"
               for row in ledger.events) == 1
    assert [row["attempt_id"] for row in ledger.events
            if row["event"] == "model_request" and row["role"] == "teacher"] == [
                "teacher-0001", "teacher-0002", "teacher-0003", "teacher-0004"]


@pytest.mark.parametrize("changed", ["system", "prompt", "schema", "metadata", "seed"])
def test_fresh_teacher_request_must_match_orphan_before_closure_or_dispatch(changed):
    ledger, _, _ = solver_history(1)
    orphan, request = interrupted_teacher_history(ledger)
    before = deepcopy(ledger.events)
    model = RecordedFixtureModel("teacher")
    teacher, _ = replay_session(ledger, "teacher", orphan=orphan, allow_fresh=True, model=model)
    request[changed] = {"changed": True} if changed in {"schema", "metadata"} else (
        20261006 if changed == "seed" else request[changed] + "-changed")
    with pytest.raises(ValueError):
        teacher.generate(**request, failure_is_invalid=False)
    assert teacher.calls == 1 and model.dispatched == [] and ledger.events == before


@pytest.mark.parametrize("defect", ["duplicate_terminal", "missing_terminal", "failure_as_response", "duplicate_request"])
def test_malformed_historical_solver_terminal_records_are_rejected(defect):
    ledger, _, _ = solver_history(1)
    if defect == "duplicate_terminal":
        ledger.events.append(deepcopy(ledger.events[-1]))
    elif defect == "missing_terminal":
        ledger.events.pop()
    elif defect == "failure_as_response":
        ledger.events[-1]["event"] = "model_failure"
        ledger.events[-1]["failure_kind"] = "backend"
    else:
        ledger.events.append(deepcopy(ledger.events[0]))
    with pytest.raises(ValueError):
        replay_session(ledger)


def test_24_retained_preparations_cannot_open_final_cases_when_32_are_required(tmp_path, monkeypatch):
    paths = []
    for index in range(24):
        path = tmp_path / "prepared" / f"public-frozen-{index}.json"
        runner.write_json(path, {"software_fixture": index})
        paths.append(path)
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError, match="All task/model/algorithm"):
        runner.lock_prepared(tmp_path, paths, 32)
    assert not (tmp_path / "prepared_lock.json").exists()
    with pytest.raises(ValueError, match="global instruction lock"):
        runner.open_final_cases(tmp_path, 32)
    assert not (tmp_path / "final_opening.json").exists()
    assert all(path.read_bytes() == original for path, original in before.items())


def test_continuation_preserves_and_skips_24_completed_artifacts_without_inference(tmp_path, monkeypatch):
    """Use only public fixture metadata; stop before any final case access."""
    root = tmp_path / "fixture-project"
    parent = root / "evaluation" / "results" / "matched_gepa" / "predecessor"
    parent.mkdir(parents=True)
    output = parent.parent / "continuation"
    amendment_path = root / "experiments" / "protocols" / "matched_original" / "continuation_20261006_v1.json"
    amendment_path.parent.mkdir(parents=True)
    amendment_path.write_text(json.dumps({"additional_wall_allowance_seconds": 60}), encoding="utf-8")
    protocol = {
        "teacher_model": "fixture:teacher", "solver_models": ["fixture:solver"],
        "seed": 20261005, "endpoint": "http://offline.invalid",
        "teacher_max_output_tokens": 2000, "student_max_output_tokens": 16,
        "context_tokens": 32768, "model_timeout_seconds": 600,
        "optimization_solver_call_ceiling_per_task_arm": 216,
        "optimization_teacher_call_ceiling_per_task_arm": 4,
        "max_instruction_words": 1800, "full": {"router_top_k": 6},
        "stop_rules": {"minimum_free_workspace_bytes": 0},
    }
    identities = {name: {"mode": "fixture", "local_only": True, "model": name,
                         "structured_decoding": "per_request_output_schema"}
                  for name in ("fixture:teacher", "fixture:solver")}
    tasks = [{"task_id": f"fixture-{index:03d}"} for index in range(12)]
    for name, payload in {
        "protocol.json": protocol, "model_identities.json": identities,
        "source_snapshot.json": {"source_hashes": {}},
        "native_package_snapshot.json": {"files_sha256": {}},
        "backend_identity.json": {"server_version": {"version": "offline"}},
        "router_identity.json": {"identity": {"local_snapshot": "unused-fixture-snapshot"},
                                 "files_sha256": {}},
        "public_tasks.json": tasks,
    }.items():
        runner.write_json(parent / name, payload)
    retained = {}
    for task in tasks:
        for arm in ("full", "gepa"):
            path = parent / "prepared" / f"fixture_solver-{task['task_id']}-{arm}.json"
            runner.write_json(path, {"retained_offline_fixture": path.name})
            retained[path.name] = path.read_bytes()
    recovery_names = ["experiments/drivers/continue_matched_original.py",
                      "experiments/drivers/report_matched_original_continuation.py",
                      "experiments/protocols/matched_original/continuation_20261006_v1.json"]
    for name in recovery_names[:2]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# offline fixture snapshot\n", encoding="utf-8")

    def git_fixture(args, **kwargs):
        if args[1] == "ls-files":
            return "\n".join(recovery_names)
        if args[1] == "rev-parse":
            return "offline-fixture-commit\n"
        assert args[1] == "diff"
        return b""

    def forbidden(*args, **kwargs):
        raise AssertionError("A retained preparation was compiled or a model/final case was accessed")

    class OfflineModel:
        def __init__(self, name, **kwargs):
            self.identity = deepcopy(identities[name])

        def _request(self, suffix, body=None):
            if suffix == "/api/version":
                assert body is None
                return {"version": "offline"}
            assert suffix == "/api/show" and body == {"model": self.identity["model"]}
            return {"model_info": {"fixture": True}, "template": "offline fixture"}

        generate = forbidden

    class ExpectedBarrier(RuntimeError):
        pass

    def barrier(path, paths, expected):
        assert path == output and expected == 32 and len(paths) == 24
        assert {item.name: item.read_bytes() for item in paths} == retained
        raise ExpectedBarrier("24 preserved preparations; final cases remain closed")

    monkeypatch.setattr(continuation, "ROOT", root)
    monkeypatch.setattr(continuation, "PREDECESSOR", parent)
    monkeypatch.setattr(continuation, "AMENDMENT", amendment_path)
    monkeypatch.setattr(continuation, "offline_preflight", lambda: {"predecessor_files_sha256": {}})
    monkeypatch.setattr(continuation.subprocess, "check_output", git_fixture)
    monkeypatch.setattr(continuation, "verify_sources", lambda _: None)
    monkeypatch.setattr(continuation, "verify_native_package", lambda _: None)
    monkeypatch.setattr(continuation.PublicTask, "from_dict", lambda row: SimpleNamespace(**row))
    monkeypatch.setattr(continuation, "make_router", lambda _: SimpleNamespace())
    monkeypatch.setattr(continuation, "compile_full", forbidden)
    monkeypatch.setattr(continuation, "compile_gepa", forbidden)
    monkeypatch.setattr(continuation, "lock_prepared", barrier)
    monkeypatch.setattr(continuation, "open_final_cases", forbidden)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from taskcompiler.procedural import local_model
    monkeypatch.setattr(local_model, "OllamaLocal", OfflineModel)
    with pytest.raises(ExpectedBarrier, match="final cases remain closed"):
        continuation.run(output)
    assert {path.name: path.read_bytes() for path in (output / "prepared").glob("*.json")} == retained
    assert {path.name: path.read_bytes() for path in (parent / "prepared").glob("*.json")} == retained
    assert not (output / "final_opening.json").exists()
    assert not (output / "final").exists()


def rewrite_fixture_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def rewrite_fixture_events(path, rows):
    path.write_text("".join(matched.canonical_json(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def recovery_audit_fixture(tmp_path):
    """Recovery audit metadata only: no sealed cases or final model results."""
    parent = tmp_path / "predecessor"
    parent.mkdir()
    output = tmp_path / "continued"
    affected = "software-fixture-gepa"
    base = datetime(2020, 1, 2, tzinfo=timezone.utc)
    ledger, _, _ = solver_history()
    orphan, request = interrupted_teacher_history(ledger)
    for index, row in enumerate(ledger.events):
        row["role"] += "_optimization"
        prefix, number = row["attempt_id"].rsplit("-", 1)
        row["attempt_id"] = f"{prefix}_optimization-{number}"
        row["recorded_at_unix"] = base.timestamp() - 5 + index / 1000
    orphan = deepcopy(ledger.events[-2])
    for index in range(24):
        runner.write_json(parent / "prepared" / f"retained-{index:02d}.json", {"software_fixture": index})
    runner.write_json(parent / "protocol.json", {"software_fixture": True})
    runner.write_json(parent / "source_snapshot.json", {"commit": "offline-original-commit"})
    identities = {"fixture:teacher": {"model": "fixture:teacher", "seed": 20261005}}
    runner.write_json(parent / "model_identities.json", identities)
    runner.write_json(parent / "backend_identity.json", {"server_version": {"version": "offline"}})
    partial = Path("calls") / f"{affected}.jsonl"
    (parent / partial).parent.mkdir()
    rewrite_fixture_events(parent / partial, ledger.events)
    inventory = {str(path.relative_to(parent)).replace("\\", "/"): runner.digest(path)
                 for path in parent.rglob("*") if path.is_file()}
    shutil.copytree(parent, output)
    amendment = {
        "amendment_id": "OFFLINE_RECOVERY_AUDIT_FIXTURE",
        "affected_cell": affected,
        "original_protocol_sha256": runner.digest(parent / "protocol.json"),
        "original_source_commit": "offline-original-commit",
        "claim_limits": ["Software fixture only; no empirical result."],
    }
    runner.write_json(output / "continuation_amendment.json", amendment)
    amendment_name = "experiments/protocols/matched_original/continuation_20261006_v1.json"
    snapshot_path = output / "continuation_source" / amendment_name
    snapshot_path.parent.mkdir(parents=True)
    shutil.copyfile(output / "continuation_amendment.json", snapshot_path)
    runner.write_json(output / "continuation_source_snapshot.json", {
        "frozen_at": (base - timedelta(seconds=1)).isoformat(),
        "source_hashes": {amendment_name: runner.digest(snapshot_path)},
    })
    runner.write_json(output / "continuation_start.json", {
        "started_at": base.isoformat(), "predecessor_path": str(parent),
    })
    runner.write_json(output / "continuation_preflight.json", {
        "model_inference_calls": 0, "replayed_solver_responses": 57,
        "predecessor_files_sha256": inventory,
    })
    runner.write_json(output / "continuation_backend_identity.json", {
        "server_version": {"version": "offline"}, "model_identities": identities,
    })
    runner.write_json(output / "prepared_lock.json", {"locked_at": (base + timedelta(seconds=1)).isoformat()})
    runner.write_json(output / "completion.json", {
        "continuation_amendment_sha256": runner.digest(output / "continuation_amendment.json"),
    })
    teacher = continuation.ReplaySession(RecordedFixtureModel("teacher"), ledger,
        "teacher_optimization", 4, deepcopy(ledger.events), orphan, allow_fresh=True)
    teacher.generate(**request, failure_is_invalid=False)
    historical_count = len((parent / partial).read_text(encoding="utf-8").splitlines())
    for index, event in enumerate(ledger.events[historical_count:]):
        event["recorded_at_unix"] = base.timestamp() + 0.1 + index / 1000
    rewrite_fixture_events(output / partial, ledger.events)
    runner.write_json(output / "prepared" / f"{affected}.json", {
        "recovery": {"replayed_solver_responses": 57, "unknown_teacher_attempts": 1},
    })
    return output, parent, affected


def test_recovery_reporter_retains_unknown_cost_and_zero_dispatch_replay(recovery_audit_fixture):
    output, parent, affected = recovery_audit_fixture
    audit = recovery_reporter.audit_recovery(output)
    assert audit["affected_cell"] == affected
    assert audit["completed_artifacts_reused_unchanged"] == 24
    assert audit["replayed_solver_responses_without_dispatch"] == 57
    assert audit["unobserved_teacher_attempts_charged"] == 1
    assert audit["unknown_token_usage_calls"] == audit["unknown_latency_calls"] == 1
    assert len(list((parent / "prepared").glob("*.json"))) == 24
    assert not (output / "sealed").exists() and not (output / "final").exists()


@pytest.mark.parametrize("defect", ["input_zero", "output_zero", "duration_zero", "latency_known", "completion_known", "duplicate_unknown"])
def test_recovery_reporter_rejects_fabricated_or_duplicate_unknown_attempts(recovery_audit_fixture, defect):
    output, _, affected = recovery_audit_fixture
    path = output / "calls" / f"{affected}.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lost = next(row for row in events if row["event"] == "model_failure")
    if defect == "input_zero":
        lost["input_tokens"] = 0
    elif defect == "output_zero":
        lost["output_tokens"] = 0
    elif defect == "duration_zero":
        lost["latency_seconds"] = 0
    elif defect == "latency_known":
        lost["latency_unknown"] = False
    elif defect == "completion_known":
        lost["completion_unknown"] = False
    else:
        events.append(deepcopy(lost))
    rewrite_fixture_events(path, events)
    with pytest.raises(ValueError):
        recovery_reporter.audit_recovery(output)


@pytest.mark.parametrize("field", ["system", "prompt", "output_schema", "metadata", "model_identity"])
def test_recovery_reporter_rejects_changed_repeated_pending_request(recovery_audit_fixture, field):
    output, _, affected = recovery_audit_fixture
    path = output / "calls" / f"{affected}.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    repeated = next(row for row in events if row.get("attempt_id") == "teacher_optimization-0002"
                    and row["event"] == "model_request")
    repeated[field] = "changed" if field in {"system", "prompt"} else {"changed": True}
    rewrite_fixture_events(path, events)
    with pytest.raises(ValueError, match="differs from the original"):
        recovery_reporter.audit_recovery(output)


@pytest.mark.parametrize("defect", ["early_event", "changed_prefix", "changed_predecessor", "nonzero_preflight", "replay_count", "freeze_order", "resumed_identity"])
def test_recovery_reporter_rejects_timeline_identity_or_evidence_tampering(recovery_audit_fixture, defect):
    output, parent, affected = recovery_audit_fixture
    path = output / "calls" / f"{affected}.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if defect == "early_event":
        next(row for row in events if row["event"] == "model_failure")["recorded_at_unix"] = (
            datetime(2020, 1, 2, tzinfo=timezone.utc).timestamp() - 0.001)
        rewrite_fixture_events(path, events)
    elif defect == "changed_prefix":
        events[0]["system"] += " changed"
        rewrite_fixture_events(path, events)
    elif defect == "changed_predecessor":
        (parent / "prepared" / "retained-00.json").write_text("changed original", encoding="utf-8")
    else:
        name = "continuation_backend_identity.json" if defect == "resumed_identity" else (
            "continuation_source_snapshot.json" if defect == "freeze_order" else "continuation_preflight.json")
        target = output / name
        value = json.loads(target.read_text(encoding="utf-8"))
        if defect == "nonzero_preflight":
            value["model_inference_calls"] = 1
        elif defect == "replay_count":
            value["replayed_solver_responses"] = 56
        elif defect == "freeze_order":
            value["frozen_at"] = datetime(2020, 1, 3, tzinfo=timezone.utc).isoformat()
        else:
            value["model_identities"]["fixture:teacher"]["seed"] += 1
        rewrite_fixture_json(target, value)
    with pytest.raises(ValueError):
        recovery_reporter.audit_recovery(output)
