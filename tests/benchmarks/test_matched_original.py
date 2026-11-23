"""Offline boundary and official-GEPA integration checks; no empirical evidence.

Fake actors implement deliberately simple three-label behavior.  GEPA itself,
the legacy DSL compiler/verifier/renderer, and the global final-data barrier
execute their production code.  No model client, retriever, or network is used.
"""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import itertools
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from taskcompiler.procedural.experiment import ModelCallError, Response
from taskcompiler.schema import EnumType, FieldDecl, RecordType, TaskSpec


ROOT = Path(__file__).resolve().parents[2]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / "drivers" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


matched = load_script("matched_original", "matched_original.py")
runner = load_script("matched_original_runner_fixture", "run_matched_original.py")

DSL = """rule Granted:
    when event.decision == YES
    then require label = allow

rule Rejected:
    when event.decision == NO
    then require label = deny

rule Undecided:
    when event.decision == MAYBE
    then abstain
"""


@pytest.fixture(autouse=True)
def no_network_or_live_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline software fixture attempted live model/network access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    from taskcompiler.procedural import local_model
    from taskcompiler import student, teacher
    monkeypatch.setattr(local_model.OllamaLocal, "__init__", forbidden)
    monkeypatch.setattr(student.StudentModel, "__init__", forbidden)
    monkeypatch.setattr(teacher.TeacherClient, "__init__", forbidden)


@pytest.fixture
def task_and_final():
    specification = TaskSpec(
        task_name="fixture_permission",
        description="Return allow for YES, deny for NO, and uncertain for MAYBE in decision. Ignore noise fields.",
        labels=["allow", "deny"],
        requirements=["Use only the decision field; preserve the unresolved MAYBE outcome."],
        sources=["software-fixture-public-spec"],
        enums=[EnumType("Choice", ["YES", "NO", "MAYBE"])],
        records=[RecordType("Event", [FieldDecl(name, "Choice") for name in ("a", "b", "c", "decision")])],
    )
    partitions = [[], [], []]
    for i, indexes in enumerate(itertools.product(range(3), repeat=4)):
        values = tuple(("YES", "NO", "MAYBE")[index] for index in indexes)
        record = {"event": dict(zip(("a", "b", "c", "decision"), values, strict=True))}
        expected = {"YES": "allow", "NO": "deny", "MAYBE": None}[values[-1]]
        # Input-only assignment gives every split all values of every field,
        # as required by the existing finite-domain representation certifier.
        partitions[sum(indexes) % 3].append(matched.PublicCase(f"fixture-{i:02d}", record, expected))
    return matched.PublicTask("fixture", specification, partitions[0], partitions[1]), partitions[2]


class FakeModel:
    identity = {"mode": "fixture", "local_only": True, "model": "offline-symbolic-actor"}

    def __init__(self, responder):
        self.responder = responder
        self.requests = []

    def generate(self, system, prompt, schema):
        self.requests.append((system, prompt, deepcopy(schema)))
        value = self.responder(system, prompt, schema)
        return value if isinstance(value, Response) else Response(value, 11, 3, 0.0)


def answer_input(system, prompt, schema):
    record = json.loads(prompt.removeprefix("Input: "))
    return {"YES": "allow", "NO": "deny", "MAYBE": "uncertain"}[record["event"]["decision"]]


def sessions(limits, solver_fn=answer_input, teacher_fn=None, ledger=None):
    ledger = ledger or matched.EventLedger()
    solver_model = FakeModel(solver_fn)
    teacher_model = FakeModel(teacher_fn or (lambda *_: json.dumps({"program": DSL})))
    return (matched.ModelSession(solver_model, ledger, "solver", limits.solver_calls),
            matched.ModelSession(teacher_model, ledger, "teacher", limits.teacher_calls),
            solver_model, teacher_model, ledger)


class FixtureRouter:
    """An explicit no-retriever fixture; it retains all supplied instructions."""
    identity = {"mode": "fixture", "implementation": "retain_all_test_instructions"}

    def __init__(self):
        self.requests = []

    def route(self, record, instructions, *, top_k, label_encoder):
        self.requests.append((deepcopy(record), list(instructions), top_k, label_encoder))
        return SimpleNamespace(instructions=list(instructions), certificate={"software_fixture": True})


def test_public_dto_snapshots_only_disclosed_fields_and_is_immutable(task_and_final):
    task, final = task_and_final
    payload = task.to_dict()
    rebuilt = matched.PublicTask.from_dict(payload)
    original_hash = rebuilt.task_hash
    payload["build_cases"][0]["input_record"]["event"]["decision"] = "NO"
    payload["spec"]["description"] = "changed outside the snapshot"
    rebuilt.spec.description = "changed returned copy"
    rebuilt.build_cases[0].input_record["event"]["decision"] = "NO"
    assert rebuilt.task_hash == original_hash == task.task_hash
    assert set(rebuilt.to_dict()) == {"task_id", "spec", "build_cases", "audit_cases"}
    assert len(task.build_cases) == len(task.audit_cases) == len(final) == 27
    assert len({case.input_json for case in (*task.public_cases, *final)}) == 81
    for split in (task.build_cases, task.audit_cases, final):
        assert all({case.input_record["event"][field] for case in split} == {"YES", "NO", "MAYBE"}
                   for field in ("a", "b", "c", "decision"))
    assert all(not hasattr(rebuilt, name) for name in ("oracle_program", "final_cases"))


@pytest.mark.parametrize("key", ["oracle_program", "final_cases", "private_query"])
def test_optimizer_dto_rejects_hidden_task_fields(task_and_final, key):
    payload = task_and_final[0].to_dict()
    payload[key] = "HIDDEN_SENTINEL"
    with pytest.raises(ValueError, match="only public"):
        matched.PublicTask.from_dict(payload)


@pytest.mark.parametrize("change", ["duplicate_id", "overlap_input", "unknown_field", "unknown_value", "unknown_label", "nested_oracle"])
def test_public_disjointness_and_schema_reject_invalid_disclosed_data(task_and_final, change):
    payload = task_and_final[0].to_dict()
    case = payload["audit_cases"][0]
    if change == "duplicate_id":
        case["case_id"] = payload["build_cases"][0]["case_id"]
    elif change == "overlap_input":
        case["input_record"] = deepcopy(payload["build_cases"][0]["input_record"])
    elif change == "unknown_field":
        case["input_record"]["event"]["private"] = "YES"
    elif change == "unknown_value":
        case["input_record"]["event"]["decision"] = "UNDECLARED"
    elif change == "unknown_label":
        case["expected_label"] = "invented"
    else:
        payload["spec"]["oracle_program"] = DSL
    with pytest.raises(ValueError):
        matched.PublicTask.from_dict(payload)


@pytest.mark.parametrize("value,expected", [("allow", "allow"), ("uncertain", None), ("", "invalid"), ("unparseable gibberish", "invalid"),
                                            (matched.MODEL_FAILURE_TEXT, "invalid"), (matched.CANDIDATE_FAILURE_TEXT, "invalid")])
def test_uncertain_is_a_valid_outcome_but_invalid_or_backend_failure_is_not(task_and_final, value, expected):
    parsed = matched.parse_label(value, task_and_final[0])
    if expected == "invalid":
        assert isinstance(parsed, str) and parsed.startswith("__INVALID__")
        assert parsed is not None
    else:
        assert parsed == expected


def test_attempt_is_durably_written_before_dispatch_and_failure_exhausts_budget(tmp_path):
    path = tmp_path / "calls.jsonl"
    ledger = matched.EventLedger(path)

    def fail_after_check(*_):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows[-1]["event"] == "model_request"
        assert rows[-1]["system"] == "system" and rows[-1]["prompt"] == "request"
        raise ModelCallError("fixture truncated", input_tokens=7, output_tokens=9)

    model = FakeModel(fail_after_check)
    session = matched.ModelSession(model, ledger, "solver", 1)
    response = session.generate("system", "request")
    assert response.text == matched.MODEL_FAILURE_TEXT
    assert (response.input_tokens, response.output_tokens) == (7, 9)
    assert session.calls == 1 and session.remaining == 0 and session.backend_failures == 1
    before = path.read_bytes()
    with pytest.raises(matched.BudgetExceeded):
        session.generate("system", "unsent")
    assert path.read_bytes() == before and len(model.requests) == 1
    assert ledger.summary()["failed_calls"] == 1
    assert ledger.summary()["unfinished_calls"] == 0
    with pytest.raises(FileExistsError):
        matched.EventLedger(path)


def test_teacher_backend_failure_stops_without_retry_and_unknown_usage_stays_unknown():
    def broken(*_):
        raise RuntimeError("fixture transport error")
    ledger = matched.EventLedger()
    session = matched.ModelSession(FakeModel(broken), ledger, "teacher", 4)
    with pytest.raises(matched.BackendFailure, match="transport error"):
        session.generate("system", "prompt", failure_is_invalid=False)
    assert session.calls == 1
    assert ledger.summary()["roles"]["teacher"]["unknown_input_usage_calls"] == 1
    assert ledger.summary()["roles"]["teacher"]["unknown_output_usage_calls"] == 1


def test_raw_truncated_payload_is_retained_and_distinguished_from_transport_failure():
    class TruncatedModel:
        identity = FakeModel.identity
        seed = 7

        def _request(self, suffix, body=None):
            return {"done_reason": "length", "response": "incomplete", "prompt_eval_count": 5, "eval_count": 9}

        def generate(self, system, prompt, schema):
            self._request("/api/generate", {"seed": self.seed, "prompt": prompt})
            raise ModelCallError("Local output reached its token cap; truncated responses are failures", input_tokens=5, output_tokens=9)

    model = TruncatedModel()
    ledger = matched.EventLedger()
    session = matched.ModelSession(model, ledger, "solver", 1)
    response = session.generate("system", "prompt", seed=11)
    assert response.text == matched.MODEL_FAILURE_TEXT and session.calls == 1
    assert session.backend_failures == 0 and model.seed == 7
    assert [event["event"] for event in ledger.events] == ["model_request", "transport_request", "transport_response", "model_failure"]
    assert ledger.events[1]["body"]["seed"] == 11
    assert ledger.events[2]["payload"]["response"] == "incomplete"
    assert ledger.events[3]["failure_kind"] == "truncation"


def test_complete_batch_reservation_rejects_without_partial_dispatch():
    session = matched.ModelSession(FakeModel(lambda *_: "allow"), matched.EventLedger(), "solver", 2)
    with pytest.raises(matched.BudgetExceeded, match="complete"):
        session.reserve_batch(3)
    assert session.calls == 0 and not session.ledger.events


def test_legacy_probe_and_final_use_the_same_exact_solver_contract(task_and_final):
    task, final = task_and_final
    instruction = "Inspect decision; return its declared outcome."
    case = final[0]
    limits = matched.Limits()
    model = FakeModel(answer_input)
    ledger = matched.EventLedger()
    public_session = matched.ModelSession(model, ledger, "solver", 2)
    student = matched.CanonicalStudent(task, public_session, limits)
    student.ask(f"Input: {case.input_record}", system="Task: ignored legacy header\n\nRules:\n- " + instruction +
                "\n\nOutput contract: legacy wrapper is replaced")
    prepared = matched.PreparedMethod("native_gepa", task.task_id, task.task_hash, (instruction,), False, "prepared", "{}")
    final_session = matched.ModelSession(model, ledger, "solver_final", 1)
    result = matched.evaluate_prepared(task, prepared, [case], final_session)
    assert model.requests[0] == model.requests[1] == (*matched.solver_messages(task, instruction, case.input_record), {})
    assert result[0]["correct"]


def test_full_generates_verifies_and_behaviorally_selects_without_an_oracle(task_and_final, monkeypatch):
    task, final = task_and_final
    import taskcompiler.translate as translate
    import taskcompiler.rules_track.synthetic_policies as synthetic
    original = translate.build_obligation_probes
    oracle_arguments = []

    def public_probes(*args, **kwargs):
        oracle_arguments.append(kwargs.get("oracle_program"))
        assert kwargs.get("oracle_program") is None
        return original(*args, **kwargs)

    monkeypatch.setattr(translate, "build_obligation_probes", public_probes)
    monkeypatch.setattr(synthetic, "make_policy_task", lambda *_: pytest.fail("Compiler attempted hidden task/oracle access"))
    limits = matched.Limits(representation_rounds=1, semantic_repair_rounds=0, counterexample_patches=0)
    solver, teacher, solver_model, teacher_model, _ = sessions(limits)
    router = FixtureRouter()
    prepared = matched.compile_full(task, solver, teacher, router, limits)
    assert teacher.calls == 1 and teacher_model.requests
    assert prepared.status == "prepared" and prepared.routed
    assert prepared.trace["initial_source_supplied"] is False
    assert prepared.trace["oracle_program_access"] is False
    assert prepared.trace["dsl"]["public_verification_passed"] is True
    assert "rule Granted" in prepared.to_dict()["dsl_source"]
    assert prepared.trace["behavioral_probe_ids"] == [case.case_id for case in task.build_cases]
    assert oracle_arguments and all(value is None for value in oracle_arguments)
    disclosed = {case.input_json for case in task.public_cases}
    assert solver_model.requests and all(matched.canonical_json(json.loads(prompt.removeprefix("Input: "))) in disclosed
                                         for _, prompt, _ in solver_model.requests)
    assert router.requests and all(row[2:] == (limits.top_k, None) for row in router.requests)
    assert matched.PreparedMethod.from_dict(prepared.to_dict()).prepared_hash == prepared.prepared_hash
    # Only the final evaluator receives undisclosed cases.
    model = FakeModel(answer_input)
    rows = matched.evaluate_prepared(task, prepared, final[:3],
                                    matched.ModelSession(model, matched.EventLedger(), "solver_final", 3), router)
    assert all(row["correct"] and row["routing_certificate"]["software_fixture"] for row in rows)


@pytest.mark.parametrize("teacher_output,stage", [("not a DSL program", "teacher_syntax"),
    (json.dumps({"program": "rule Granted:\n    when event.decision == YES\n    then require label = allow"}), "teacher_validation_gate")])
def test_full_invalid_teacher_program_keeps_the_task_with_explicit_raw_fallback(task_and_final, teacher_output, stage):
    task = task_and_final[0]
    limits = matched.Limits(semantic_repair_rounds=0, counterexample_patches=0, representation_rounds=0)
    solver, teacher, solver_model, _, _ = sessions(limits, teacher_fn=lambda *_: teacher_output)
    prepared = matched.compile_full(task, solver, teacher, FixtureRouter(), limits)
    assert prepared.status == "raw_fallback" and not prepared.routed
    assert prepared.trace["failure"]["stage"] == stage
    assert prepared.instructions == (matched.public_task_text(task),)
    assert prepared.to_dict()["dsl_source"] is None
    assert teacher.calls == 1 and solver.calls == 0 and not solver_model.requests


def test_real_native_gepa_reflects_edits_and_selects_a_fully_evaluated_improvement(task_and_final):
    task = task_and_final[0]
    # First improvement requires 54 initial + 3 parent + 3 proposal + 54 full;
    # the actual frozen 216-call ceiling must also stop subsequent native work.
    limits = matched.Limits(solver_calls=216, teacher_calls=4)
    def improvable(system, prompt, schema):
        return answer_input(system, prompt, schema) if "IMPROVED" in system else "unparseable gibberish"
    solver, teacher, model, editor, _ = sessions(limits, improvable, lambda *_: "```\nIMPROVED: use decision exactly.\n```")
    prepared = matched.compile_gepa(task, solver, teacher, limits)
    trace = prepared.trace
    native = trace["native_result"]
    assert trace["implementation"] == "official_gepa_default_adapter" and trace["gepa_version"] == "0.1.4"
    assert trace["candidate_count"] >= 2 and trace["edited_candidate_count"] >= 1
    assert matched.PreparedMethod.from_dict(prepared.to_dict()).prepared_hash == prepared.prepared_hash
    assert native["parents"][1] == [0]
    assert native["val_aggregate_scores"][0] == 0
    assert native["val_aggregate_scores"][trace["best_candidate_index"]] == 1
    assert "IMPROVED" in prepared.instructions[0]
    assert teacher.calls >= 1 and editor.requests
    assert solver.calls <= limits.solver_calls and teacher.calls <= limits.teacher_calls
    selected_hash = matched.content_hash({"system_prompt": prepared.instructions[0]})
    complete = [event for event in trace["native_history"] if event["event"] == "evaluation_completed"
                and event["candidate_hash"] == selected_hash and len(event["case_ids"]) == 54]
    assert complete and complete[-1]["scores"] == [1.0] * 54
    assert trace["public_train_case_ids"] == trace["public_validation_case_ids"] == [case.case_id for case in task.public_cases]
    # GEPA's seed and every proposal use the same task/input/output wrapper.
    for system, prompt, schema in model.requests:
        instruction = system.split("Task instructions:\n", 1)[1].rsplit("\n\nOutput contract:", 1)[0]
        record = json.loads(prompt.removeprefix("Input: "))
        assert (system, prompt) == matched.solver_messages(task, instruction, record)
        assert schema == {}
    assert all("final-" not in prompt and "oracle_program" not in prompt for _, prompt, _ in editor.requests)


def test_native_budget_too_small_for_iteration_keeps_fully_scored_seed(task_and_final):
    task = task_and_final[0]
    limits = matched.Limits(solver_calls=54, teacher_calls=1)
    solver, teacher, _, editor, _ = sessions(limits)
    prepared = matched.compile_gepa(task, solver, teacher, limits)
    assert solver.calls == 54 and teacher.calls == 0 and not editor.requests
    assert prepared.trace["candidate_count"] == 1
    assert prepared.instructions == (matched.public_task_text(task),)
    assert prepared.trace["native_result"]["val_aggregate_scores"] == [1.0]


def test_native_oversized_proposal_never_dispatches_or_displaces_seed(task_and_final):
    task = task_and_final[0]
    limits = matched.Limits(solver_calls=114, teacher_calls=1, max_instruction_words=300)
    solver, teacher, model, _, _ = sessions(limits, lambda *_: "unparseable gibberish", lambda *_: "```\n" + "OVERSIZED " * 301 + "\n```")
    prepared = matched.compile_gepa(task, solver, teacher, limits)
    assert teacher.calls == 1 and prepared.trace["candidate_word_guard"]
    assert not any("OVERSIZED" in system for system, _, _ in model.requests)
    assert prepared.instructions == (matched.public_task_text(task),)
    assert solver.calls <= limits.solver_calls


def test_native_internal_reflection_retry_cannot_dispatch_fifth_teacher_call(task_and_final, monkeypatch):
    from gepa.strategies.instruction_proposal import InstructionProposalSignature
    extract = InstructionProposalSignature.output_extractor

    def faulty_extract(cls, output):
        if output == "MALFORMED":
            raise ValueError("Deliberate software-fixture native proposal parse failure")
        return extract(output)

    # Only parsing is fault-injected; the official native search/retry/state
    # machinery remains intact, including its batched-to-single-call retry.
    monkeypatch.setattr(InstructionProposalSignature, "output_extractor", classmethod(faulty_extract))
    attempts = []
    def reflection(*_):
        attempts.append(len(attempts) + 1)
        return "```\nVALID_BUT_NO_GAIN\n```" if len(attempts) == 1 else "MALFORMED"

    limits = matched.Limits(solver_calls=216, teacher_calls=4)
    solver, teacher, _, editor, ledger = sessions(limits, lambda *_: "unparseable gibberish", reflection)
    prepared = matched.compile_gepa(task_and_final[0], solver, teacher, limits)
    trace = prepared.trace
    assert trace["native_stop_reason"] == "teacher_attempt_ceiling_during_native_retry"
    assert teacher.calls == len(editor.requests) == len(attempts) == 4
    requests = [row for row in ledger.events if row["event"] == "model_request" and row["role"] == "teacher"]
    responses = [row for row in ledger.events if row["event"] == "model_response" and row["role"] == "teacher"]
    assert len(requests) == len(responses) == 4
    assert not any(row.get("attempt_id") == "teacher-0005" for row in ledger.events)
    assert solver.calls <= limits.solver_calls
    assert prepared.instructions == (matched.public_task_text(task_and_final[0]),)
    assert trace["native_budget_interruption"]["unfinished_proposal_discarded"] is True
    assert trace["native_result"]["val_aggregate_scores"] == [0.0]
    assert all(len(scores) == 54 for scores in trace["native_result"]["val_subscores"])
    assert matched.PreparedMethod.from_dict(prepared.to_dict()).prepared_hash == prepared.prepared_hash


def test_final_failures_are_wrong_even_when_uncertain_is_the_reference(task_and_final):
    task, final = task_and_final
    cases = [case for case in final if case.expected_label is None][:3]
    answers = iter(["uncertain", "gibberish", RuntimeError("fixture backend failure")])
    def emit(*_):
        value = next(answers)
        if isinstance(value, Exception):
            raise value
        return value
    session = matched.ModelSession(FakeModel(emit), matched.EventLedger(), "solver_final", 3)
    rows = matched.evaluate_prepared(task, matched.PreparedMethod.raw(task), cases, session)
    assert [row["correct"] for row in rows] == [True, False, False]
    assert [row["invalid"] for row in rows] == [False, True, True]
    assert [row["backend_failed"] for row in rows] == [False, False, True]
    assert session.calls == len(rows) == 3


def test_final_truncation_is_wrong_but_not_misclassified_as_a_backend_outage(task_and_final):
    task, final = task_and_final
    case = next(case for case in final if case.expected_label is None)

    def truncated(*_):
        raise ModelCallError("Local output reached its token cap; truncated responses are failures",
                             input_tokens=5, output_tokens=16)

    session = matched.ModelSession(FakeModel(truncated), matched.EventLedger(), "solver_final", 1)
    row = matched.evaluate_prepared(task, matched.PreparedMethod.raw(task), [case], session)[0]
    assert row["correct"] is False and row["invalid"] is True
    assert row["model_failed"] is True and row["backend_failed"] is False
    assert row["failure_kind"] == "truncation"
    assert row["output_tokens"] == 16 and session.backend_failures == 0


@pytest.mark.parametrize("problem", ["task_binding", "public_overlap", "insufficient_budget", "wrong_session"])
def test_final_validation_precedes_every_model_dispatch(task_and_final, problem):
    task, final = task_and_final
    prepared = matched.PreparedMethod.raw(task)
    model = FakeModel(answer_input)
    session = matched.ModelSession(model, matched.EventLedger(), "solver_final", 2)
    cases = final[:2]
    if problem == "task_binding":
        prepared = replace(prepared, public_task_hash="0" * 64)
    elif problem == "public_overlap":
        cases = [task.build_cases[0]]
    elif problem == "insufficient_budget":
        cases = final[:3]
    else:
        session.role = "solver"
    with pytest.raises((ValueError, matched.BudgetExceeded)):
        matched.evaluate_prepared(task, prepared, cases, session)
    assert session.calls == 0 and not model.requests


def test_prepare_data_produces_three_disjoint_27_case_splits_without_optimizer_oracle(tmp_path):
    runner.prepare_data(tmp_path, {"task_ids": [120]})
    public = json.loads((tmp_path / "public_tasks.json").read_text(encoding="utf-8"))
    final = json.loads((tmp_path / "sealed/final_cases.json").read_text(encoding="utf-8"))
    dto = matched.PublicTask.from_dict(public[0])
    sealed = [matched.PublicCase(**row) for row in final[dto.task_id]]
    assert len(dto.build_cases) == len(dto.audit_cases) == len(sealed) == 27
    assert len({case.input_json for case in (*dto.public_cases, *sealed)}) == 81
    assert all(set(row) == {"task_id", "spec", "build_cases", "audit_cases"} for row in public)
    assert not (tmp_path / "final_opening.json").exists()


def test_global_lock_blocks_final_access_until_all_artifacts_exist(tmp_path, monkeypatch):
    final_path = tmp_path / "sealed/final_cases.json"
    final_path.parent.mkdir()
    final_path.write_text('{"sealed":"FINAL_SENTINEL"}', encoding="utf-8")
    runner.write_json(tmp_path / "case_commitment.json", {"final_cases_sha256": runner.digest(final_path)})
    original = Path.read_text
    opened = []
    def monitored(path, *args, **kwargs):
        if path == final_path:
            opened.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", monitored)
    with pytest.raises(ValueError, match="global instruction lock"):
        runner.open_final_cases(tmp_path, 2)
    assert not opened
    first = tmp_path / "prepared/first.json"
    runner.write_json(first, {"instruction": "first"})
    with pytest.raises(ValueError, match="All task/model"):
        runner.lock_prepared(tmp_path, [first], 2)
    assert not (tmp_path / "prepared_lock.json").exists() and not opened
    second = tmp_path / "prepared/second.json"
    runner.write_json(second, {"instruction": "second"})
    lock = runner.lock_prepared(tmp_path, [first, second], 2)
    assert len(lock["artifacts"]) == 2
    assert runner.open_final_cases(tmp_path, 2) == {"sealed": "FINAL_SENTINEL"}
    assert opened == [final_path]
    proof = json.loads((tmp_path / "final_opening.json").read_text(encoding="utf-8"))
    assert proof["prepared_lock_sha256"] == runner.digest(tmp_path / "prepared_lock.json")


@pytest.mark.parametrize("tamper", ["instruction", "final", "escape", "count"])
def test_sealed_instruction_and_final_bytes_cannot_change_before_opening(tmp_path, tamper):
    prepared = tmp_path / "prepared/one.json"
    runner.write_json(prepared, {"instruction": "unchanged"})
    final = tmp_path / "sealed/final_cases.json"
    runner.write_json(final, {"sealed": [1]})
    runner.write_json(tmp_path / "case_commitment.json", {"final_cases_sha256": runner.digest(final)})
    runner.lock_prepared(tmp_path, [prepared], 1)
    if tamper == "instruction":
        prepared.write_text('{}', encoding="utf-8")
    elif tamper == "final":
        final.write_text('{}', encoding="utf-8")
    else:
        path = tmp_path / "prepared_lock.json"
        lock = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "escape":
            lock["artifacts"] = {"sealed/final_cases.json": runner.digest(final)}
        else:
            lock["expected_artifacts"] = 2
        path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.open_final_cases(tmp_path, 1)
    assert not (tmp_path / "final_opening.json").exists()
