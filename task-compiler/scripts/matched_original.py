"""Matched original DSL pipeline and native GEPA, with no hidden oracle input.

This module contains the algorithms and audited local model boundary.  The
separate experiment runner owns sampling, source freezing and final evaluation.
Optimization accepts only ``PublicTask``; final cases enter only
``evaluate_prepared`` after the runner has frozen every prepared method.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable


GEPA_VERSION = "0.1.4"
RETRIEVER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FAILURE_TEXT = "__MODEL_FAILURE__"
CANDIDATE_FAILURE_TEXT = "__CANDIDATE_REJECTED__"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False)
class PublicCase:
    """An immutable snapshot of one disclosed input/reference pair."""

    case_id: str
    input_json: str
    expected_label: str | None

    def __init__(self, case_id: str, input_record: dict, expected_label: str | None):
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Every case needs a nonempty identifier")
        if not isinstance(input_record, dict) or not input_record:
            raise ValueError("Every case needs a nonempty structured input")
        if expected_label is not None and (not isinstance(expected_label, str) or not expected_label):
            raise ValueError("References must be a declared label or None for uncertain")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "input_json", canonical_json(input_record))
        object.__setattr__(self, "expected_label", expected_label)

    @property
    def input_record(self) -> dict:
        return json.loads(self.input_json)

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "input_record": self.input_record, "expected_label": self.expected_label}


@dataclass(frozen=True, init=False)
class PublicTask:
    """Public DTO with an explicit field allowlist; no oracle/final parameters."""

    task_id: str
    spec_json: str
    build_cases: tuple[PublicCase, ...]
    audit_cases: tuple[PublicCase, ...]

    def __init__(self, task_id: str, spec: Any, build_cases: Iterable[PublicCase], audit_cases: Iterable[PublicCase]):
        from taskcompiler.schema import TaskSpec

        value = spec.to_dict() if isinstance(spec, TaskSpec) else copy.deepcopy(spec)
        allowed = {"task_name", "description", "labels", "requirements", "forbidden_shortcuts", "sources", "enums", "records"}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("Public specifications accept only the declared TaskSpec fields")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Task identifier must be nonempty")
        specification = TaskSpec.from_dict(value)
        if not specification.description.strip() or not specification.labels:
            raise ValueError("A public task needs a description and declared labels")
        if any(not isinstance(label, str) or label != label.lower() or not label for label in specification.labels):
            raise ValueError("Semantic labels must be nonempty lowercase strings")
        if len(set(specification.labels)) != len(specification.labels) or set(specification.labels) & {"uncertain", "unknown", "none", "abstain"}:
            raise ValueError("Declared labels must be distinct from abstention markers")
        builds, audits = tuple(build_cases), tuple(audit_cases)
        if not builds or not audits or any(not isinstance(case, PublicCase) for case in (*builds, *audits)):
            raise ValueError("Both public splits must contain PublicCase snapshots")
        identifiers = [case.case_id for case in (*builds, *audits)]
        inputs = [case.input_json for case in (*builds, *audits)]
        if len(set(identifiers)) != len(identifiers) or len(set(inputs)) != len(inputs):
            raise ValueError("Public build/audit identifiers and inputs must be disjoint and unique")
        schema = _input_schema(specification)
        for case in (*builds, *audits):
            if case.expected_label is not None and case.expected_label not in specification.labels:
                raise ValueError("A public reference is outside the task's declared labels")
            _validate_record(case.input_record, schema)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "spec_json", canonical_json(specification.to_dict()))
        object.__setattr__(self, "build_cases", builds)
        object.__setattr__(self, "audit_cases", audits)

    @property
    def spec(self):
        from taskcompiler.schema import TaskSpec
        return TaskSpec.from_dict(json.loads(self.spec_json))

    @property
    def public_cases(self) -> tuple[PublicCase, ...]:
        return self.build_cases + self.audit_cases

    @property
    def task_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "spec": json.loads(self.spec_json), "build_cases": [case.to_dict() for case in self.build_cases], "audit_cases": [case.to_dict() for case in self.audit_cases]}

    @classmethod
    def from_dict(cls, data: dict) -> "PublicTask":
        if not isinstance(data, dict) or set(data) != {"task_id", "spec", "build_cases", "audit_cases"}:
            raise ValueError("Optimizer DTO must contain only public task/specification/build/audit fields")
        return cls(data["task_id"], data["spec"], [PublicCase(**row) for row in data["build_cases"]],
                   [PublicCase(**row) for row in data["audit_cases"]])


def _validate_record(record: dict, schema: dict) -> None:
    if set(record) != set(schema):
        raise ValueError("Input roots differ from the public schema")
    for root, declared in schema.items():
        values = record[root]
        if not isinstance(values, dict) or set(values) != set(declared):
            raise ValueError("Input fields differ from the public schema")
        if any(value not in declared[name] for name, value in values.items()):
            raise ValueError("Input contains an undeclared enum value")


def _input_schema(specification) -> dict:
    enums = {"Boolean3": {"TRUE", "FALSE", "UNKNOWN"}, **{enum.name: set(enum.values) for enum in specification.enums}}
    return {record.name[:1].lower() + record.name[1:]: {field.name: enums[field.type_name] for field in record.fields}
            for record in specification.records}


@dataclass(frozen=True)
class Limits:
    solver_calls: int = 216
    teacher_calls: int = 4
    teacher_tokens: int = 2000
    max_instruction_words: int = 1800
    top_k: int = 6
    seed: int = 20261005
    semantic_repair_rounds: int = 3
    counterexample_patches: int = 3
    representation_rounds: int = 3
    reflection_minibatch_size: int = 3

    def __post_init__(self) -> None:
        for name in ("solver_calls", "teacher_calls", "teacher_tokens", "max_instruction_words", "top_k", "reflection_minibatch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("semantic_repair_rounds", "counterexample_patches", "representation_rounds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


class BackendFailure(RuntimeError):
    """A parent-model transport/backend failure stops the study, never retries."""


class BudgetExceeded(RuntimeError):
    """A strict attempted-call ceiling would be exceeded."""


class _NativeAbort(BaseException):
    """Escape GEPA's Exception-based proposal retry logic."""

    def __init__(self, error: Exception):
        self.error = error


class EventLedger:
    """Append/fsync the attempted request before model dispatch.

    There is no resume or overwrite path. Transport responses from Ollama are
    preserved as well as the normalized response and any reported failed usage.
    """

    def __init__(self, path: Path | str | None = None, *, context: dict | None = None,
                 on_event: Callable[[dict], None] | None = None):
        self.path = Path(path) if path is not None else None
        self.context = copy.deepcopy(context or {})
        self.events: list[dict] = []
        self.on_event = on_event
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8"):
                pass

    def append(self, event: str, **values: Any) -> dict:
        row = {"event_index": len(self.events), "event": event, "recorded_at_unix": time.time(), **self.context, **values}
        canonical_json(row)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        self.events.append(row)
        if self.on_event is not None:
            self.on_event(copy.deepcopy(row))
        return row

    def summary(self) -> dict:
        attempts = [row for row in self.events if row["event"] == "model_request"]
        terminal = [row for row in self.events if row["event"] in {"model_response", "model_failure"}]
        roles = sorted({row["role"] for row in attempts})
        return {"attempted_calls": len(attempts), "completed_calls": sum(row["event"] == "model_response" for row in terminal),
                "failed_calls": sum(row["event"] == "model_failure" for row in terminal),
                "unfinished_calls": len(attempts) - len(terminal),
                "roles": {role: {"attempted_calls": sum(row["role"] == role for row in attempts),
                                 "failed_calls": sum(row["role"] == role and row["event"] == "model_failure" for row in terminal),
                                 "input_tokens_known": sum(row.get("input_tokens") or 0 for row in terminal if row["role"] == role),
                                 "output_tokens_known": sum(row.get("output_tokens") or 0 for row in terminal if row["role"] == role),
                                 "unknown_input_usage_calls": sum(row.get("input_tokens") is None for row in terminal if row["role"] == role),
                                 "unknown_output_usage_calls": sum(row.get("output_tokens") is None for row in terminal if row["role"] == role),
                                 "latency_seconds": sum(row.get("latency_seconds", 0) for row in terminal if row["role"] == role)} for role in roles}}


class ModelSession:
    """One model role/cell with strict attempted-call accounting."""

    def __init__(self, model: Any, ledger: EventLedger, role: str, limit: int):
        if not role or type(limit) is not int or limit < 1:
            raise ValueError("A model session needs a role and positive attempted-call limit")
        identity = getattr(model, "identity", {})
        if identity.get("mode") != "fixture" and not identity.get("local_only"):
            raise ValueError("The matched runner requires an explicitly local model")
        self.model, self.ledger, self.role, self.limit = model, ledger, role, limit
        self.attempts = 0
        self.last_failure_kind: str | None = None
        self.identity = copy.deepcopy(identity)

    @property
    def remaining(self) -> int:
        return self.limit - self.attempts

    @property
    def calls(self) -> int:
        return self.attempts

    @property
    def backend_failures(self) -> int:
        return sum(row["event"] == "model_failure" and row.get("role") == self.role and row.get("failure_kind") == "backend"
                   for row in self.ledger.events)

    def reserve_batch(self, size: int) -> None:
        if type(size) is not int or size < 0 or size > self.remaining:
            raise BudgetExceeded(f"{self.role}: cannot reserve a complete {size}-case batch; {self.remaining} attempts remain")

    def generate(self, system: str, prompt: str, schema: dict | None = None, *, metadata: dict | None = None,
                 seed: int | None = None, failure_is_invalid: bool = True):
        from taskcompiler.procedural.experiment import Response

        self.reserve_batch(1)
        self.attempts += 1
        self.last_failure_kind = None
        attempt_id = f"{self.role}-{self.attempts:04d}"
        identity = copy.deepcopy(self.identity)
        if seed is not None:
            identity["seed"] = seed
        self.ledger.append("model_request", role=self.role, attempt_id=attempt_id, model_identity=identity,
                           system=system, prompt=prompt, output_schema=schema or {}, metadata=metadata or {})
        old_seed = getattr(self.model, "seed", None)
        old_request = getattr(self.model, "_request", None)
        patched_request = old_request is not None
        if seed is not None and hasattr(self.model, "seed"):
            self.model.seed = seed
        if patched_request:
            def audited_request(suffix: str, body: dict | None = None):
                self.ledger.append("transport_request", role=self.role, attempt_id=attempt_id, suffix=suffix, body=body)
                payload = old_request(suffix, body)
                self.ledger.append("transport_response", role=self.role, attempt_id=attempt_id, payload=payload)
                return payload
            self.model._request = audited_request
        started = time.perf_counter()
        try:
            response = self.model.generate(system, prompt, schema or {})
            if not isinstance(response.text, str):
                raise BackendFailure("Local model returned a non-text response")
        except Exception as exc:
            elapsed = time.perf_counter() - started
            payloads = [row.get("payload", {}) for row in self.ledger.events
                        if row["event"] == "transport_response" and row.get("attempt_id") == attempt_id]
            truncated = any(payload.get("done_reason") == "length" for payload in payloads)
            truncated = truncated or str(exc).startswith("Local output reached its token cap; truncated responses are failures")
            self.last_failure_kind = "truncation" if truncated else "backend"
            self.ledger.append("model_failure", role=self.role, attempt_id=attempt_id,
                               error_type=type(exc).__name__, error=str(exc), failure_kind=self.last_failure_kind,
                               input_tokens=getattr(exc, "input_tokens", None),
                               output_tokens=getattr(exc, "output_tokens", None), latency_seconds=elapsed)
            if not failure_is_invalid:
                raise BackendFailure(f"{self.role} backend failed on {attempt_id}: {type(exc).__name__}: {exc}") from exc
            return Response(MODEL_FAILURE_TEXT, getattr(exc, "input_tokens", None), getattr(exc, "output_tokens", None), elapsed)
        finally:
            if seed is not None and hasattr(self.model, "seed"):
                self.model.seed = old_seed
            if patched_request:
                self.model._request = old_request
        self.ledger.append("model_response", role=self.role, attempt_id=attempt_id, text=response.text,
                           input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                           latency_seconds=response.latency_seconds)
        return response


class LocalTeacher:
    """Duck-typed TeacherClient backed by the charged local editor session."""

    def __init__(self, session: ModelSession, limits: Limits):
        self.session, self.limits = session, limits
        self.identity = session.identity

    def call(self, system: str, prompt: str, max_tokens: int = 2000, response_format: dict | None = None,
             seed: int | None = None):
        from taskcompiler.teacher import TeacherResponse

        if max_tokens != self.limits.teacher_tokens:
            raise ValueError("Teacher token limit differs from the frozen allowance")
        schema = (response_format or {}).get("json_schema", {}).get("schema", {})
        response = self.session.generate(system, prompt, schema, seed=seed, failure_is_invalid=False)
        return TeacherResponse(response.text, response.input_tokens or 0, response.output_tokens or 0,
                               response.latency_seconds, str(self.identity.get("model", "fixture")), "local_ollama")


def public_task_text(task: PublicTask) -> str:
    spec = json.loads(task.spec_json)
    parts = [f"Task: {spec['task_name']}", spec["description"], "Declared labels: " + ", ".join(spec["labels"]),
             "Public input schema: " + canonical_json({"enums": spec["enums"], "records": spec["records"]})]
    for field, title in (("requirements", "Requirements"), ("forbidden_shortcuts", "Forbidden shortcuts"), ("sources", "Public source identifiers")):
        if spec[field]:
            parts.append(title + ":\n" + "\n".join(spec[field]))
    return "\n\n".join(parts)


def solver_messages(task: PublicTask, instruction: str, input_record: dict) -> tuple[str, str]:
    """The same immutable wrapper, input serialization and parser in every arm."""
    spec = json.loads(task.spec_json)
    system = (f"Perform task {spec['task_name']}.\n"
              f"Public input schema: {canonical_json({'enums': spec['enums'], 'records': spec['records']})}\n"
              f"Allowed labels: {', '.join(spec['labels'])}, uncertain.\n\n"
              f"Task instructions:\n{instruction}\n\n"
              "Output contract: return exactly one declared label, or uncertain if the task does not decide. "
              "Do not include reasoning or formatting.")
    return system, "Input: " + canonical_json(input_record)


def parse_label(text: str, task: PublicTask) -> str | None:
    from taskcompiler.translate import normalise_student_label
    if text in {MODEL_FAILURE_TEXT, CANDIDATE_FAILURE_TEXT}:
        return "__INVALID__:model_or_candidate_failure"
    return normalise_student_label(text, set(task.spec.labels))


def word_count(instruction: str) -> int:
    return len(instruction.split())


def _guard_instruction(instruction: str, limits: Limits) -> None:
    if not isinstance(instruction, str) or not instruction.strip() or word_count(instruction) > limits.max_instruction_words:
        raise ValueError("Instruction is empty or exceeds the frozen whitespace-word allowance")


class CanonicalStudent:
    """Adapt legacy Full probes to the common solver messages.

    The original probe renderer passes a generic deployment description, rule
    lines and output contract. Only its rule lines are candidate material; the
    same immutable schema/output wrapper encloses GEPA's text as well.
    """

    def __init__(self, task: PublicTask, session: ModelSession, limits: Limits):
        self.task, self.session, self.limits = task, session, limits

    def ask(self, prompt: str, *, system: str):
        from taskcompiler.student import StudentResponse

        if not prompt.startswith("Input: ") or "\n\nRules:\n" not in system or "\n\nOutput contract:" not in system:
            raise ValueError("Unexpected legacy probe rendering")
        instruction = system.split("\n\nRules:\n", 1)[1].rsplit("\n\nOutput contract:", 1)[0]
        instruction = "\n".join(line[2:] if line.startswith("- ") else line for line in instruction.splitlines())
        _guard_instruction(instruction, self.limits)
        record = ast.literal_eval(prompt.removeprefix("Input: "))
        if not isinstance(record, dict):
            raise ValueError("Legacy probe input must be a structured record")
        common_system, common_prompt = solver_messages(self.task, instruction, record)
        response = self.session.generate(common_system, common_prompt, {}, metadata={"phase": "public_behavioral_selection", "instruction_hash": content_hash(instruction)})
        return StudentResponse(response.text, response.input_tokens or 0, response.output_tokens or 0, response.latency_seconds)


@dataclass(frozen=True)
class PreparedMethod:
    method: str
    task_id: str
    public_task_hash: str
    instructions: tuple[str, ...]
    routed: bool
    status: str
    trace_json: str

    @property
    def trace(self) -> dict:
        return json.loads(self.trace_json)

    @property
    def prepared_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict:
        dsl = self.trace.get("dsl", {})
        return {"method": self.method, "task_id": self.task_id, "public_task_hash": self.public_task_hash,
                "instructions": list(self.instructions), "routed": self.routed, "status": self.status, "trace": self.trace,
                "dsl_source": dsl.get("final_source") if dsl.get("public_verification_passed") else None}

    @classmethod
    def from_dict(cls, data: dict) -> "PreparedMethod":
        allowed = {"method", "task_id", "public_task_hash", "instructions", "routed", "status", "trace", "dsl_source"}
        if not isinstance(data, dict) or set(data) != allowed:
            raise ValueError("Malformed serialized prepared method")
        row = cls(data["method"], data["task_id"], data["public_task_hash"], tuple(data["instructions"]),
                  data["routed"], data["status"], canonical_json(data["trace"]))
        if row.to_dict() != data:
            raise ValueError("Prepared-method fields disagree with their verification trace")
        return row

    @classmethod
    def raw(cls, task: PublicTask, limits: Limits | None = None) -> "PreparedMethod":
        return prepare_raw(task, limits)

    def instructions_for(self, record: dict, router: Any | None = None) -> tuple[str, dict | None]:
        if not self.routed:
            return "\n".join(self.instructions), None
        if router is None:
            raise ValueError("Routed Full deployment requires its frozen local router")
        decision = router.route(record, list(self.instructions), top_k=self.trace["limits"]["top_k"], label_encoder=None)
        return "\n".join(decision.instructions), decision.certificate


def _prepared(task: PublicTask, method: str, instructions: list[str], *, routed: bool, status: str, trace: dict) -> PreparedMethod:
    # GEPA result dictionaries use integer validation ids. JSON serializes these
    # keys as strings; normalize once so a persisted/reloaded artifact has the
    # same canonical bytes and prepared hash as its in-memory counterpart.
    trace_json = canonical_json(json.loads(canonical_json(trace)))
    return PreparedMethod(method, task.task_id, task.task_hash, tuple(instructions), routed, status, trace_json)


def prepare_raw(task: PublicTask, limits: Limits | None = None) -> PreparedMethod:
    limits = limits or Limits()
    instruction = public_task_text(task)
    _guard_instruction(instruction, limits)
    return _prepared(task, "raw", [instruction], routed=False, status="prepared", trace={"limits": asdict(limits), "optimizer_inputs": "public_task_description_schema_only"})


def _raw_fallback(task: PublicTask, limits: Limits, trace: dict, stage: str, message: str) -> PreparedMethod:
    trace["failure"] = {"stage": stage, "message": message, "fallback": "raw_task"}
    trace["limits"] = asdict(limits)
    instruction = public_task_text(task)
    _guard_instruction(instruction, limits)
    return _prepared(task, "full_dsl", [instruction], routed=False, status="raw_fallback", trace=trace)


def compile_full(task: PublicTask, solver: ModelSession, teacher: ModelSession, router: Any, limits: Limits | None = None) -> PreparedMethod:
    """Actual teacher-generated DSL, public verification/repair, certified search."""
    from taskcompiler.dsl.parser import DslSyntaxError, render_program_source
    from taskcompiler.spec_compiler import compile_specification
    from taskcompiler.translate import behavioral_compile, build_obligation_probes, render_program
    from taskcompiler.verify import RuleCase

    limits = limits or Limits()
    _guard_instruction(public_task_text(task), limits)
    if teacher.limit != limits.teacher_calls or solver.limit != limits.solver_calls:
        raise ValueError("Session ceilings differ from the frozen method allowance")
    trace: dict = {"limits": asdict(limits), "initial_source_supplied": False, "oracle_program_access": False,
                   "final_access_during_optimization": False, "public_build_case_ids": [case.case_id for case in task.build_cases],
                   "public_audit_refinement_case_ids": [case.case_id for case in task.audit_cases],
                   "teacher_raw_text": [], "router_identity": copy.deepcopy(getattr(router, "identity", {}))}
    builds = [RuleCase(case.input_record, case.expected_label) for case in task.build_cases]
    audits = [RuleCase(case.input_record, case.expected_label) for case in task.audit_cases]
    try:
        compilation = compile_specification(LocalTeacher(teacher, limits), task.spec, builds, audits,
                                            max_repair_rounds=limits.semantic_repair_rounds, max_syntax_repair_rounds=0,
                                            max_tokens=limits.teacher_tokens, source_id=task.spec.sources[0] if task.spec.sources else task.task_id,
                                            repair=True, refine_with_audit=True, max_counterexample_patches=limits.counterexample_patches,
                                            structured_output=True, seed=limits.seed,
                                            on_teacher_text=lambda phase, text: trace["teacher_raw_text"].append({"phase": phase, "text": text}))
    except DslSyntaxError as exc:
        trace["failed_dsl_source"] = getattr(exc, "failed_source", "")
        trace["teacher_calls"] = teacher.attempts
        return _raw_fallback(task, limits, trace, "teacher_syntax", str(exc))
    program = compilation.final_program
    trace["dsl"] = {"initial_source": render_program_source(compilation.initial_program), "final_source": render_program_source(program),
                    "initial_build": asdict(compilation.initial_build), "final_build": asdict(compilation.final_build),
                    "final_audit": asdict(compilation.final_audit), "repairs": [asdict(row) for row in compilation.repairs],
                    "teacher_calls": compilation.teacher_calls}
    passed = compilation.final_build.passed == compilation.final_build.total and compilation.final_audit.passed == compilation.final_audit.total
    trace["dsl"]["public_verification_passed"] = passed
    if not passed:
        return _raw_fallback(task, limits, trace, "teacher_validation_gate", "Teacher-generated program does not pass every disclosed build/audit reference")
    canonical = render_program(program)
    if word_count("\n".join(canonical)) > limits.max_instruction_words:
        return _raw_fallback(task, limits, trace, "canonical_instruction_budget", "Initial generated-program rendering exceeds the shared word allowance")
    # Trace obligations from the teacher program only.  Disclosed independent
    # reference labels replace the program labels; no oracle AST is available.
    _, probes = build_obligation_probes(program, [case.input_record for case in task.build_cases], f"{task.task_id}-build", causal_only=True)
    for probe, case in zip(probes, task.build_cases, strict=True):
        probe.probe_id = case.case_id
        probe.expected_label = case.expected_label
    trace["behavioral_probe_ids"] = [probe.probe_id for probe in probes]
    trace["behavioral_obligations"] = {probe.probe_id: list(probe.obligations) for probe in probes}
    student = CanonicalStudent(task, solver, limits)
    def routed(candidate, probe):
        return router.route(probe.input_record, candidate.render(), top_k=limits.top_k, label_encoder=None).instructions
    try:
        behavioral = behavioral_compile(student, "Follow the task instructions using its declared input schema.", program, probes, [],
                                        max_rounds=limits.representation_rounds, token_budget=limits.max_instruction_words,
                                        mode="certified_search", max_student_calls=min(limits.solver_calls, len(probes) * (1 + limits.representation_rounds)),
                                        max_protected_regressions=0, render_for_probe=routed, label_decoder=None)
    except RuntimeError as exc:
        if str(exc) != "no certified representation fit the compilation budget":
            raise
        return _raw_fallback(task, limits, trace, "certified_representation_budget", str(exc))
    instructions = behavioral.bundle.render()
    _guard_instruction("\n".join(instructions), limits)
    trace["behavioral"] = {"representation_kind": behavioral.bundle.representation_kind,
                           "semantic_certificate": behavioral.bundle.semantic_certificate,
                           "rounds": [asdict(row) for row in behavioral.rounds],
                           "initial_build_results": [asdict(row) for row in behavioral.initial_build_results],
                           "selected_build_results": [asdict(row) for row in behavioral.build_results],
                           "student_calls": behavioral.student_calls, "build_accuracy": behavioral.build_pass_rate}
    return _prepared(task, "full_dsl", instructions, routed=True, status="prepared", trace=trace)


def compile_gepa(task: PublicTask, solver: ModelSession, teacher: ModelSession, limits: Limits | None = None) -> PreparedMethod:
    """Official native GEPA's free natural-language system-prompt search.

    A stopper reserves a complete parent minibatch, proposal minibatch and
    public full evaluation before each native iteration.  Candidate guards
    return explicit zero-scored invalid outputs without model dispatch.
    """
    if version("gepa") != GEPA_VERSION:
        raise ValueError(f"Expected official gepa=={GEPA_VERSION}")
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter, EvaluationResult
    from gepa.core.adapter import EvaluationBatch

    limits = limits or Limits()
    if teacher.limit != limits.teacher_calls or solver.limit != limits.solver_calls:
        raise ValueError("Session ceilings differ from the frozen method allowance")
    seed_prompt = public_task_text(task)
    _guard_instruction(seed_prompt, limits)
    rows = [{"input": case.input_json, "answer": case.expected_label, "additional_context": {"case_id": case.case_id}} for case in task.public_cases]
    if limits.solver_calls < len(rows):
        raise ValueError("GEPA allowance must cover its initial complete public evaluation")
    history: list[dict] = []
    guards: list[dict] = []
    logs: list[str] = []
    budget_interruption: dict | None = None

    def task_model(messages):
        if [message["role"] for message in messages] != ["system", "user"]:
            raise ValueError("Unexpected native GEPA solver messages")
        instruction = messages[0]["content"]
        record = json.loads(messages[1]["content"])
        system, prompt = solver_messages(task, instruction, record)
        try:
            return solver.generate(system, prompt, {}, metadata={"phase": "gepa_public_evaluation", "instruction_hash": content_hash(instruction)}).text
        except BudgetExceeded as exc:
            raise _NativeAbort(exc) from exc

    def evaluator(data, response):
        parsed = parse_label(response, task)
        correct = parsed == data["answer"]
        expected = data["answer"] if data["answer"] is not None else "uncertain"
        feedback = f"{'Correct' if correct else 'Incorrect'}. Parsed semantic label: {parsed!r}. Public reference label: {expected}."
        return EvaluationResult(float(correct), feedback, None)

    class AuditedDefaultAdapter(DefaultAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            if set(candidate) != {"system_prompt"}:
                raise ValueError("Native GEPA candidate must contain exactly system_prompt")
            instruction = candidate["system_prompt"]
            candidate_hash = content_hash(candidate)
            history.append({"event": "evaluation_requested", "candidate_hash": candidate_hash, "candidate": candidate,
                            "case_ids": [row["additional_context"]["case_id"] for row in batch], "capture_traces": capture_traces})
            try:
                _guard_instruction(instruction, limits)
            except ValueError as exc:
                guards.append({"candidate_hash": candidate_hash, "candidate": candidate, "reason": str(exc), "batch_size": len(batch)})
                outputs = [{"full_assistant_response": CANDIDATE_FAILURE_TEXT} for _ in batch]
                trajectories = [{"data": row, "full_assistant_response": CANDIDATE_FAILURE_TEXT, "feedback": "Candidate rejected by the shared instruction word allowance; this case scores zero."} for row in batch] if capture_traces else None
                return EvaluationBatch(outputs=outputs, scores=[0.0] * len(batch), trajectories=trajectories)
            try:
                solver.reserve_batch(len(batch))
            except BudgetExceeded as exc:
                raise _NativeAbort(exc) from exc
            result = super().evaluate(batch, candidate, capture_traces)
            history.append({"event": "evaluation_completed", "candidate_hash": candidate_hash, "case_ids": [row["additional_context"]["case_id"] for row in batch],
                            "capture_traces": capture_traces, "scores": list(result.scores), "outputs": result.outputs})
            return result

    def reflection_model(prompt):
        if isinstance(prompt, str):
            text = prompt
        else:
            text = "\n\n".join(f"{row['role']}:\n{row['content']}" for row in prompt)
        try:
            response = teacher.generate("Improve reusable instructions for the public task from its public evaluation traces. "
                                        "Follow the native optimizer's requested proposal format. Public examples may be used.",
                                        text, {}, metadata={"phase": "gepa_reflection"}, seed=limits.seed, failure_is_invalid=False)
        except (BackendFailure, BudgetExceeded) as exc:
            raise _NativeAbort(exc) from exc
        history.append({"event": "reflection_completed", "text": response.text, "teacher_attempt": teacher.attempts})
        return response.text

    class IterationStopper:
        reason: str | None = None
        state: Any | None = None
        def __call__(self, state):
            # This is the official mutable search state.  Retain a reference
            # so a proposal-parser retry hitting a strict role budget can stop
            # without replacing native selection or discarding scored parents.
            self.state = state
            reservation = len(rows) + 2 * min(limits.reflection_minibatch_size, len(rows))
            metric_remaining = limits.solver_calls - state.total_num_evals
            if teacher.remaining < 1:
                self.reason = "teacher_attempt_ceiling"
            elif solver.remaining < reservation or metric_remaining < reservation:
                self.reason = "insufficient_allowance_for_complete_native_iteration"
            else:
                return False
            return True

    class Logger:
        def log(self, message):
            logs.append(str(message))

    stopper = IterationStopper()
    adapter = AuditedDefaultAdapter(task_model, evaluator)
    try:
        result = gepa.optimize(seed_candidate={"system_prompt": seed_prompt}, trainset=rows, valset=rows,
                               adapter=adapter, reflection_lm=reflection_model,
                               reflection_minibatch_size=limits.reflection_minibatch_size,
                               max_metric_calls=limits.solver_calls, stop_callbacks=stopper,
                               candidate_selection_strategy="pareto", use_merge=False,
                               val_evaluation_policy="full_eval", cache_evaluation=False,
                               track_best_outputs=False, display_progress_bar=False,
                               logger=Logger(), seed=limits.seed, raise_on_exception=True)
    except _NativeAbort as abort:
        if not isinstance(abort.error, BudgetExceeded):
            teacher.ledger.append("native_aborted", reason_type=type(abort.error).__name__, reason=str(abort.error),
                                  history=history, guards=guards, logs=logs)
            raise abort.error
        # Native GEPA can retry proposal parsing inside one iteration.  Its
        # iteration-level stopper does not run between those retries. A denied
        # next call is therefore a clean budget stop, while backend failures
        # above still interrupt the whole study. Use the package's own result
        # conversion and winner selection; never manufacture a candidate.
        from gepa.core.result import GEPAResult
        state = stopper.state
        if state is None or not state.is_consistent():
            raise RuntimeError("Budget stop has no consistent official GEPA state") from abort.error
        if not state.program_candidates or len(state.program_candidates) != len(state.prog_candidate_val_subscores):
            raise RuntimeError("Budget stop has an incomplete native candidate state") from abort.error
        expected_ids = set(range(len(rows)))
        if any(set(scores) != expected_ids for scores in state.prog_candidate_val_subscores):
            raise RuntimeError("Budget stop cannot select a partially evaluated native candidate") from abort.error
        result = GEPAResult.from_state(state, seed=limits.seed)
        stopper.reason = "teacher_attempt_ceiling_during_native_retry" if teacher.remaining == 0 else "solver_attempt_ceiling_during_native_iteration"
        budget_interruption = {"reason": str(abort.error), "official_result_conversion": "GEPAResult.from_state",
                               "fully_scored_candidates": result.num_candidates, "native_metric_calls": result.total_metric_calls,
                               "teacher_attempts": teacher.attempts, "solver_attempts": solver.attempts,
                               "unfinished_proposal_discarded": True}
        teacher.ledger.append("native_budget_stopped", **budget_interruption, history=history, guards=guards, logs=logs)
    selected = result.best_candidate
    if not isinstance(selected, dict) or set(selected) != {"system_prompt"}:
        raise RuntimeError("Native GEPA returned a malformed candidate")
    instruction = selected["system_prompt"]
    _guard_instruction(instruction, limits)
    trace = {"limits": asdict(limits), "implementation": "official_gepa_default_adapter", "gepa_version": GEPA_VERSION,
             "oracle_program_access": False, "final_access_during_optimization": False,
             "public_train_case_ids": [case.case_id for case in task.public_cases],
             "public_validation_case_ids": [case.case_id for case in task.public_cases],
             "validation_reused_for_optimization": True, "candidate_word_guard": guards,
             "native_history": history, "native_logs": logs, "native_stop_reason": stopper.reason,
             "native_budget_interruption": budget_interruption,
             "metric_calls": result.total_metric_calls, "solver_attempts": solver.attempts, "teacher_attempts": teacher.attempts,
             "native_result": result.to_dict(), "best_candidate_index": result.best_idx,
             "candidate_count": result.num_candidates, "edited_candidate_count": sum(candidate != {"system_prompt": seed_prompt} for candidate in result.candidates),
             "merge": False, "selection": "pareto", "reflection_minibatch_size": limits.reflection_minibatch_size,
             "whole_iteration_reservation": len(rows) + 2 * min(limits.reflection_minibatch_size, len(rows))}
    return _prepared(task, "native_gepa", [instruction], routed=False, status="prepared", trace=trace)


def evaluate_prepared(task: PublicTask, prepared: PreparedMethod, final_cases: Iterable[PublicCase], solver: ModelSession,
                      router: Any | None = None) -> list[dict]:
    """Evaluate a frozen artifact; final inputs never enter either optimizer."""
    if prepared.task_id != task.task_id or prepared.public_task_hash != task.task_hash:
        raise ValueError("Prepared method does not match the immutable public task")
    if solver.role != "solver_final":
        raise ValueError("Final evaluation needs a separate final-only model session")
    cases = tuple(final_cases)
    public_ids = {case.case_id for case in task.public_cases}
    public_inputs = {case.input_json for case in task.public_cases}
    if not cases or len({case.case_id for case in cases}) != len(cases) or len({case.input_json for case in cases}) != len(cases):
        raise ValueError("Final cases must be nonempty and unique")
    if any(case.case_id in public_ids or case.input_json in public_inputs for case in cases):
        raise ValueError("Final cases overlap disclosed optimization inputs")
    schema = _input_schema(task.spec)
    for case in cases:
        _validate_record(case.input_record, schema)
        if case.expected_label is not None and case.expected_label not in task.spec.labels:
            raise ValueError("Final reference is outside the declared task labels")
    solver.reserve_batch(len(cases))
    rows: list[dict] = []
    for case in cases:
        instruction, certificate = prepared.instructions_for(case.input_record, router)
        system, prompt = solver_messages(task, instruction, case.input_record)
        response = solver.generate(system, prompt, {}, metadata={"phase": "final_evaluation", "case_id": case.case_id,
                                                               "prepared_hash": prepared.prepared_hash, "instruction_hash": content_hash(instruction)})
        parsed = parse_label(response.text, task)
        rows.append({"task_id": task.task_id, "method": prepared.method, "case_id": case.case_id,
                     "input_record": case.input_record, "expected_label": case.expected_label, "student_text": response.text,
                     "predicted_label": parsed, "correct": parsed == case.expected_label,
                     "invalid": isinstance(parsed, str) and parsed.startswith("__INVALID__"),
                     "model_failed": solver.last_failure_kind is not None,
                     "backend_failed": solver.last_failure_kind == "backend", "failure_kind": solver.last_failure_kind,
                     "prepared_hash": prepared.prepared_hash,
                     "system": system, "prompt": prompt, "routing_certificate": certificate,
                     "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                     "latency_seconds": response.latency_seconds})
    return rows


def make_router(snapshot: str | Path):
    """Load the already installed pinned local MiniLM snapshot; no download."""
    from taskcompiler.route import CertifiedInstructionRouter
    location = Path(snapshot).resolve(strict=True)
    if location.name != RETRIEVER_REVISION or not (location / "config.json").is_file():
        raise ValueError("Expected the exact installed MiniLM snapshot revision")
    router = CertifiedInstructionRouter(str(location), retriever_revision=RETRIEVER_REVISION)
    router.identity.update({"model": "sentence-transformers/all-MiniLM-L6-v2", "revision": RETRIEVER_REVISION, "local_snapshot": str(location)})
    return router
