"""Deterministic compositional policy tasks for compiler evaluation."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from taskcompiler.config import APE_OPTIMIZER, GEPA_OPTIMIZER, RunConfig
from taskcompiler.dsl.ast import (
    And,
    Comparison,
    Effect,
    FieldRef,
    Literal,
    Not,
    Or,
    PredicateCall,
    PredicateDecl,
    Program,
    Rule,
)
from taskcompiler.dsl.interpreter import run
from taskcompiler.dsl.parser import (
    DslSyntaxError,
    render_program_source,
    validate_program,
)
from taskcompiler.lineage import validate_lineage_config
from taskcompiler.metrics import RunRecorder
from taskcompiler.route import CertifiedInstructionRouter
from taskcompiler.schema import EnumType, FieldDecl, RecordType, TaskSpec
from taskcompiler.spec_compiler import compile_specification
from taskcompiler.statistics import wilson_accuracy
from taskcompiler.student import StudentModel
from taskcompiler.teacher import TeacherClient
from taskcompiler.translate import (
    FreeTextBundle,
    InstructionBundle,
    InstructionEntry,
    behavioral_compile,
    build_obligation_probes,
    generic_test_driven_compile,
    normalise_student_label,
    probe_student,
    render_program,
)
from taskcompiler.verify import RuleCase


SUITE_VERSION = "compositional-policies-v2"
TEACHER_ONE_SHOT = "teacher_one_shot"
TEACHER_REPAIRED_RENDER = "teacher_repaired_render"
TEACHER_VALIDATED = "teacher_validated"
TEACHER_VALIDATED_ROUTING = "teacher_validated_routing"
FROZEN_BUNDLE = "frozen_bundle"
FROZEN_BUNDLE_ROUTING = "frozen_bundle_routing"
TEACHER_COMPILED_BASELINES = {
    TEACHER_ONE_SHOT,
    TEACHER_REPAIRED_RENDER,
    TEACHER_VALIDATED,
    TEACHER_VALIDATED_ROUTING,
}


def _passes_teacher_validation(
    compilation,
    config: RunConfig,
) -> bool:
    """Check the frozen source-derived validation thresholds."""
    return (
        compilation.final_build.pass_rate
        >= config.teacher_build_pass_threshold
        and compilation.final_audit.pass_rate
        >= config.teacher_audit_pass_threshold
    )


def _routing_summary(
    predictions: list[dict],
    *,
    top_k: int,
    opaque_labels: bool,
) -> dict:
    """Summarize examples that used a certified route."""
    routed = [
        row
        for row in predictions
        if row.get("routing_certificate") is not None
    ]
    if not routed:
        return {"enabled": False}
    return {
        "enabled": True,
        "top_k": top_k,
        "opaque_labels": opaque_labels,
        "routed_examples": len(routed),
        "fallback_examples": len(predictions) - len(routed),
        "coverage_rate": sum(
            bool(row["routing_certificate"]["coverage_closed"])
            for row in routed
        )
        / len(routed),
        "closure_add_rate": sum(
            bool(
                row["routing_certificate"]["closure_added_rule_indices"]
            )
            for row in routed
        )
        / len(routed),
        "mean_instruction_count": sum(
            row["routed_instruction_count"] for row in routed
        )
        / len(routed),
    }


TASK_SPLITS = {
    "development": range(0, 40),
    "validation": range(40, 80),
    "final": range(80, 120),
    "baseline_pilot": range(120, 122),
    "baseline_confirmation": range(122, 130),
    # Keeps post-fix confirmation separate from the eight opened tasks.
    "validation_confirmation": (
        41,
        60,
        47,
        45,
        50,
        53,
        54,
        77,
        62,
        79,
        63,
        57,
        48,
        75,
        43,
        73,
        49,
        44,
        76,
        46,
        68,
        58,
        51,
        70,
        59,
        65,
        71,
        72,
        56,
        42,
        66,
        64,
    ),
}


@dataclass(frozen=True)
class PolicyTask:
    task_id: str
    suite_split: str
    task_spec: TaskSpec
    oracle_program: Program
    build_cases: list[RuleCase]
    audit_cases: list[RuleCase]
    final_cases: list[RuleCase]


def _eq(field: str, value: str) -> Comparison:
    return Comparison(FieldRef(field), "==", Literal(value))


def _in(field: str, values: list[str]) -> Comparison:
    return Comparison(FieldRef(field), "in", [Literal(value) for value in values])


def _prefer(label: str, priority: int) -> Effect:
    return Effect("prefer", label, priority)


def _task_split(task_number: int) -> str:
    for name, numbers in TASK_SPLITS.items():
        if task_number in numbers:
            return name
    raise ValueError(f"task number is outside the locked suite: {task_number}")


def _select_cases(
    program: Program,
    records: list[dict],
    *,
    seed: int,
) -> tuple[list[RuleCase], list[RuleCase], list[RuleCase]]:
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)

    def coverage(record: dict) -> set[str]:
        trace = run(program, record)
        return {
            *(f"rule:{name}" for name in trace.fired_rules),
            f"output:{trace.output_label or 'uncertain'}",
        }

    required = {
        *(f"rule:{rule.name}" for rule in program.rules),
        "output:uncertain",
        *(f"output:{label}" for label in {
            effect.label
            for rule in program.rules
            for effect in rule.effects
            if effect.label
        }),
    }

    def select(source: list[dict], size: int) -> tuple[list[dict], list[dict]]:
        chosen: list[dict] = []
        uncovered = set(required)
        remaining = list(source)
        while uncovered and remaining:
            best = max(
                remaining,
                key=lambda record: len(coverage(record) & uncovered),
            )
            if not coverage(best) & uncovered:
                break
            chosen.append(best)
            remaining.remove(best)
            uncovered -= coverage(best)
        if uncovered:
            raise RuntimeError(
                f"policy split cannot cover obligations: {sorted(uncovered)}"
            )
        needed = size - len(chosen)
        chosen.extend(remaining[:needed])
        return chosen, remaining[needed:]

    build_records, pool = select(pool, 27)
    audit_records, final_records = select(pool, 27)

    def materialize(items: list[dict]) -> list[RuleCase]:
        return [
            RuleCase(copy.deepcopy(record), run(program, record).output_label)
            for record in items
        ]

    return (
        materialize(build_records),
        materialize(audit_records),
        materialize(final_records),
    )


def make_policy_task(task_number: int) -> PolicyTask:
    """Create one locked policy task.

    Args:
        task_number: Stable task number from 0 through 129.
    """
    suite_split = _task_split(task_number)
    rng = random.Random(f"{SUITE_VERSION}:{task_number}")
    roles = rng.sample(["admin", "analyst", "guest", "auditor", "operator"], 3)
    actions = rng.sample(["inspect", "change", "export", "delete", "approve"], 3)
    contexts = rng.sample(["internal", "partner", "external", "sandbox"], 3)
    labels = rng.sample(["allow", "deny", "review"], 3)
    allow, deny, review = labels

    family = task_number % 4
    if family == 0:
        dangerous_expr = Or(
            _eq("event.risk", "TRUE"),
            _eq("event.action", actions[2]),
        )
        trusted_expr = And(
            _eq("event.role", roles[0]),
            Not(_eq("event.context", contexts[2])),
        )
        dangerous_text = (
            f"risk is TRUE or action is {actions[2]}"
        )
        trusted_text = (
            f"role is {roles[0]} outside context {contexts[2]}"
        )
    elif family == 1:
        dangerous_expr = Or(
            And(
                _eq("event.risk", "TRUE"),
                _eq("event.context", contexts[2]),
            ),
            _eq("event.action", actions[2]),
        )
        trusted_expr = Or(
            _eq("event.role", roles[0]),
            And(
                _eq("event.context", contexts[0]),
                _eq("event.risk", "FALSE"),
            ),
        )
        dangerous_text = (
            f"(risk TRUE in context {contexts[2]}) or action {actions[2]}"
        )
        trusted_text = (
            f"role {roles[0]}, or context {contexts[0]} with FALSE risk"
        )
    elif family == 2:
        dangerous_expr = Or(
            _eq("event.risk", "TRUE"),
            And(
                _eq("event.action", actions[2]),
                _eq("event.context", contexts[2]),
            ),
        )
        trusted_expr = And(
            Or(
                _eq("event.role", roles[0]),
                _eq("event.role", roles[1]),
            ),
            Not(_eq("event.action", actions[2])),
        )
        dangerous_text = (
            f"risk TRUE, or action {actions[2]} in context {contexts[2]}"
        )
        trusted_text = (
            f"role {roles[0]} or {roles[1]}, except for action {actions[2]}"
        )
    else:
        dangerous_expr = Or(
            And(
                _eq("event.risk", "TRUE"),
                _eq("event.role", roles[2]),
            ),
            _eq("event.action", actions[2]),
        )
        trusted_expr = And(
            _eq("event.context", contexts[0]),
            Not(_eq("event.role", roles[2])),
        )
        dangerous_text = (
            f"(risk TRUE for role {roles[2]}) or action {actions[2]}"
        )
        trusted_text = (
            f"context {contexts[0]} with any role except {roles[2]}"
        )
    dangerous = PredicateDecl("Dangerous", "event", dangerous_expr)
    trusted = PredicateDecl("Trusted", "event", trusted_expr)
    reviewer_condition = (
        And(
            _eq("event.role", roles[1]),
            _eq("event.context", contexts[1]),
        )
        if family in {1, 3}
        else Or(
            _eq("event.role", roles[1]),
            _eq("event.context", contexts[1]),
        )
    )
    reviewer_join = "and" if family in {1, 3} else "or"
    rules = [
        Rule(
            "CriticalRisk",
            PredicateCall("Dangerous"),
            [Effect("require", deny)],
            "clause-critical-risk",
        ),
        Rule(
            "UnknownExternal",
            And(
                _eq("event.risk", "UNKNOWN"),
                _eq("event.context", contexts[2]),
            ),
            [Effect("abstain", None)],
            "clause-unknown-external",
        ),
        Rule(
            "UnknownRisk",
            _eq("event.risk", "UNKNOWN"),
            [_prefer(review, 90)],
            "clause-unknown-risk",
        ),
        Rule(
            "TrustedRoutine",
            And(PredicateCall("Trusted"), _eq("event.action", actions[0])),
            [_prefer(allow, 80)],
            "clause-trusted-routine",
        ),
        Rule(
            "ExternalChange",
            And(
                _eq("event.context", contexts[2]),
                _eq("event.action", actions[1]),
            ),
            [_prefer(deny, 70)],
            "clause-external-change",
        ),
        Rule(
            "ReviewerCase",
            reviewer_condition,
            [_prefer(review, 40)],
            "clause-reviewer-case",
        ),
        Rule(
            "TrustedNoReview",
            PredicateCall("Trusted"),
            [Effect("forbid", review)],
            "clause-trusted-no-review",
        ),
        Rule(
            "Default",
            _in("event.role", roles),
            [_prefer(allow, 1)],
            "clause-default",
        ),
    ]
    program = Program([dangerous, trusted], rules, [])
    record = RecordType(
        "Event",
        [
            FieldDecl("role", "Role"),
            FieldDecl("action", "Action"),
            FieldDecl("context", "Context"),
            FieldDecl("risk", "Boolean3"),
        ],
    )
    requirements = [
        f"Critical means {dangerous_text}; critical cases require {deny}.",
        (
            f"Unknown external: risk UNKNOWN in context {contexts[2]} "
            "returns uncertain."
        ),
        f"Unknown risk otherwise prefers {review} at priority 90.",
        (
            f"Trusted means {trusted_text}; "
            f"a trusted {actions[0]} prefers {allow} at priority 80."
        ),
        (
            f"Action {actions[1]} in context {contexts[2]} prefers {deny} "
            "at priority 70."
        ),
        (
            f"Role {roles[1]} {reviewer_join} context {contexts[1]} "
            f"prefers {review} "
            "at priority 40."
        ),
        f"Trusted cases forbid {review}.",
        f"All other declared roles prefer {allow} at priority 1.",
        (
            "Required labels override preferences. Abstention or conflicting "
            "required labels returns uncertain. Forbidden labels are removed; "
            "the remaining highest-priority preference wins."
        ),
    ]
    task_spec = TaskSpec(
        task_name=f"Compositional policy {task_number:03d}",
        description=(
            "Classify one structured event using the following public policy. "
            + " ".join(requirements)
        ),
        labels=labels,
        requirements=requirements,
        forbidden_shortcuts=["Use every field named by a relevant rule."],
        sources=[
            f"{SUITE_VERSION}:family-{family}:task-{task_number:03d}"
        ],
        enums=[
            EnumType("Role", roles),
            EnumType("Action", actions),
            EnumType("Context", contexts),
        ],
        records=[record],
    )
    validate_program(program, task_spec)
    records = [
        {
            "event": {
                "role": role,
                "action": action,
                "context": context,
                "risk": risk,
            }
        }
        for role, action, context, risk in itertools.product(
            roles, actions, contexts, ["TRUE", "FALSE", "UNKNOWN"]
        )
    ]
    build, audit, final = _select_cases(
        program, records, seed=task_number
    )
    return PolicyTask(
        task_id=f"policy-{task_number:03d}",
        suite_split=suite_split,
        task_spec=task_spec,
        oracle_program=program,
        build_cases=build,
        audit_cases=audit,
        final_cases=final,
    )


def suite_manifest(split: str) -> dict:
    """Return the immutable suite manifest without model outputs."""
    if split not in TASK_SPLITS:
        raise ValueError(f"unknown compositional suite split: {split}")
    tasks = [make_policy_task(number) for number in TASK_SPLITS[split]]
    payload = {
        "suite_version": SUITE_VERSION,
        "split": split,
        "tasks": [
            {
                "task_id": task.task_id,
                "source": task.task_spec.sources[0],
                "program_sha256": hashlib.sha256(
                    render_program_source(task.oracle_program).encode("utf-8")
                ).hexdigest(),
                "build_inputs": [
                    case.input_record for case in task.build_cases
                ],
                "audit_inputs": [
                    case.input_record for case in task.audit_cases
                ],
                "final_inputs": [
                    case.input_record for case in task.final_cases
                ],
            }
            for task in tasks
        ],
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _student(config: RunConfig) -> StudentModel:
    return StudentModel(
        config.student_model,
        max_new_tokens=config.max_answer_tokens,
        revision=config.student_revision,
        backend=config.student_backend,
        ollama_base_url=config.ollama_base_url,
        openrouter_base_url=config.openrouter_base_url,
        openrouter_timeout_seconds=config.student_openrouter_timeout_seconds,
        openrouter_model_name=config.student_openrouter_model,
        openrouter_provider=config.student_openrouter_provider,
        ollama_timeout_seconds=config.ollama_timeout_seconds,
        seed=config.student_seed,
        context_tokens=config.student_context_tokens,
        enable_thinking=config.student_thinking,
    )


def _prompt(task: PolicyTask, rules: list[str] | None, record: dict) -> str:
    sections = [f"Task: {task.task_spec.description}"]
    if rules:
        sections.append("Compiled rules:\n" + "\n".join(f"- {text}" for text in rules))
    sections.append(f"Input: {record}")
    sections.append(
        "Answer with only one declared label, or uncertain if the policy does not decide."
    )
    return "\n\n".join(sections)


def _deployment_description(
    task: PolicyTask,
    include_task_spec: bool,
    output_names: list[str] | None = None,
) -> str:
    if include_task_spec:
        return task.task_spec.description
    labels = ", ".join(
        output_names
        or [*task.task_spec.labels, "uncertain"]
    )
    return (
        "Classify one structured event using the supplied compiled "
        f"instructions. The allowed outputs are {labels}. "
        "These names are arbitrary symbols, so do not infer their meaning. "
        "Copy the output stated by the matching compiled instruction."
    )


def _opaque_label_codebook(
    task: PolicyTask,
) -> tuple[dict[str, str], dict[str, str | None]]:
    labels = sorted([*task.task_spec.labels, "uncertain"])
    encoder = {
        label: f"choice_{chr(ord('a') + index)}"
        for index, label in enumerate(labels)
    }
    decoder = {
        code: None if label == "uncertain" else label
        for label, code in encoder.items()
    }
    return encoder, decoder


def _prompt_prefix(
    task: PolicyTask,
    rules: list[str],
    task_description: str | None = None,
    output_names: list[str] | None = None,
) -> str:
    output_contract = ", ".join(
        output_names
        or [*task.task_spec.labels, "uncertain"]
    )
    return "\n\n".join(
        [
            f"Task: {task_description or task.task_spec.description}",
            "Compiled rules:\n" + "\n".join(f"- {text}" for text in rules),
            (
                f"Output contract: return exactly one of {output_contract}. "
                "Do not include reasoning or formatting."
            ),
        ]
    )


def _prompt_input(record: dict) -> str:
    return "\n\n".join(
        [
            f"Input: {record}",
            (
                "Answer with only one declared label, or uncertain if the "
                "policy does not decide."
            ),
        ]
    )


def _evaluate_cases(
    task: PolicyTask,
    cases: list[RuleCase],
    student: StudentModel | None,
    rules: list[str] | None,
    optimized_prefix: str | None = None,
    instruction_router: CertifiedInstructionRouter | None = None,
    top_k_rules: int = 0,
    task_description: str | None = None,
    label_encoder: dict[str, str] | None = None,
    label_decoder: dict[str, str | None] | None = None,
) -> tuple[int, list[dict], int, int]:
    labels = {label.lower() for label in task.task_spec.labels}
    correct = 0
    predictions = []
    input_tokens = 0
    output_tokens = 0
    for index, case in enumerate(cases):
        routed = None
        if student is None:
            predicted = run(task.oracle_program, case.input_record).output_label
            raw = predicted or "uncertain"
        else:
            routed = (
                instruction_router.route(
                    case.input_record,
                    rules,
                    top_k=top_k_rules,
                    label_encoder=label_encoder,
                )
                if instruction_router is not None and rules
                else None
            )
            case_rules = routed.instructions if routed is not None else rules
            system = (
                optimized_prefix
                if optimized_prefix is not None
                else _prompt_prefix(
                    task,
                    case_rules,
                    task_description,
                    list(label_decoder) if label_decoder else None,
                )
                if case_rules
                else "\n\n".join(
                    [
                            f"Task: {task_description or task.task_spec.description}",
                        (
                            "Output contract: return exactly one declared "
                            "label, or uncertain if the policy does not "
                            "decide. Do not include reasoning or formatting."
                        ),
                    ]
                )
            )
            response = student.ask(
                f"Input: {case.input_record}",
                system=system,
            )
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            raw = response.text
            predicted = normalise_student_label(
                raw,
                set(label_decoder) if label_decoder else labels,
            )
            if label_decoder is not None and predicted is not None:
                predicted = label_decoder.get(
                    predicted,
                    f"__INVALID__:{predicted}",
                )
        matched = predicted == case.expected_label
        correct += int(matched)
        predictions.append(
            {
                "task_id": task.task_id,
                "example_id": f"{task.task_id}-final-{index:03d}",
                "input": case.input_record,
                "gold": case.expected_label or "uncertain",
                "predicted": predicted or "uncertain",
                "raw_output": raw,
                "correct": matched,
                "input_tokens": (
                    response.input_tokens if student is not None else 0
                ),
                "output_tokens": (
                    response.output_tokens if student is not None else 0
                ),
                "routed_instruction_count": (
                    len(routed.instructions) if routed is not None else None
                ),
                "routing_certificate": (
                    routed.certificate if routed is not None else None
                ),
            }
        )
    return correct, predictions, input_tokens, output_tokens


def execute(
    config: RunConfig,
    recorder: RunRecorder,
    run_dir: Path,
) -> dict:
    """Run the locked compositional-policy benchmark."""
    if config.baseline not in {
        "raw_task",
        "one_shot_render",
        "test_driven",
        "full_method",
        "certified_search",
        "certified_routing",
        "direct_dsl",
        APE_OPTIMIZER,
        GEPA_OPTIMIZER,
        *TEACHER_COMPILED_BASELINES,
        FROZEN_BUNDLE,
        FROZEN_BUNDLE_ROUTING,
    }:
        raise ValueError(
            f"unknown baseline for synthetic_policies: {config.baseline}"
        )
    if config.benchmark_split not in TASK_SPLITS:
        raise ValueError(
            f"benchmark_split must be one of {sorted(TASK_SPLITS)}"
        )
    if config.compilation_failure_fallback not in {"abort", "raw_task"}:
        raise ValueError(
            "compilation_failure_fallback must be abort or raw_task"
        )
    if config.max_teacher_syntax_repair_rounds < 0:
        raise ValueError(
            "max_teacher_syntax_repair_rounds must be nonnegative"
        )
    for name, threshold in {
        "teacher_build_pass_threshold": (
            config.teacher_build_pass_threshold
        ),
        "teacher_audit_pass_threshold": (
            config.teacher_audit_pass_threshold
        ),
    }.items():
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    manifest = suite_manifest(config.benchmark_split)
    (run_dir / "artifacts" / "suite_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    split_numbers = set(TASK_SPLITS[config.benchmark_split])
    if config.benchmark_task_ids:
        numbers = list(config.benchmark_task_ids)
        invalid = sorted(set(numbers) - split_numbers)
        if invalid:
            raise ValueError(
                f"task ids are outside {config.benchmark_split}: {invalid}"
            )
        if len(numbers) != len(set(numbers)):
            raise ValueError("benchmark_task_ids must be unique")
    else:
        numbers = list(split_numbers)
        random.Random(config.seed).shuffle(numbers)
        numbers = numbers[: min(config.num_tasks, len(numbers))]
    tasks = [make_policy_task(number) for number in numbers]
    lineage = (
        validate_lineage_config(config)
        if config.baseline in TEACHER_COMPILED_BASELINES
        else None
    )

    student = None if config.baseline == "direct_dsl" else _student(config)
    if student is not None:
        recorder.set_model_identity("student", student.identity)
    instruction_router = (
        CertifiedInstructionRouter(
            config.retriever_model,
            retriever_revision=config.retriever_revision,
        )
        if config.baseline in {
            "certified_routing",
            TEACHER_VALIDATED_ROUTING,
            FROZEN_BUNDLE_ROUTING,
        }
        else None
    )
    if instruction_router is not None:
        recorder.set_model_identity("router", instruction_router.identity)
    teacher = (
        TeacherClient(
            config.teacher_model,
            backend=config.teacher_backend,
            base_url=config.openrouter_base_url,
            timeout_seconds=config.teacher_timeout_seconds,
            provider=config.teacher_openrouter_provider,
        )
        if config.baseline
        in {
            "test_driven",
            APE_OPTIMIZER,
            GEPA_OPTIMIZER,
            *TEACHER_COMPILED_BASELINES,
        }
        else None
    )
    if teacher is not None:
        recorder.set_model_identity("teacher", teacher.identity)
    if config.baseline == GEPA_OPTIMIZER:
        recorder.set_model_identity(
            "optimizer",
            {"backend": "anthropic", "model": config.teacher_model},
        )

    all_predictions = []
    task_metrics = []
    total_correct = 0
    total_examples = 0
    compilation_student_calls = 0
    compilation_teacher_calls = 0
    audit_student_calls = 0
    compilation_input_tokens = 0
    compilation_output_tokens = 0
    audit_input_tokens = 0
    audit_output_tokens = 0
    final_input_tokens = 0
    final_output_tokens = 0
    teacher_initial_build_rates = []
    teacher_final_build_rates = []
    teacher_final_audit_rates = []
    teacher_repair_counts = []
    compilation_failures = []
    with recorder.stage("compile_and_freeze_artifacts"):
        frozen = []
        for task in tasks:
            label_encoder, label_decoder = (
                _opaque_label_codebook(task)
                if instruction_router is not None
                and config.routing_opaque_labels
                else (None, None)
            )
            deployment_uses_full_task_spec = (
                config.deployment_include_task_spec
                or config.baseline == "raw_task"
            )
            deployment_description = _deployment_description(
                task,
                deployment_uses_full_task_spec,
                list(label_decoder) if label_decoder else None,
            )
            teacher_compilation = None
            frozen_bundle_metadata = None
            compilation_failure = None
            program_for_transfer = task.oracle_program
            if config.baseline in TEACHER_COMPILED_BASELINES:
                assert teacher is not None
                frozen_teacher_source = None
                if config.teacher_program_path:
                    teacher_program_path = Path(
                        config.teacher_program_path
                    )
                    if not teacher_program_path.is_absolute():
                        teacher_program_path = (
                            Path.cwd() / teacher_program_path
                        )
                    frozen_teacher_source = (
                        teacher_program_path.read_text(encoding="utf-8")
                    )
                try:
                    teacher_compilation = compile_specification(
                        teacher,
                        task.task_spec,
                        task.build_cases,
                        task.audit_cases,
                        max_repair_rounds=(
                            config.max_teacher_repair_rounds
                            if config.max_teacher_repair_rounds is not None
                            else config.max_repair_rounds
                        ),
                        max_syntax_repair_rounds=(
                            config.max_teacher_syntax_repair_rounds
                        ),
                        max_tokens=max(600, min(config.token_budget, 4000)),
                        source_id=task.task_spec.sources[0],
                        repair=config.baseline != TEACHER_ONE_SHOT,
                        refine_with_audit=config.teacher_refine_with_audit,
                        max_counterexample_patches=(
                            config.max_counterexample_patches
                        ),
                        structured_output=config.teacher_structured_outputs,
                        initial_source=frozen_teacher_source,
                        seed=config.seed,
                        on_teacher_response=lambda response: (
                            recorder.add_teacher_tokens(
                                response.input_tokens,
                                response.output_tokens,
                            )
                        ),
                        on_teacher_text=lambda phase, text: (
                            run_dir
                            / "artifacts"
                            / f"{task.task_id}.teacher.{phase}.raw.txt"
                        ).write_text(text, encoding="utf-8"),
                    )
                except DslSyntaxError as exc:
                    if config.compilation_failure_fallback == "abort":
                        raise
                    failed_source = getattr(exc, "failed_source", "")
                    syntax_repairs = getattr(exc, "syntax_repairs", [])
                    teacher_calls = getattr(exc, "teacher_calls", 0)
                    teacher_input_tokens = getattr(
                        exc, "teacher_input_tokens", 0
                    )
                    teacher_output_tokens = getattr(
                        exc, "teacher_output_tokens", 0
                    )
                    compilation_teacher_calls += teacher_calls
                    compilation_failure = {
                        "stage": "teacher_syntax",
                        "message": str(exc),
                        "fallback": "raw_task",
                        "teacher_calls": teacher_calls,
                        "teacher_input_tokens": teacher_input_tokens,
                        "teacher_output_tokens": teacher_output_tokens,
                        "syntax_repairs": [
                            asdict(item) for item in syntax_repairs
                        ],
                    }
                    compilation_failures.append(
                        {
                            "task_id": task.task_id,
                            **compilation_failure,
                        }
                    )
                    deployment_uses_full_task_spec = True
                    deployment_description = _deployment_description(
                        task,
                        True,
                        list(label_decoder) if label_decoder else None,
                    )
                    if failed_source:
                        (
                            run_dir
                            / "artifacts"
                            / f"{task.task_id}.teacher.failed.dsl"
                        ).write_text(failed_source, encoding="utf-8")
                    (
                        run_dir
                        / "artifacts"
                        / f"{task.task_id}.teacher.compilation.json"
                    ).write_text(
                        json.dumps(
                            {
                                "validation_gate": {"passed": False},
                                "failure": compilation_failure,
                                "teacher_received_final_cases": False,
                                "teacher_received_private_deployment_data": (
                                    False
                                ),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                if teacher_compilation is None:
                    program_for_transfer = task.oracle_program
                    build_inputs = [
                        case.input_record for case in task.build_cases
                    ]
                    audit_inputs = [
                        case.input_record for case in task.audit_cases
                    ]
                else:
                    program_for_transfer = (
                        teacher_compilation.initial_program
                        if config.baseline == TEACHER_ONE_SHOT
                        else teacher_compilation.final_program
                    )
                    compilation_teacher_calls += (
                        teacher_compilation.teacher_calls
                    )
                    teacher_initial_build_rates.append(
                        teacher_compilation.initial_build.pass_rate
                    )
                    teacher_final_build_rates.append(
                        teacher_compilation.final_build.pass_rate
                    )
                    teacher_final_audit_rates.append(
                        teacher_compilation.final_audit.pass_rate
                    )
                    teacher_repair_counts.append(
                        len(teacher_compilation.repairs)
                    )
                if teacher_compilation is not None:
                    (
                        run_dir
                        / "artifacts"
                        / f"{task.task_id}.teacher.initial.dsl"
                    ).write_text(
                        teacher_compilation.initial_source,
                        encoding="utf-8",
                    )
                    (
                        run_dir
                        / "artifacts"
                        / f"{task.task_id}.teacher.final.dsl"
                    ).write_text(
                        render_program_source(
                            teacher_compilation.final_program
                        ),
                        encoding="utf-8",
                    )
                    (
                        run_dir
                        / "artifacts"
                        / f"{task.task_id}.teacher.compilation.json"
                    ).write_text(
                        json.dumps(
                            {
                                "initial_build_pass_rate": (
                                    teacher_compilation.initial_build.pass_rate
                                ),
                                "final_build_pass_rate": (
                                    teacher_compilation.final_build.pass_rate
                                ),
                                "final_audit_pass_rate": (
                                    teacher_compilation.final_audit.pass_rate
                                ),
                                "validation_gate": {
                                    "build_pass_threshold": (
                                        config.teacher_build_pass_threshold
                                    ),
                                    "audit_pass_threshold": (
                                        config.teacher_audit_pass_threshold
                                    ),
                                    "passed": (
                                        _passes_teacher_validation(
                                            teacher_compilation,
                                            config,
                                        )
                                    ),
                                },
                                "repairs": [
                                    asdict(item)
                                    for item in teacher_compilation.repairs
                                ],
                                "syntax_repairs": [
                                    asdict(item)
                                    for item in (
                                        teacher_compilation.syntax_repairs
                                    )
                                ],
                                "teacher_calls": (
                                    teacher_compilation.teacher_calls
                                ),
                                "teacher_input_tokens": (
                                    teacher_compilation.teacher_input_tokens
                                ),
                                "teacher_output_tokens": (
                                    teacher_compilation.teacher_output_tokens
                                ),
                                "teacher_received_final_cases": False,
                                "teacher_received_private_deployment_data": (
                                    False
                                ),
                                "audit_scenarios_used_for_refinement": (
                                    config.teacher_refine_with_audit
                                ),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                if (
                    teacher_compilation is not None
                    and
                    config.baseline != TEACHER_ONE_SHOT
                    and not _passes_teacher_validation(
                        teacher_compilation,
                        config,
                    )
                ):
                    failure_message = (
                        f"{task.task_id} parent program failed its "
                        "source-derived validation gate: build="
                        f"{teacher_compilation.final_build.pass_rate:.3f}, "
                        "audit="
                        f"{teacher_compilation.final_audit.pass_rate:.3f}; "
                        "required build="
                        f"{config.teacher_build_pass_threshold:.3f}, audit="
                        f"{config.teacher_audit_pass_threshold:.3f}"
                    )
                    if config.compilation_failure_fallback == "abort":
                        raise RuntimeError(failure_message)
                    compilation_failure = {
                        "stage": "teacher_validation_gate",
                        "message": failure_message,
                        "fallback": "raw_task",
                    }
                    compilation_failures.append(
                        {
                            "task_id": task.task_id,
                            **compilation_failure,
                        }
                    )
                    deployment_uses_full_task_spec = True
                    deployment_description = _deployment_description(
                        task,
                        True,
                        list(label_decoder) if label_decoder else None,
                    )
                    program_for_transfer = task.oracle_program
            build_inputs = [case.input_record for case in task.build_cases]
            audit_inputs = [case.input_record for case in task.audit_cases]
            obligations, build_probes = build_obligation_probes(
                program_for_transfer,
                build_inputs,
                f"{task.task_id}-build",
                oracle_program=task.oracle_program,
                causal_only=True,
            )
            _, audit_probes = build_obligation_probes(
                program_for_transfer,
                audit_inputs,
                f"{task.task_id}-audit",
                oracle_program=task.oracle_program,
                causal_only=True,
            )
            compilation = None
            optimizer_protocol = None
            optimized_prefix = None
            if compilation_failure is not None:
                bundle = FreeTextBundle([])
            elif config.baseline in {
                TEACHER_VALIDATED,
                TEACHER_VALIDATED_ROUTING,
            }:
                compilation = behavioral_compile(
                    student,
                    deployment_description,
                    program_for_transfer,
                    build_probes,
                    [],
                    max_rounds=config.max_repair_rounds,
                    token_budget=config.token_budget,
                    max_student_calls=len(build_probes)
                    * (1 + config.max_repair_rounds),
                    max_protected_regressions=(
                        config.max_protected_regressions
                    ),
                    mode="certified_search",
                    render_for_probe=(
                        (
                            lambda candidate, probe: (
                                instruction_router.route(
                                    probe.input_record,
                                    candidate.render(),
                                    top_k=config.top_k_rules,
                                    label_encoder=label_encoder,
                                ).instructions
                            )
                        )
                        if instruction_router is not None
                        else None
                    ),
                    label_decoder=label_decoder,
                )
                bundle = compilation.bundle
            elif config.baseline in {
                FROZEN_BUNDLE,
                FROZEN_BUNDLE_ROUTING,
            }:
                if not config.compiled_guidance_path:
                    raise ValueError(
                        "frozen bundle baselines require "
                        "compiled_guidance_path"
                    )
                bundle_path = Path(config.compiled_guidance_path)
                if not bundle_path.is_absolute():
                    bundle_path = Path.cwd() / bundle_path
                frozen_bundle_metadata = json.loads(
                    bundle_path.read_text(encoding="utf-8")
                )
                if frozen_bundle_metadata.get("task_id") != task.task_id:
                    raise ValueError(
                        "frozen bundle task does not match selected task"
                    )
                texts = frozen_bundle_metadata.get("instructions")
                if not isinstance(texts, list) or not all(
                    isinstance(text, str) for text in texts
                ):
                    raise ValueError(
                        "frozen bundle instructions must be strings"
                    )
                bundle = FreeTextBundle(texts)
            elif config.baseline in {
                "full_method",
                "certified_search",
                "certified_routing",
            }:
                compilation = behavioral_compile(
                    student,
                    deployment_description,
                    task.oracle_program,
                    build_probes,
                    [],
                    max_rounds=config.max_repair_rounds,
                    token_budget=config.token_budget,
                    max_student_calls=len(build_probes)
                    * (1 + config.max_repair_rounds),
                    max_protected_regressions=config.max_protected_regressions,
                    mode=(
                        "certified_search"
                        if config.baseline
                        in {"certified_search", "certified_routing"}
                        else "obligation_balanced"
                    ),
                    render_for_probe=(
                        (
                            lambda candidate, probe: instruction_router.route(
                                probe.input_record,
                                candidate.render(),
                                top_k=config.top_k_rules,
                                label_encoder=label_encoder,
                            ).instructions
                        )
                        if instruction_router is not None
                        else None
                    ),
                    label_decoder=label_decoder,
                )
                bundle = compilation.bundle
            elif config.baseline == "test_driven":
                compilation = generic_test_driven_compile(
                    student,
                    teacher,
                    deployment_description,
                    render_program(task.oracle_program),
                    build_probes,
                    [],
                    set(task.task_spec.labels),
                    max_rounds=config.max_repair_rounds,
                    token_budget=config.token_budget,
                    max_student_calls=len(build_probes)
                    * (1 + config.max_repair_rounds),
                    max_protected_regressions=config.max_protected_regressions,
                )
                bundle = compilation.bundle
                recorder.add_teacher_tokens(
                    compilation.teacher_input_tokens,
                    compilation.teacher_output_tokens,
                )
            elif config.baseline == APE_OPTIMIZER:
                from taskcompiler.guidance_track.ape_baseline import (
                    optimize_label_prompt,
                )

                development_rows = [
                    {
                        "item_id": (
                            f"{task.task_id}-{split_name}-{index:03d}"
                        ),
                        "input": case.input_record,
                        "answer": case.expected_label or "uncertain",
                    }
                    for split_name, cases in (
                        ("build", task.build_cases),
                        ("audit", task.audit_cases),
                    )
                    for index, case in enumerate(cases)
                ]
                optimized_prefix, optimizer_protocol = optimize_label_prompt(
                    config,
                    recorder,
                    run_dir,
                    task_id=task.task_id,
                    task_description=task.task_spec.description,
                    rows=development_rows,
                    starting_prompt=_prompt_prefix(
                        task,
                        [],
                        task.task_spec.description,
                    ),
                    labels=set(task.task_spec.labels) | {"uncertain"},
                    student=student,
                    teacher=teacher,
                )
                bundle = None
                compilation_student_calls += optimizer_protocol[
                    "task_model_calls"
                ]
                compilation_teacher_calls += optimizer_protocol[
                    "teacher_calls"
                ]
            elif config.baseline == GEPA_OPTIMIZER:
                from taskcompiler.guidance_track.gepa_baseline import (
                    optimize_label_prompt,
                )

                rendered = render_program(task.oracle_program)
                build_rows = [
                    {
                        "item_id": f"{task.task_id}-build-{index:03d}",
                        "input": _prompt_input(case.input_record),
                        "answer": case.expected_label or "uncertain",
                    }
                    for index, case in enumerate(task.build_cases)
                ]
                optimized_prefix, optimizer_protocol = optimize_label_prompt(
                    config,
                    recorder,
                    run_dir,
                    task_id=task.task_id,
                    rows=build_rows,
                    starting_prompt=_prompt_prefix(
                        task, rendered, deployment_description
                    ),
                    labels=set(task.task_spec.labels) | {"uncertain"},
                    student=student,
                    teacher=teacher,
                )
                bundle = None
                compilation_student_calls += optimizer_protocol[
                    "task_model_calls"
                ]
                compilation_teacher_calls += optimizer_protocol[
                    "reflection_calls"
                ]
            elif config.baseline in {
                "one_shot_render",
                TEACHER_ONE_SHOT,
                TEACHER_REPAIRED_RENDER,
            }:
                rendered = render_program(program_for_transfer)
                prefix_count = len(program_for_transfer.predicates)
                suffix_start = (
                    prefix_count + len(program_for_transfer.rules)
                )
                bundle = InstructionBundle(
                    [
                        InstructionEntry(copy.deepcopy(rule))
                        for rule in program_for_transfer.rules
                    ],
                    prefixes=rendered[:prefix_count],
                    suffixes=rendered[suffix_start:],
                )
            else:
                bundle = FreeTextBundle([])

            if compilation:
                compilation_student_calls += compilation.student_calls
                compilation_teacher_calls += getattr(
                    compilation, "teacher_calls", 0
                )
                recorder.add_student_tokens(
                    compilation.student_input_tokens,
                    compilation.student_output_tokens,
                )
                compilation_input_tokens += compilation.student_input_tokens
                compilation_output_tokens += compilation.student_output_tokens
            rules = (
                bundle.render()
                if bundle is not None
                and config.baseline not in {"raw_task", "direct_dsl"}
                else None
            )
            semantic_certificate = None
            selected_verified_decision_list = False
            if config.baseline in {
                "full_method",
                "certified_search",
                "certified_routing",
                TEACHER_VALIDATED,
                TEACHER_VALIDATED_ROUTING,
            } and compilation_failure is None:
                semantic_certificate = bundle.semantic_certificate
                selected_verified_decision_list = (
                    bundle.representation_kind
                    == "verified_decision_list"
                )
            artifact = {
                "task_id": task.task_id,
                "suite_split": task.suite_split,
                "source": task.task_spec.to_dict(),
                "distillation_lineage": (
                    lineage.to_dict() if lineage is not None else None
                ),
                "oracle_program_sha256": hashlib.sha256(
                    render_program_source(task.oracle_program).encode("utf-8")
                ).hexdigest(),
                "teacher_compilation": (
                    {
                        "enabled": True,
                        "teacher_backend": config.teacher_backend,
                        "teacher_model": config.teacher_model,
                        "teacher_selection": config.teacher_selection,
                        "initial_program_path": (
                            config.teacher_program_path
                        ),
                        "teacher_received_final_cases": False,
                        "teacher_received_private_deployment_data": False,
                        "audit_scenarios_used_for_refinement": (
                            config.teacher_refine_with_audit
                        ),
                        "source_derived_build_scenarios": len(
                            task.build_cases
                        ),
                        "source_derived_audit_scenarios": len(
                            task.audit_cases
                        ),
                        "initial_program_sha256": hashlib.sha256(
                            render_program_source(
                                teacher_compilation.initial_program
                            ).encode("utf-8")
                        ).hexdigest(),
                        "final_program_sha256": hashlib.sha256(
                            render_program_source(
                                teacher_compilation.final_program
                            ).encode("utf-8")
                        ).hexdigest(),
                        "initial_build_pass_rate": (
                            teacher_compilation.initial_build.pass_rate
                        ),
                        "final_build_pass_rate": (
                            teacher_compilation.final_build.pass_rate
                        ),
                        "final_audit_pass_rate": (
                            teacher_compilation.final_audit.pass_rate
                        ),
                        "validation_gate": {
                            "build_pass_threshold": (
                                config.teacher_build_pass_threshold
                            ),
                            "audit_pass_threshold": (
                                config.teacher_audit_pass_threshold
                            ),
                            "passed": _passes_teacher_validation(
                                teacher_compilation,
                                config,
                            ),
                        },
                        "repairs": [
                            asdict(item)
                            for item in teacher_compilation.repairs
                        ],
                        "teacher_input_tokens": (
                            teacher_compilation.teacher_input_tokens
                        ),
                        "teacher_output_tokens": (
                            teacher_compilation.teacher_output_tokens
                        ),
                        "teacher_calls": (
                            teacher_compilation.teacher_calls
                        ),
                    }
                    if teacher_compilation is not None
                    else {"enabled": False}
                ),
                "compilation_failure": compilation_failure,
                "rendered_bundle": rules,
                "frozen_bundle": frozen_bundle_metadata,
                "optimized_prompt_prefix": optimized_prefix,
                "optimizer_protocol": optimizer_protocol,
                "verified_decision_list_certificate": semantic_certificate,
                "representation_kind": (
                    bundle.representation_kind
                    if isinstance(bundle, InstructionBundle)
                    else None
                ),
                "local_instruction_routing": (
                    {
                        "enabled": True,
                        "top_k": config.top_k_rules,
                        "router": instruction_router.identity,
                    }
                    if instruction_router is not None
                    else {"enabled": False}
                ),
                "deployment_description": deployment_description,
                "deployment_includes_full_task_spec": (
                    deployment_uses_full_task_spec
                ),
                "opaque_label_codebook": (
                    {
                        "enabled": True,
                        "decoder": label_decoder,
                    }
                    if label_decoder is not None
                    else {"enabled": False}
                ),
                "selected_verified_decision_list": (
                    selected_verified_decision_list
                ),
                "num_obligations": len(obligations),
                "build_pass_rate": (
                    compilation.build_pass_rate
                    if compilation
                    else optimizer_protocol["best_build_accuracy"]
                    if optimizer_protocol
                    else None
                ),
                "initial_build_pass_rate": (
                    sum(
                        result.matched
                        for result in compilation.initial_build_results
                    )
                    / len(compilation.initial_build_results)
                    if hasattr(compilation, "initial_build_results")
                    else None
                ),
                "compilation_rounds": (
                    [
                        asdict(item)
                        if hasattr(item, "__dataclass_fields__")
                        else item
                        for item in compilation.rounds
                    ]
                    if compilation
                    else []
                ),
                "compilation_student_calls": (
                    compilation.student_calls
                    if compilation
                    else optimizer_protocol["task_model_calls"]
                    if optimizer_protocol
                    else 0
                ),
            }
            artifact_text = json.dumps(artifact, sort_keys=True)
            artifact["artifact_sha256"] = hashlib.sha256(
                artifact_text.encode("utf-8")
            ).hexdigest()
            (run_dir / "artifacts" / f"{task.task_id}.json").write_text(
                json.dumps(artifact, indent=2), encoding="utf-8"
            )
            frozen.append(
                (
                    task,
                    rules,
                    audit_probes,
                    optimized_prefix,
                    deployment_description,
                    label_encoder,
                    label_decoder,
                )
            )

    with recorder.stage("held_out_audit"):
        audit_rates = []
        for (
            task,
            rules,
            audit_probes,
            optimized_prefix,
            deployment_description,
            label_encoder,
            label_decoder,
        ) in frozen:
            if student is None:
                audit_rates.append(1.0)
                continue
            if optimized_prefix is not None:
                correct, _, input_tokens, output_tokens = _evaluate_cases(
                    task,
                    task.audit_cases,
                    student,
                    None,
                    optimized_prefix,
                    instruction_router,
                    config.top_k_rules,
                    deployment_description,
                    label_encoder,
                    label_decoder,
                )
                recorder.add_student_tokens(input_tokens, output_tokens)
                audit_input_tokens += input_tokens
                audit_output_tokens += output_tokens
                audit_student_calls += len(task.audit_cases)
                audit_rates.append(correct / len(task.audit_cases))
                continue
            results = [
                probe_student(
                    student,
                    deployment_description,
                    (
                        instruction_router.route(
                            probe.input_record,
                            rules or [],
                            top_k=config.top_k_rules,
                            label_encoder=label_encoder,
                        ).instructions
                        if instruction_router is not None and rules
                        else rules or []
                    ),
                    probe,
                    set(task.task_spec.labels),
                    label_decoder,
                )
                for probe in audit_probes
            ]
            recorder.add_student_tokens(
                sum(result.input_tokens for result in results),
                sum(result.output_tokens for result in results),
            )
            audit_input_tokens += sum(
                result.input_tokens for result in results
            )
            audit_output_tokens += sum(
                result.output_tokens for result in results
            )
            audit_student_calls += len(results)
            audit_rates.append(
                sum(result.matched for result in results) / len(results)
            )

    if config.open_final_evaluation:
        with recorder.stage("load_and_evaluate_locked_final"):
            for (
                task,
                rules,
                _,
                optimized_prefix,
                deployment_description,
                label_encoder,
                label_decoder,
            ), audit_rate in zip(frozen, audit_rates):
                cases = task.final_cases[: config.num_examples]
                correct, predictions, input_tokens, output_tokens = (
                    _evaluate_cases(
                        task,
                        cases,
                        student,
                        rules,
                        optimized_prefix,
                        instruction_router,
                        config.top_k_rules,
                        deployment_description,
                        label_encoder,
                        label_decoder,
                    )
                )
                recorder.add_student_tokens(input_tokens, output_tokens)
                final_input_tokens += input_tokens
                final_output_tokens += output_tokens
                total_correct += correct
                total_examples += len(cases)
                all_predictions.extend(predictions)
                task_metrics.append(
                    {
                        "task_id": task.task_id,
                        "accuracy": correct / len(cases),
                        "solved": correct == len(cases),
                        "num_examples": len(cases),
                        "audit_accuracy": audit_rate,
                    }
                )
    else:
        task_metrics = [
            {
                "task_id": task.task_id,
                "accuracy": None,
                "solved": None,
                "num_examples": 0,
                "audit_accuracy": audit_rate,
            }
            for (task, *_), audit_rate in zip(frozen, audit_rates)
        ]

    (run_dir / "artifacts" / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in all_predictions),
        encoding="utf-8",
    )
    interval = (
        wilson_accuracy(total_correct, total_examples)
        if total_examples
        else None
    )
    metrics = {
        "dataset": "synthetic_policies",
        "suite_version": SUITE_VERSION,
        "suite_split": config.benchmark_split,
        "manifest_sha256": manifest["manifest_sha256"],
        "baseline": config.baseline,
        "accuracy": (
            total_correct / total_examples if total_examples else None
        ),
        "macro_task_accuracy": (
            sum(item["accuracy"] for item in task_metrics)
            / len(task_metrics)
            if total_examples
            else None
        ),
        "solved_task_rate": (
            sum(item["solved"] for item in task_metrics)
            / len(task_metrics)
            if total_examples
            else None
        ),
        "accuracy_wilson_95": (
            {
                "low": interval.low,
                "high": interval.high,
                "descriptive_only": True,
            }
            if interval is not None
            else None
        ),
        "final_evaluation_opened": config.open_final_evaluation,
        "num_final_examples": total_examples,
        "num_task_groups": len(task_metrics),
        "inferential_unit": "independently generated policy task",
        "source_derived_scenario_outcomes_used_during_compilation": (
            config.baseline
            in {
                "test_driven",
                "full_method",
                "certified_search",
                "certified_routing",
                APE_OPTIMIZER,
                GEPA_OPTIMIZER,
                TEACHER_REPAIRED_RENDER,
                TEACHER_VALIDATED,
                TEACHER_VALIDATED_ROUTING,
            }
        ),
        "benchmark_labels_used_during_compilation": (
            config.baseline
            in {
                "test_driven",
                "full_method",
                "certified_search",
                "certified_routing",
                APE_OPTIMIZER,
                GEPA_OPTIMIZER,
                TEACHER_VALIDATED,
                TEACHER_VALIDATED_ROUTING,
            }
        ),
        "final_labels_used_during_compilation": False,
        "private_deployment_data_used_during_compilation": False,
        "teacher_received_final_cases": False,
        "teacher_received_private_deployment_data": False,
        "teacher_compilation": (
            {
                "enabled": True,
                "mean_initial_build_pass_rate": sum(
                    teacher_initial_build_rates
                )
                / len(teacher_initial_build_rates),
                "mean_final_build_pass_rate": sum(
                    teacher_final_build_rates
                )
                / len(teacher_final_build_rates),
                "mean_final_audit_pass_rate": sum(
                    teacher_final_audit_rates
                )
                / len(teacher_final_audit_rates),
                "total_repairs": sum(teacher_repair_counts),
                "tasks_compiled": len(teacher_initial_build_rates),
            }
            if teacher_initial_build_rates
            else {"enabled": False}
        ),
        "student_calls": {
            "compilation": compilation_student_calls,
            "audit": audit_student_calls,
            "final": total_examples if student is not None else 0,
        },
        "student_tokens_by_phase": {
            "compilation": {
                "input": compilation_input_tokens,
                "output": compilation_output_tokens,
            },
            "audit": {
                "input": audit_input_tokens,
                "output": audit_output_tokens,
            },
            "final": {
                "input": final_input_tokens,
                "output": final_output_tokens,
            },
        },
        "routing": (
            _routing_summary(
                all_predictions,
                top_k=config.top_k_rules,
                opaque_labels=config.routing_opaque_labels,
            )
            if instruction_router is not None and all_predictions
            else {"enabled": False}
        ),
        "teacher_calls": compilation_teacher_calls,
        "compilation_failures": compilation_failures,
        "compilation_failure_rate": (
            len(compilation_failures) / len(tasks)
        ),
        "compilation_failure_fallback": (
            config.compilation_failure_fallback
        ),
        "distillation_lineage": (
            lineage.to_dict() if lineage is not None else None
        ),
        "task_metrics": task_metrics,
    }
    recorder.set_task_metrics(metrics)
    return metrics
