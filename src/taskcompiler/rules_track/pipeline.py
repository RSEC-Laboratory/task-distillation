"""Run the ContractNLI rules-track experiment."""

from __future__ import annotations

import json
import hashlib
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from taskcompiler.config import DATA_DIR, RunConfig
from taskcompiler.dsl.ast import Program
from taskcompiler.dsl.mutate import run_mutation_testing
from taskcompiler.dsl.parser import (
    parse_program,
    render_program_source,
    validate_program,
)
from taskcompiler.evidence import EvidenceGraph, RuleRecord
from taskcompiler.metrics import RunRecorder
from taskcompiler.rules_track.contract_nli import (
    ContractNliExample,
    filter_by_hypothesis,
    load_split,
)
from taskcompiler.rules_track.prompts import (
    DSL_GENERATION_SYSTEM_PROMPT,
    build_dsl_generation_prompt,
)
from taskcompiler.rules_track.text_predicates import (
    Boolean3,
    TextEntailmentPredicate,
)
from taskcompiler.schema import FieldDecl, RecordType, TaskSpec
from taskcompiler.student import StudentModel
from taskcompiler.statistics import wilson_accuracy
from taskcompiler.teacher import TeacherClient
from taskcompiler.translate import (
    BehavioralCompilationResult,
    FreeTextBundle,
    GenericCompilationResult,
    InstructionBundle,
    InstructionEntry,
    behavioral_compile,
    build_obligation_probes,
    generic_test_driven_compile,
    render_program,
    roundtrip_check,
)
from taskcompiler.verify import RuleCase, repair_loop, run_unit_tests


LABEL_FOR_GOLD = {
    "entailment": "entailment",
    "contradiction": "contradiction",
    "not_mentioned": "not_mentioned",
}
GOLD_TO_BOOLEAN3 = {
    "entailment": Boolean3.TRUE,
    "contradiction": Boolean3.FALSE,
    "not_mentioned": Boolean3.UNKNOWN,
}
BASELINES = (
    "raw_task",
    "one_shot_render",
    "test_driven",
    "full_method",
    "direct_dsl",
    "contract_local_retrieval",
    "contract_compiled_transfer",
)
HYPOTHESIS_IDS = (
    "nda-1",
    "nda-2",
    "nda-3",
    "nda-4",
    "nda-5",
    "nda-7",
    "nda-8",
    "nda-10",
    "nda-11",
    "nda-12",
    "nda-13",
    "nda-15",
    "nda-16",
    "nda-17",
    "nda-18",
    "nda-19",
    "nda-20",
)


@dataclass
class CompilationState:
    task_description: str
    labels: list[str]
    dsl_source: str = ""
    program: Program | None = None
    repairs_applied: list[str] = field(default_factory=list)


def _public_task_spec(hypothesis_text: str) -> TaskSpec:
    return TaskSpec(
        task_name="Contract obligation status",
        description=(
            "Given the output of a frozen semantic predicate for the stated "
            f"contract obligation ({hypothesis_text}), return entailment when "
            "event.entails is TRUE, contradiction when it is FALSE, and "
            "not_mentioned when it is UNKNOWN."
        ),
        labels=["entailment", "contradiction", "not_mentioned"],
        requirements=[
            "event.entails == TRUE requires entailment",
            "event.entails == FALSE requires contradiction",
            "event.entails == UNKNOWN requires not_mentioned",
        ],
        forbidden_shortcuts=[
            "Do not infer a label from contract length or vocabulary alone."
        ],
        sources=["contract-nli-public-label-semantics"],
        records=[
            RecordType(
                "Event",
                [FieldDecl("entails", "Boolean3")],
            )
        ],
    )


def _synthetic_inputs(split: str) -> list[dict]:
    # The marker keeps build and audit artifacts disjoint.
    return [
        {"event": {"entails": value}, "_realization": split}
        for value in ("TRUE", "FALSE", "UNKNOWN")
    ]


def _synthetic_cases(program: Program, split: str) -> list[RuleCase]:
    from taskcompiler.dsl.interpreter import run

    return [
        RuleCase(record, run(program, record).output_label)
        for record in _synthetic_inputs(split)
    ]


def _source_expected_cases(split: str) -> list[RuleCase]:
    expected = {
        "TRUE": "entailment",
        "FALSE": "contradiction",
        "UNKNOWN": "not_mentioned",
    }
    return [
        RuleCase(record, expected[record["event"]["entails"]])
        for record in _synthetic_inputs(split)
    ]


def _load_final_examples(
    config: RunConfig, hypothesis_ids: list[str]
) -> list[ContractNliExample]:
    complete = load_split("test")
    selected = []
    for hypothesis_id in hypothesis_ids:
        examples = filter_by_hypothesis(complete, hypothesis_id)
        random.Random(f"{config.seed}:{hypothesis_id}").shuffle(examples)
        selected.extend(examples[: config.num_examples])
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_dsl_source(text: str) -> str:
    """Extract DSL source from an optional Markdown code fence."""
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    parts = stripped.split("```")
    for candidate in parts[1::2]:
        source = candidate.strip()
        first_line, separator, remainder = source.partition("\n")
        if first_line.strip().casefold() in {"dsl", "text", "plaintext"}:
            source = remainder.strip() if separator else ""
        if source.startswith(("predicate ", "rule ")):
            return source
    return stripped


def _generate_dsl(
    teacher: TeacherClient,
    state: CompilationState,
    task_spec: TaskSpec,
    recorder: RunRecorder,
    run_dir: Path,
) -> None:
    prompt = build_dsl_generation_prompt(
        state.task_description,
        state.labels,
        requirements=task_spec.requirements,
        forbidden_shortcuts=task_spec.forbidden_shortcuts,
        sources=task_spec.sources,
        task_spec=task_spec,
    )
    response = teacher.call(DSL_GENERATION_SYSTEM_PROMPT, prompt)
    recorder.add_teacher_tokens(response.input_tokens, response.output_tokens)
    (run_dir / "artifacts" / "dsl_teacher_response.txt").write_text(
        response.text, encoding="utf-8"
    )
    state.dsl_source = _extract_dsl_source(response.text)
    state.program = parse_program(state.dsl_source, task_spec=task_spec)
    for rule in state.program.rules:
        rule.source_clause = "contract-nli-public-label-semantics"


def _bundle_to_json(bundle) -> dict:
    if isinstance(bundle, FreeTextBundle):
        return {
            "artifact_type": "unconstrained_test_driven_baseline",
            "instructions": bundle.render(),
        }
    return {
        "artifact_type": "typed_instruction_bundle",
        "representation_kind": bundle.representation_kind,
        "semantic_certificate": bundle.semantic_certificate,
        "global_instructions": list(bundle.prefixes) + list(bundle.suffixes),
        "compiled_instructions": (
            list(bundle.override_texts)
            if bundle.override_texts is not None
            else None
        ),
        "entries": [
            {
                "rule_name": entry.rule_name,
                "source_clause": entry.rule.source_clause,
                "style": entry.style,
                "text": entry.render(),
                "examples": [
                    {"input": record, "expected_label": label}
                    for record, label in entry.examples
                ],
            }
            for entry in bundle.entries
        ]
    }


def _compile_upstream(
    config: RunConfig,
    recorder: RunRecorder,
    run_dir: Path,
    task_spec: TaskSpec,
) -> tuple[
    Program,
    InstructionBundle | FreeTextBundle,
    BehavioralCompilationResult | GenericCompilationResult | None,
    dict,
]:
    teacher = TeacherClient(
        config.teacher_model,
        backend=config.teacher_backend,
        base_url=config.openrouter_base_url,
        timeout_seconds=config.teacher_timeout_seconds,
        provider=config.teacher_openrouter_provider,
    )
    state = CompilationState(task_spec.description, task_spec.labels)
    with recorder.stage("generate_dsl"):
        _generate_dsl(teacher, state, task_spec, recorder, run_dir)

    build_cases = _source_expected_cases("build")
    audit_cases = _source_expected_cases("audit")
    before = run_unit_tests(state.program, build_cases)
    with recorder.stage("source_grounded_dsl_repair"):
        predicate_source = render_program_source(
            Program(state.program.predicates, [], [])
        )

        def parse_single_rule(text: str):
            source = _extract_dsl_source(text)
            if not source.startswith("rule "):
                source = f"rule Repaired:\n{source}"
            combined = f"{predicate_source}\n{source}" if predicate_source.strip() else source
            return parse_program(combined).rules[-1]

        repaired, attempts = repair_loop(
            state.program,
            render_rule=lambda rule: render_program_source(Program([], [rule], [])),
            parse_rule=parse_single_rule,
            test_cases=build_cases,
            teacher=teacher,
            max_rounds=config.max_repair_rounds,
            validate_candidate=lambda candidate: validate_program(
                candidate, task_spec
            ),
            on_teacher_response=lambda response: recorder.add_teacher_tokens(
                response.input_tokens, response.output_tokens
            ),
        )
        state.program = repaired
        for rule in state.program.rules:
            rule.source_clause = "contract-nli-public-label-semantics"
        state.repairs_applied = [attempt.rule_name for attempt in attempts if attempt.accepted]

    after = run_unit_tests(state.program, build_cases)
    audit = run_unit_tests(state.program, audit_cases)
    if after.pass_rate < 1.0:
        raise RuntimeError(
            "source-generated DSL failed the declared public build contract after repair"
        )

    build_inputs = _synthetic_inputs("student-build")
    audit_inputs = _synthetic_inputs("student-audit")
    obligations, build_probes = build_obligation_probes(
        state.program, build_inputs, "student-build"
    )
    _, audit_probes = build_obligation_probes(
        state.program, audit_inputs, "student-audit"
    )

    compilation: BehavioralCompilationResult | GenericCompilationResult | None = None
    rendered = render_program(state.program)
    prefix_count = len(state.program.predicates)
    suffix_start = prefix_count + len(state.program.rules)
    bundle = InstructionBundle(
        [InstructionEntry(rule=rule) for rule in state.program.rules],
        prefixes=rendered[:prefix_count],
        suffixes=rendered[suffix_start:],
    )
    if config.baseline in {"test_driven", "full_method"}:
        student = StudentModel(
            config.student_model,
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
        if hasattr(recorder, "set_model_identity"):
            recorder.set_model_identity(
                "student",
                getattr(
                    student,
                    "identity",
                    {"backend": config.student_backend, "model": config.student_model},
                ),
            )
        if config.baseline == "test_driven":
            with recorder.stage("generic_test_driven_prompt_compile"):
                compilation = generic_test_driven_compile(
                    student,
                    teacher,
                    task_spec.description,
                    render_program(state.program),
                    build_probes,
                    audit_probes,
                    set(task_spec.labels),
                    max_rounds=config.max_repair_rounds,
                    token_budget=config.token_budget,
                    max_protected_regressions=config.max_protected_regressions,
                )
                recorder.add_teacher_tokens(
                    compilation.teacher_input_tokens,
                    compilation.teacher_output_tokens,
                )
        else:
            with recorder.stage("obligation_balanced_behavioral_compile"):
                compilation = behavioral_compile(
                    student,
                    task_spec.description,
                    state.program,
                    build_probes,
                    audit_probes,
                    max_rounds=config.max_repair_rounds,
                    token_budget=config.token_budget,
                    mode="obligation_balanced",
                    max_protected_regressions=config.max_protected_regressions,
                )
        bundle = compilation.bundle
        recorder.add_student_tokens(
            compilation.student_input_tokens,
            compilation.student_output_tokens,
        )

    test_pairs = [(case.input_record, case.expected_label) for case in build_cases]
    mutation = run_mutation_testing(
        state.program,
        _synthetic_inputs("finite-domain"),
        test_pairs,
        exhaustive_equivalence_domain=True,
    )
    roundtrip = {
        rule.name: roundtrip_check(rule, build_inputs, state.program.predicates)
        for rule in state.program.rules
    }

    with recorder.stage("record_evidence"):
        evidence = EvidenceGraph(run_dir / "evidence_graph.jsonl")
        failed_before = (
            [
                result.probe_id
                for result in compilation.initial_build_results
                if not result.matched
            ]
            if isinstance(compilation, BehavioralCompilationResult)
            else []
        )
        failed_after = (
            [
                result.probe_id
                for result in compilation.audit_results
                if not result.matched
            ]
            if isinstance(compilation, BehavioralCompilationResult)
            else []
        )
        for rule in state.program.rules:
            rule_obligations = [
                obligation.obligation_id
                for obligation in obligations
                if obligation.rule_name == rule.name
            ]
            evidence.add(
                RuleRecord(
                    rule_name=rule.name,
                    source_clause=rule.source_clause,
                    build_tests_passed=after.passed,
                    build_tests_total=after.total,
                    audit_tests_passed=audit.passed,
                    audit_tests_total=audit.total,
                    mutation_adequacy=mutation.adequacy,
                    obligation_ids=rule_obligations,
                    student_failures_before=failed_before,
                    student_failures_after=failed_after,
                    repairs_applied=[
                        name for name in state.repairs_applied if name == rule.name
                    ],
                    translation_roundtrip_ok=roundtrip[rule.name],
                )
            )

    metadata = {
        "dsl_build_pass_rate_before": before.pass_rate,
        "dsl_build_pass_rate_after": after.pass_rate,
        "dsl_audit_pass_rate": audit.pass_rate,
        "mutation_adequacy": mutation.adequacy,
        "roundtrip_all_passed": all(roundtrip.values()),
        "num_obligations": len(obligations),
        "student_build_pass_rate": (
            compilation.build_pass_rate if compilation else None
        ),
        "student_audit_pass_rate": (
            compilation.audit_pass_rate if compilation else None
        ),
        "worst_obligation_violation": (
            compilation.worst_obligation_violation
            if isinstance(compilation, BehavioralCompilationResult)
            else None
        ),
    }
    (run_dir / "artifacts" / "dsl_final.txt").write_text(
        render_program_source(state.program), encoding="utf-8"
    )
    (run_dir / "artifacts" / "instruction_bundle.json").write_text(
        json.dumps(_bundle_to_json(bundle), indent=2), encoding="utf-8"
    )
    if compilation:
        (run_dir / "artifacts" / "compilation_rounds.json").write_text(
            json.dumps(
                [
                    asdict(item) if hasattr(item, "__dataclass_fields__") else item
                    for item in compilation.rounds
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    return state.program, bundle, compilation, metadata


def execute(
    config: RunConfig, recorder: RunRecorder, run_dir: Path
) -> dict:
    if config.dataset == "synthetic_policies":
        from taskcompiler.rules_track.synthetic_policies import execute as execute_policies

        return execute_policies(config, recorder, run_dir)
    if config.dataset != "contract_nli":
        raise ValueError(f"unknown rules_track dataset: {config.dataset}")
    if config.baseline in {
        "contract_local_retrieval",
        "contract_compiled_transfer",
    }:
        from taskcompiler.rules_track.contract_transfer import execute as transfer

        return transfer(config, recorder, run_dir)
    if config.baseline not in BASELINES:
        raise ValueError(f"unknown baseline for rules_track: {config.baseline}")

    task_ids = list(HYPOTHESIS_IDS)
    random.Random(config.seed).shuffle(task_ids)
    task_ids = task_ids[: min(config.num_tasks, len(task_ids))]
    # This public label contract contains no benchmark data.
    public_hypothesis = "the stated contractual obligation"
    task_spec = _public_task_spec(public_hypothesis)

    program = None
    bundle = None
    compilation_metadata = {}
    if config.baseline != "raw_task":
        program, bundle, _, compilation_metadata = _compile_upstream(
            config, recorder, run_dir, task_spec
        )

    # Benchmark records load only after all artifacts and settings are frozen.
    with recorder.stage("load_frozen_final_evaluation"):
        final_examples = _load_final_examples(config, task_ids)
        recorder.log(
            f"opened {len(final_examples)} held-out final examples after artifact freezing"
        )

    predicate = TextEntailmentPredicate(
        config.nli_model,
        config.nli_entailment_threshold,
        revision=config.nli_revision,
    )
    student = (
        None
        if config.baseline == "direct_dsl"
        else StudentModel(
            config.student_model,
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
    )
    if student is not None:
        recorder.set_model_identity(
            "student",
            getattr(
                student,
                "identity",
                {"backend": config.student_backend, "model": config.student_model},
            ),
        )
    predictions = []
    correct = 0
    predicate_correct = 0

    with recorder.stage("final_evaluation"):
        for example in final_examples:
            fact = predicate.evaluate(
                example.contract_text, example.hypothesis_text
            )
            predicate_correct += int(fact == GOLD_TO_BOOLEAN3[example.label])
            if config.baseline == "direct_dsl":
                from taskcompiler.dsl.interpreter import run

                predicted = run(
                    program, {"event": {"entails": fact.value}}
                ).output_label
            elif config.baseline == "raw_task":
                prompt = (
                    f"Task: {task_spec.description}\n\n"
                    f"Frozen semantic predicate: event.entails={fact.value}\n\n"
                    "Answer with only one label: entailment, contradiction, or not_mentioned."
                )
                response = student.ask(prompt)
                recorder.add_student_tokens(
                    response.input_tokens, response.output_tokens
                )
                predicted = response.text.strip().lower()
            else:
                rules_text = "\n".join(
                    f"- {text}" for text in bundle.render()
                )
                prompt = (
                    f"Task: {task_spec.description}\n\nRules:\n{rules_text}\n\n"
                    f"Frozen semantic predicate: event.entails={fact.value}\n"
                    "Answer with only one label: entailment, contradiction, or not_mentioned."
                )
                response = student.ask(prompt)
                recorder.add_student_tokens(
                    response.input_tokens, response.output_tokens
                )
                predicted = response.text.strip().lower()
            gold = LABEL_FOR_GOLD[example.label]
            correct += int(predicted == gold)
            predictions.append(
                {
                    "document_id": example.document_id,
                    "task_id": example.hypothesis_id,
                    "predicted": predicted,
                    "gold": gold,
                    "predicate": fact.value,
                }
            )

    accuracy = correct / len(final_examples) if final_examples else 0.0
    predicate_accuracy = (
        predicate_correct / len(final_examples) if final_examples else 0.0
    )
    (run_dir / "artifacts" / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in predictions),
        encoding="utf-8",
    )
    accuracy_interval = (
        wilson_accuracy(correct, len(final_examples))
        if final_examples
        else None
    )
    metrics = {
        "baseline": config.baseline,
        "accuracy": accuracy,
        "accuracy_wilson_95": (
            {
                "low": accuracy_interval.low,
                "high": accuracy_interval.high,
                "descriptive_only": True,
            }
            if accuracy_interval
            else None
        ),
        "predicate_accuracy": predicate_accuracy,
        "num_final_examples": len(final_examples),
        "num_task_groups": len(task_ids),
        "task_groups": task_ids,
        "inferential_unit": "contract-obligation task group",
        "evaluation_split": "test",
        "evaluation_file_sha256": _sha256(
            DATA_DIR / "contract_nli" / "test.json"
        ),
        "benchmark_labels_used_during_compilation": False,
        **compilation_metadata,
    }
    recorder.set_task_metrics(metrics)
    return metrics
