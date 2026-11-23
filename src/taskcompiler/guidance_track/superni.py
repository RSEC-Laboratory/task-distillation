"""Evaluate task-knowledge transfer on Super-NaturalInstructions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from taskcompiler.config import (
    ANTHROPIC_TEACHER,
    DATA_DIR,
    OLLAMA_BACKEND,
    OLLAMA_OPENROUTER_BACKEND,
    OPENROUTER_BACKEND,
    OPENROUTER_TEACHER,
    SUPERNI_COMPILER_MATRIX,
    RunConfig,
)
from taskcompiler.metrics import RunRecorder
from taskcompiler.statistics import task_bootstrap_difference
from taskcompiler.student import StudentModel
from taskcompiler.teacher import TeacherClient, TeacherResponse


SUPERNI_ROOT = DATA_DIR / "superni"
SUPERNI_COMMIT = "55a365637381ce7f3748fa2eac7aef1a113bbb82"
DEVELOPMENT_TASKS = (
    "task201_mnli_neutral_classification",
    "task893_gap_fill_the_blank_coreference_resolution",
    "task1624_disfl_qa_question_yesno_classification",
    "task935_defeasible_nli_atomic_classification",
)
CONFIRMATION_TASKS = (
    "task242_tweetqa_classification",
    "task1344_glue_entailment_classification",
    "task1439_doqa_cooking_isanswerable",
    "task738_perspectrum_classification",
    "task1615_sick_tclassify_b_relation_a",
    "task879_schema_guided_dstc8_classification",
    "task1387_anli_r3_entailment",
    "task642_esnli_classification",
    "task1390_wscfixed_coreference",
    "task1640_aqa1.0_answerable_unanswerable_question_classification",
    "task1388_cb_entailment",
    "task1534_daily_dialog_question_classification",
    "task232_iirc_link_number_classification",
    "task362_spolin_yesand_prompt_response_sub_classification",
    "task1554_scitail_classification",
    "task970_sherliic_causal_relationship",
    "task391_causal_relationship",
    "task220_rocstories_title_classification",
    "task050_multirc_answerability",
    "task349_squad2.0_answerable_unanswerable_question_classification",
)
CONDITIONS = (
    "raw_definition",
    "positive_examples",
    "parent_one_shot",
    "validated_full",
    "validated_routed",
)
REPRESENTATION_FIELDS = (
    "objective",
    "output_labels",
    "decision_rules",
    "boundary_conditions",
    "counterfactual_constraints",
    "verification_checks",
)
COMPILER_SYSTEM_PROMPT = (
    "Compile a public NLP task definition into reusable task knowledge for a "
    "frozen smaller model. Return only the requested JSON object. State general "
    "decision rules, boundary conditions, counterfactual constraints, and "
    "checks. Every rule must give an operational test that distinguishes output "
    "labels; do not merely paraphrase the definition. Do not solve or quote any "
    "hidden task instance."
)
STUDENT_SYSTEM_PROMPT = (
    "Apply the supplied public task knowledge to the private input. Return only "
    "one valid output label with no explanation."
)


@dataclass(frozen=True)
class SuperNIExample:
    item_id: str
    input_text: str
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class SuperNITask:
    task_id: str
    definition: str
    categories: tuple[str, ...]
    sources: tuple[str, ...]
    positive_examples: tuple[SuperNIExample, ...]
    negative_examples: tuple[SuperNIExample, ...]
    instances: tuple[SuperNIExample, ...]
    source_path: Path
    source_sha256: str


@dataclass(frozen=True)
class TaskRepresentation:
    objective: str
    output_labels: tuple[str, ...]
    decision_rules: tuple[str, ...]
    boundary_conditions: tuple[str, ...]
    counterfactual_constraints: tuple[str, ...]
    verification_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "output_labels": list(self.output_labels),
            "decision_rules": list(self.decision_rules),
            "boundary_conditions": list(self.boundary_conditions),
            "counterfactual_constraints": list(
                self.counterfactual_constraints
            ),
            "verification_checks": list(self.verification_checks),
        }


@dataclass
class CompilationResult:
    initial: TaskRepresentation | None
    final: TaskRepresentation | None
    initial_text: str
    final_text: str
    initial_validation: list[dict]
    final_validation: list[dict]
    revisions: list[dict]
    teacher_calls: int
    teacher_input_tokens: int
    teacher_output_tokens: int
    failure: dict | None = None


class LocalInstructionRanker:
    """Rank optional task instructions with a frozen local embedding model."""

    def __init__(self, model_name: str, revision: str | None = None):
        self.model = SentenceTransformer(model_name, revision=revision)
        self.identity = {
            "backend": "sentence_transformers",
            "model": model_name,
            "revision": revision,
        }
        self._cache: dict[tuple[str, ...], Any] = {}

    def select(
        self,
        query: str,
        representation: TaskRepresentation,
        top_k: int,
    ) -> tuple[list[str], dict]:
        mandatory = _mandatory_lines(representation)
        optional = _optional_lines(representation)
        if not optional or top_k >= len(optional):
            return mandatory + optional, {
                "selected_optional_indices": list(range(len(optional))),
                "optional_count": len(optional),
                "top_k": top_k,
            }
        key = tuple(optional)
        vectors = self._cache.get(key)
        if vectors is None:
            vectors = self.model.encode(optional, normalize_embeddings=True)
            self._cache[key] = vectors
        query_vector = self.model.encode(query, normalize_embeddings=True)
        scores = vectors @ query_vector
        ranked = sorted(
            range(len(optional)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        selected = sorted(ranked[: max(0, top_k)])
        return mandatory + [optional[index] for index in selected], {
            "selected_optional_indices": selected,
            "optional_count": len(optional),
            "top_k": top_k,
        }


def _as_string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    cleaned = tuple(item.strip() for item in value if isinstance(item, str))
    if len(cleaned) != len(value) or any(not item for item in cleaned):
        raise ValueError(f"{field} must contain non-empty strings")
    return cleaned


def _mapping_lines(
    value: Any,
    keys: tuple[str, ...],
) -> list[str]:
    lines = []
    if not isinstance(value, dict):
        return lines
    for label, item in value.items():
        if isinstance(item, str) and item.strip():
            lines.append(f"{label}: {item.strip()}")
            continue
        if not isinstance(item, dict):
            continue
        for key in keys:
            detail = item.get(key)
            if isinstance(detail, str) and detail.strip():
                lines.append(f"{label}: {detail.strip()}")
    return lines


def _normalise_provider_representation(
    value: dict,
    task: SuperNITask,
) -> dict:
    """Map equivalent provider fields into the canonical schema."""
    label_source = value.get("output_labels", value.get("labels"))
    if isinstance(label_source, list):
        output_labels = label_source
    elif isinstance(label_source, dict):
        output_labels = list(label_source)
    else:
        output_labels = []
    if not output_labels:
        output_labels = list(
            dict.fromkeys(
                output
                for example in task.positive_examples
                for output in example.outputs
            )
        )

    rules = value.get("decision_rules")
    if not isinstance(rules, list):
        rules = _mapping_lines(
            value.get("rules"),
            ("test", "description", "example"),
        )
    if not rules:
        rules = _mapping_lines(
            label_source,
            ("test", "description"),
        )

    boundaries = value.get("boundary_conditions")
    if isinstance(boundaries, dict):
        boundaries = _mapping_lines(
            boundaries,
            ("test", "description"),
        )
    if not isinstance(boundaries, list) or not boundaries:
        boundaries = _mapping_lines(
            label_source,
            (
                "missing_evidence",
                "missing_evidence_test",
                "ambiguity",
                "ambiguity_test",
                "role_change",
                "role_change_test",
            ),
        )

    counterfactuals = value.get("counterfactual_constraints")
    if isinstance(counterfactuals, dict):
        counterfactuals = _mapping_lines(
            counterfactuals,
            ("test", "description"),
        )
    if not isinstance(counterfactuals, list) or not counterfactuals:
        counterfactuals = _mapping_lines(
            label_source,
            (
                "counterfactual",
                "counterfactual_constraint",
                "negation",
                "negation_test",
            ),
        )

    checks = value.get("verification_checks", value.get("checks"))
    if isinstance(checks, dict):
        checks = _mapping_lines(checks, ("test", "description"))

    return {
        "objective": (
            value.get("objective")
            if isinstance(value.get("objective"), str)
            else "Apply the public task definition and return its required label."
        ),
        "output_labels": output_labels,
        "decision_rules": rules,
        "boundary_conditions": boundaries,
        "counterfactual_constraints": counterfactuals,
        "verification_checks": checks,
    }


def parse_representation(
    text: str,
    task: SuperNITask | None = None,
) -> TaskRepresentation:
    """Parse and validate one parent-produced task representation."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(
            r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE
        )
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"task representation is not valid JSON: {exc.msg}"
        ) from exc
    if (
        isinstance(value, dict)
        and set(value) != set(REPRESENTATION_FIELDS)
        and task is not None
    ):
        value = _normalise_provider_representation(value, task)
    if not isinstance(value, dict) or set(value) != set(REPRESENTATION_FIELDS):
        raise ValueError("task representation keys do not match the schema")
    objective = value["objective"]
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective must be a non-empty string")
    return TaskRepresentation(
        objective=objective.strip(),
        output_labels=_as_string_tuple(
            value["output_labels"], "output_labels"
        ),
        decision_rules=_as_string_tuple(
            value["decision_rules"], "decision_rules"
        ),
        boundary_conditions=_as_string_tuple(
            value["boundary_conditions"], "boundary_conditions"
        ),
        counterfactual_constraints=_as_string_tuple(
            value["counterfactual_constraints"],
            "counterfactual_constraints",
        ),
        verification_checks=_as_string_tuple(
            value["verification_checks"], "verification_checks"
        ),
    )


def _example(row: dict, fallback_id: str) -> SuperNIExample:
    outputs = row.get("output")
    if isinstance(outputs, str):
        outputs = [outputs]
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"{fallback_id} has no output")
    return SuperNIExample(
        item_id=str(row.get("id") or fallback_id),
        input_text=str(row["input"]),
        outputs=tuple(str(output) for output in outputs),
    )


def load_task(task_id: str) -> SuperNITask:
    """Load one frozen Super-NaturalInstructions task."""
    path = SUPERNI_ROOT / "tasks" / f"{task_id}.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    positives = tuple(
        _example(row, f"{task_id}-positive-{index:02d}")
        for index, row in enumerate(value.get("Positive Examples") or [])
    )
    negatives = tuple(
        _example(row, f"{task_id}-negative-{index:02d}")
        for index, row in enumerate(value.get("Negative Examples") or [])
    )
    instances = tuple(
        _example(row, f"{task_id}-instance-{index:04d}")
        for index, row in enumerate(value.get("Instances") or [])
    )
    if len(positives) < 2 or len(instances) < 100:
        raise ValueError(f"{task_id} does not satisfy the frozen protocol")
    return SuperNITask(
        task_id=task_id,
        definition=" ".join(str(item) for item in value["Definition"]).strip(),
        categories=tuple(str(item) for item in value.get("Categories") or []),
        sources=tuple(str(item) for item in value.get("Source") or []),
        positive_examples=positives,
        negative_examples=negatives,
        instances=instances,
        source_path=path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _public_examples(examples: tuple[SuperNIExample, ...]) -> str:
    rows = []
    for index, example in enumerate(examples, start=1):
        rows.append(
            f"Example {index}\nInput:\n{example.input_text}\n"
            f"Correct output:\n{example.outputs[0]}"
        )
    return "\n\n".join(rows)


def _negative_constraints(examples: tuple[SuperNIExample, ...]) -> str:
    if not examples:
        return "None supplied."
    rows = []
    for index, example in enumerate(examples, start=1):
        rows.append(
            f"Counterexample {index}\nInput:\n{example.input_text}\n"
            f"Explicitly incorrect output:\n{example.outputs[0]}"
        )
    return "\n\n".join(rows)


def _compiler_prompt(task: SuperNITask) -> str:
    return (
        f"Task ID: {task.task_id}\n\n"
        f"Public definition:\n{task.definition}\n\n"
        f"Public positive examples:\n"
        f"{_public_examples(task.positive_examples)}\n\n"
        f"Public negative examples:\n"
        f"{_negative_constraints(task.negative_examples)}\n\n"
        "Create a compact reusable representation. Output labels must be exact. "
        "Explain the input format and map each label to concrete observable "
        "tests. Use the public examples to infer reusable distinctions, but do "
        "not copy an example as a special-case rule. Include tests for negation, "
        "missing evidence, role or reference changes, and ambiguity when they "
        "apply. Boundary conditions must cover ambiguous and minimally supported "
        "cases. Counterfactual constraints must name minimal changes that alter "
        "or preserve a label. Verification checks must form a short ordered "
        "procedure that a small model can execute before answering."
    )


def _response_format() -> dict:
    array = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": 1,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "task_knowledge_representation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string", "minLength": 1},
                    "output_labels": array,
                    "decision_rules": array,
                    "boundary_conditions": array,
                    "counterfactual_constraints": array,
                    "verification_checks": array,
                },
                "required": list(REPRESENTATION_FIELDS),
                "additionalProperties": False,
            },
        },
    }


def _mandatory_lines(representation: TaskRepresentation) -> list[str]:
    return [
        f"Objective: {representation.objective}",
        "Output contract: return exactly one of "
        + ", ".join(representation.output_labels)
        + ".",
        *[
            f"Verification check: {item}"
            for item in representation.verification_checks
        ],
    ]


def _optional_lines(representation: TaskRepresentation) -> list[str]:
    return [
        *[f"Decision rule: {item}" for item in representation.decision_rules],
        *[
            f"Boundary condition: {item}"
            for item in representation.boundary_conditions
        ],
        *[
            f"Counterfactual constraint: {item}"
            for item in representation.counterfactual_constraints
        ],
    ]


def render_representation(representation: TaskRepresentation) -> list[str]:
    """Render a structured artifact into deterministic student instructions."""
    return _mandatory_lines(representation) + _optional_lines(representation)


def _normalise_label(value: str) -> str:
    cleaned = value.strip().strip("`\"' ")
    cleaned = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:|is|:)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.splitlines()[0].strip().strip("`\"' ")
    return re.sub(r"[\s.]+$", "", cleaned).casefold()


def response_matches(response: str, outputs: tuple[str, ...]) -> bool:
    """Match a concise model response against any allowed reference."""
    predicted = _normalise_label(response)
    return any(predicted == _normalise_label(output) for output in outputs)


def _student_prompt(
    definition: str,
    input_text: str,
    instructions: list[str] | None = None,
    examples: tuple[SuperNIExample, ...] = (),
) -> str:
    sections = [f"Public task definition:\n{definition}"]
    if examples:
        sections.append(f"Public examples:\n{_public_examples(examples)}")
    if instructions:
        sections.append("Task knowledge:\n" + "\n".join(instructions))
    sections.append(f"Private input:\n{input_text}\n\nOutput label:")
    return "\n\n".join(sections)


def _teacher_call(
    teacher: TeacherClient,
    system: str,
    prompt: str,
    config: RunConfig,
    recorder: RunRecorder,
    *,
    seed: int,
) -> TeacherResponse:
    response = teacher.call(
        system,
        prompt,
        max_tokens=max(600, min(config.token_budget, 3000)),
        response_format=(
            _response_format()
            if (
                config.teacher_backend == OPENROUTER_TEACHER
                and config.teacher_structured_outputs
            )
            else None
        ),
        seed=seed,
    )
    recorder.add_teacher_tokens(
        response.input_tokens, response.output_tokens
    )
    return response


def _validate_representation(
    task: SuperNITask,
    representation: TaskRepresentation,
    student: StudentModel,
    recorder: RunRecorder,
) -> list[dict]:
    instructions = render_representation(representation)
    rows = []
    for example in task.positive_examples:
        response = student.ask(
            _student_prompt(
                task.definition,
                example.input_text,
                instructions,
            ),
            system=STUDENT_SYSTEM_PROMPT,
        )
        recorder.add_student_tokens(
            response.input_tokens, response.output_tokens
        )
        rows.append(
            {
                "item_id": example.item_id,
                "expected": list(example.outputs),
                "response": response.text,
                "correct": response_matches(
                    response.text, example.outputs
                ),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        )
    return rows


def _compile_task(
    task: SuperNITask,
    teacher: TeacherClient,
    student: StudentModel,
    config: RunConfig,
    recorder: RunRecorder,
) -> CompilationResult:
    teacher_calls = 0
    teacher_input_tokens = 0
    teacher_output_tokens = 0
    response = _teacher_call(
        teacher,
        COMPILER_SYSTEM_PROMPT,
        _compiler_prompt(task),
        config,
        recorder,
        seed=config.seed,
    )
    teacher_calls += 1
    teacher_input_tokens += response.input_tokens
    teacher_output_tokens += response.output_tokens
    initial_text = response.text
    try:
        initial = parse_representation(
            initial_text,
            task=task if not config.teacher_structured_outputs else None,
        )
    except ValueError as exc:
        return CompilationResult(
            initial=None,
            final=None,
            initial_text=initial_text,
            final_text=initial_text,
            initial_validation=[],
            final_validation=[],
            revisions=[],
            teacher_calls=teacher_calls,
            teacher_input_tokens=teacher_input_tokens,
            teacher_output_tokens=teacher_output_tokens,
            failure={
                "stage": "representation_schema",
                "message": str(exc),
                "fallback": "positive_examples",
            },
        )

    initial_validation = _validate_representation(
        task, initial, student, recorder
    )
    current = initial
    current_text = initial_text
    current_validation = initial_validation
    revisions = []
    protected = {
        row["item_id"] for row in current_validation if row["correct"]
    }
    max_rounds = (
        config.max_teacher_repair_rounds
        if config.max_teacher_repair_rounds is not None
        else config.max_repair_rounds
    )
    max_rounds = min(2, max(0, max_rounds))
    for round_index in range(1, max_rounds + 1):
        failures = [row for row in current_validation if not row["correct"]]
        if not failures:
            break
        repair_prompt = (
            f"{_compiler_prompt(task)}\n\n"
            f"Current representation:\n{current_text}\n\n"
            "The frozen student failed these public positive examples:\n"
            + json.dumps(failures, ensure_ascii=False, indent=2)
            + "\n\nReturn a complete revised JSON representation. Improve the "
            "failed examples without changing the intended task or regressing "
            "previously correct public examples."
        )
        repair_response = _teacher_call(
            teacher,
            COMPILER_SYSTEM_PROMPT,
            repair_prompt,
            config,
            recorder,
            seed=config.seed + round_index,
        )
        teacher_calls += 1
        teacher_input_tokens += repair_response.input_tokens
        teacher_output_tokens += repair_response.output_tokens
        attempt = {
            "round": round_index,
            "text": repair_response.text,
            "accepted": False,
            "parse_error": None,
            "validation": [],
        }
        try:
            proposed = parse_representation(
                repair_response.text,
                task=(
                    task
                    if not config.teacher_structured_outputs
                    else None
                ),
            )
        except ValueError as exc:
            attempt["parse_error"] = str(exc)
            revisions.append(attempt)
            continue
        proposed_validation = _validate_representation(
            task, proposed, student, recorder
        )
        attempt["validation"] = proposed_validation
        proposed_correct = {
            row["item_id"] for row in proposed_validation if row["correct"]
        }
        improved = len(proposed_correct) > len(protected)
        nonregressing = protected <= proposed_correct
        if improved and nonregressing:
            attempt["accepted"] = True
            current = proposed
            current_text = repair_response.text
            current_validation = proposed_validation
            protected = proposed_correct
        revisions.append(attempt)

    failure = None
    if not current_validation or not all(
        row["correct"] for row in current_validation
    ):
        failure = {
            "stage": "positive_example_certificate",
            "message": (
                f"{sum(row['correct'] for row in current_validation)}/"
                f"{len(current_validation)} public positive examples passed"
            ),
            "fallback": "positive_examples",
        }
    return CompilationResult(
        initial=initial,
        final=current,
        initial_text=initial_text,
        final_text=current_text,
        initial_validation=initial_validation,
        final_validation=current_validation,
        revisions=revisions,
        teacher_calls=teacher_calls,
        teacher_input_tokens=teacher_input_tokens,
        teacher_output_tokens=teacher_output_tokens,
        failure=failure,
    )


def _teacher(config: RunConfig) -> TeacherClient:
    if config.teacher_backend == ANTHROPIC_TEACHER:
        return TeacherClient(config.teacher_model)
    return TeacherClient(
        config.teacher_model,
        backend=config.teacher_backend,
        base_url=config.openrouter_base_url,
        timeout_seconds=config.teacher_timeout_seconds,
        provider=config.teacher_openrouter_provider,
    )


def _student(config: RunConfig) -> StudentModel:
    return StudentModel(
        config.student_model,
        revision=config.student_revision,
        backend=config.student_backend,
        ollama_base_url=config.ollama_base_url,
        openrouter_base_url=config.openrouter_base_url,
        openrouter_timeout_seconds=config.student_openrouter_timeout_seconds,
        openrouter_model_name=config.student_openrouter_model,
        openrouter_provider=config.student_openrouter_provider,
        ollama_timeout_seconds=config.ollama_timeout_seconds,
        max_new_tokens=config.max_answer_tokens,
        context_tokens=config.student_context_tokens,
        enable_thinking=config.student_thinking,
        seed=config.student_seed,
    )


def _task_ids(config: RunConfig) -> tuple[str, ...]:
    if config.benchmark_split == "development":
        complete = DEVELOPMENT_TASKS
    elif config.benchmark_split in {"confirmation", "validation"}:
        complete = CONFIRMATION_TASKS
    else:
        raise ValueError(
            "SuperNI benchmark_split must be development or confirmation"
        )
    return complete[: min(config.num_tasks, len(complete))]


def _manifest(tasks: list[SuperNITask], split: str) -> dict:
    payload = {
        "benchmark": "Super-NaturalInstructions",
        "source_repository": "allenai/natural-instructions",
        "source_commit": SUPERNI_COMMIT,
        "split": split,
        "official_instance_order": True,
        "tasks": [
            {
                "task_id": task.task_id,
                "source_sha256": task.source_sha256,
                "sources": list(task.sources),
                "categories": list(task.categories),
            }
            for task in tasks
        ],
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _condition_instructions(
    condition: str,
    compilation: CompilationResult,
    task: SuperNITask,
    query: str,
    ranker: LocalInstructionRanker,
    top_k: int,
) -> tuple[list[str] | None, tuple[SuperNIExample, ...], dict | None]:
    if condition == "raw_definition":
        return None, (), None
    if condition == "positive_examples":
        return None, task.positive_examples, None
    if condition == "parent_one_shot":
        if compilation.initial is None:
            return None, task.positive_examples, {
                "fallback": "positive_examples"
            }
        return (
            render_representation(compilation.initial),
            task.positive_examples,
            None,
        )
    if compilation.failure is not None or compilation.final is None:
        return None, task.positive_examples, {
            "fallback": "positive_examples"
        }
    if condition == "validated_full":
        return (
            render_representation(compilation.final),
            task.positive_examples,
            None,
        )
    if condition == "validated_routed":
        selected, routing = ranker.select(
            query, compilation.final, top_k
        )
        return selected, task.positive_examples, routing
    raise ValueError(f"unknown SuperNI condition: {condition}")


def _evaluate(
    task: SuperNITask,
    compilation: CompilationResult,
    examples: tuple[SuperNIExample, ...],
    student: StudentModel,
    ranker: LocalInstructionRanker,
    config: RunConfig,
    recorder: RunRecorder,
) -> tuple[dict[str, dict], list[dict]]:
    scores = {
        condition: {"correct": 0, "total": 0}
        for condition in CONDITIONS
    }
    predictions = []
    for example in examples:
        response_cache = {}
        for condition in CONDITIONS:
            instructions, public_examples, routing = _condition_instructions(
                condition,
                compilation,
                task,
                example.input_text,
                ranker,
                config.top_k_rules,
            )
            prompt = _student_prompt(
                task.definition,
                example.input_text,
                instructions,
                public_examples,
            )
            cache_key = (STUDENT_SYSTEM_PROMPT, prompt)
            reused = cache_key in response_cache
            if reused:
                response = response_cache[cache_key]
            else:
                response = student.ask(
                    prompt,
                    system=STUDENT_SYSTEM_PROMPT,
                )
                response_cache[cache_key] = response
                recorder.add_student_tokens(
                    response.input_tokens, response.output_tokens
                )
            correct = response_matches(response.text, example.outputs)
            scores[condition]["correct"] += int(correct)
            scores[condition]["total"] += 1
            predictions.append(
                {
                    "task_id": task.task_id,
                    "item_id": example.item_id,
                    "condition": condition,
                    "references": list(example.outputs),
                    "response": response.text,
                    "correct": correct,
                    "routing": routing,
                    "response_reused_for_identical_prompt": reused,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                }
            )
    for row in scores.values():
        row["accuracy"] = (
            row["correct"] / row["total"] if row["total"] else None
        )
    return scores, predictions


def execute(
    config: RunConfig,
    recorder: RunRecorder,
    run_dir: Path,
) -> dict:
    """Compile every task before opening any official final instance."""
    if config.baseline != SUPERNI_COMPILER_MATRIX:
        raise ValueError(
            f"SuperNI requires baseline={SUPERNI_COMPILER_MATRIX}"
        )
    if config.teacher_backend != OPENROUTER_TEACHER:
        raise ValueError("the frozen SuperNI study uses OpenRouter parents")
    if config.student_backend not in {
        OLLAMA_BACKEND,
        OLLAMA_OPENROUTER_BACKEND,
        OPENROUTER_BACKEND,
    }:
        raise ValueError("the SuperNI study requires Ollama or OpenRouter")

    with recorder.stage("load_superni_public_specifications"):
        tasks = [load_task(task_id) for task_id in _task_ids(config)]
        manifest = _manifest(tasks, config.benchmark_split)
        (run_dir / "artifacts" / "suite_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    student = _student(config)
    teacher = _teacher(config)
    ranker = LocalInstructionRanker(
        config.retriever_model,
        revision=config.retriever_revision,
    )
    recorder.set_model_identity("student", student.identity)
    recorder.set_model_identity("teacher", teacher.identity)
    recorder.set_model_identity("router", ranker.identity)

    compilations = {}
    with recorder.stage("compile_validate_and_freeze_all_tasks"):
        for index, task in enumerate(tasks, start=1):
            recorder.log(
                f"SuperNI artifact {index}/{len(tasks)}: {task.task_id}"
            )
            compilation = _compile_task(
                task, teacher, student, config, recorder
            )
            compilations[task.task_id] = compilation
            artifact = {
                "task_id": task.task_id,
                "source_sha256": task.source_sha256,
                "parent_received_instances": False,
                "student_validation_used_instances": False,
                "initial_representation": (
                    compilation.initial.to_dict()
                    if compilation.initial is not None
                    else None
                ),
                "final_representation": (
                    compilation.final.to_dict()
                    if compilation.final is not None
                    else None
                ),
                "initial_raw_text": compilation.initial_text,
                "final_raw_text": compilation.final_text,
                "initial_validation": compilation.initial_validation,
                "final_validation": compilation.final_validation,
                "revisions": compilation.revisions,
                "failure": compilation.failure,
                "teacher_calls": compilation.teacher_calls,
                "teacher_input_tokens": compilation.teacher_input_tokens,
                "teacher_output_tokens": compilation.teacher_output_tokens,
            }
            artifact["artifact_sha256"] = hashlib.sha256(
                json.dumps(artifact, sort_keys=True).encode("utf-8")
            ).hexdigest()
            (
                run_dir / "artifacts" / f"{task.task_id}.json"
            ).write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    task_metrics = []
    predictions = []
    if config.open_final_evaluation:
        with recorder.stage("evaluate_frozen_official_instances"):
            for index, task in enumerate(tasks, start=1):
                recorder.log(
                    f"SuperNI final {index}/{len(tasks)}: {task.task_id}"
                )
                examples = task.instances[: min(config.num_examples, 100)]
                scores, rows = _evaluate(
                    task,
                    compilations[task.task_id],
                    examples,
                    student,
                    ranker,
                    config,
                    recorder,
                )
                task_metrics.append(
                    {
                        "task_id": task.task_id,
                        "conditions": scores,
                    }
                )
                predictions.extend(rows)

    with (run_dir / "artifacts" / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    macro = {}
    for condition in CONDITIONS:
        values = [
            row["conditions"][condition]["accuracy"]
            for row in task_metrics
        ]
        macro[condition] = (
            sum(values) / len(values) if values else None
        )
    bootstrap = {}
    if len(task_metrics) >= 2:
        samples = (
            100_000
            if config.benchmark_split in {"confirmation", "validation"}
            else 10_000
        )
        full = {
            row["task_id"]: row["conditions"]["validated_routed"]["accuracy"]
            for row in task_metrics
        }
        for control in ("raw_definition", "positive_examples"):
            comparator = {
                row["task_id"]: row["conditions"][control]["accuracy"]
                for row in task_metrics
            }
            bootstrap[control] = asdict(
                task_bootstrap_difference(
                    full,
                    comparator,
                    samples=samples,
                    seed=20260727,
                )
            )

    metrics = {
        "dataset": "superni",
        "baseline": config.baseline,
        "benchmark_split": config.benchmark_split,
        "source_commit": SUPERNI_COMMIT,
        "manifest_sha256": manifest["manifest_sha256"],
        "num_tasks": len(tasks),
        "num_examples_per_task": (
            min(config.num_examples, 100)
            if config.open_final_evaluation
            else 0
        ),
        "final_evaluation_opened": config.open_final_evaluation,
        "conditions": list(CONDITIONS),
        "macro_task_accuracy": macro,
        "paired_task_bootstrap": bootstrap,
        "task_metrics": task_metrics,
        "compilation_failures": [
            {
                "task_id": task_id,
                **compilation.failure,
            }
            for task_id, compilation in compilations.items()
            if compilation.failure is not None
        ],
        "teacher_calls": sum(
            compilation.teacher_calls
            for compilation in compilations.values()
        ),
        "parent_received_official_instances": False,
        "student_validation_used_official_instances": False,
        "final_answers_generated_locally": True,
        "inferential_unit": "source-distinct task",
    }
    recorder.set_task_metrics(metrics)
    return metrics
