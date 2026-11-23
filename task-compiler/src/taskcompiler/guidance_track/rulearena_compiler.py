"""Compile and verify reusable task knowledge for RuleArena."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from taskcompiler.config import (
    ANTHROPIC_TEACHER,
    OLLAMA_BACKEND,
    OLLAMA_OPENROUTER_BACKEND,
    OPENROUTER_BACKEND,
    OPENROUTER_TEACHER,
    RULEARENA_COMPILER_MATRIX,
    RULEARENA_ONE_SHOT,
    RULEARENA_ZERO_SHOT,
    RunConfig,
)
from taskcompiler.guidance_track.rulearena import (
    RULEARENA_COMMIT,
    RULEARENA_ROOT,
    RuleArenaExample,
    _airline_target,
    load_examples,
    render_prompt,
    score_response,
)
from taskcompiler.metrics import RunRecorder
from taskcompiler.student import StudentModel
from taskcompiler.teacher import TeacherClient, TeacherResponse


BUILD_SEED = 2026072701
AUDIT_SEED = 2026072702
SCENARIOS_PER_SPLIT = 12
REPRESENTATION_FIELDS = (
    "objective",
    "output_contract",
    "fare_and_route_rules",
    "allowance_rules",
    "checked_bag_fee_rules",
    "overweight_oversize_rules",
    "exception_and_precedence_rules",
    "calculation_checks",
)
CONDITIONS = (
    RULEARENA_ZERO_SHOT,
    RULEARENA_ONE_SHOT,
    "parent_one_shot_compiled",
    "validated_compiled_full",
    "validated_compiled_routed",
)
COMPILER_SYSTEM_PROMPT = (
    "Compile public airline policies into a compact, reusable, exact rule "
    "manual for a frozen smaller language model. Return only the requested JSON "
    "object. Preserve every numeric threshold, fee, region, cabin class, free "
    "allowance, exception, and precedence rule. Never simplify by inventing a "
    "default or omitting a boundary."
)


@dataclass(frozen=True)
class AirlineRepresentation:
    objective: str
    output_contract: str
    fare_and_route_rules: tuple[str, ...]
    allowance_rules: tuple[str, ...]
    checked_bag_fee_rules: tuple[str, ...]
    overweight_oversize_rules: tuple[str, ...]
    exception_and_precedence_rules: tuple[str, ...]
    calculation_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "output_contract": self.output_contract,
            "fare_and_route_rules": list(self.fare_and_route_rules),
            "allowance_rules": list(self.allowance_rules),
            "checked_bag_fee_rules": list(self.checked_bag_fee_rules),
            "overweight_oversize_rules": list(
                self.overweight_oversize_rules
            ),
            "exception_and_precedence_rules": list(
                self.exception_and_precedence_rules
            ),
            "calculation_checks": list(self.calculation_checks),
        }


@dataclass
class AirlineCompilation:
    initial: AirlineRepresentation | None
    final: AirlineRepresentation | None
    initial_text: str
    final_text: str
    initial_build: list[dict]
    final_build: list[dict]
    repairs: list[dict]
    teacher_calls: int
    teacher_input_tokens: int
    teacher_output_tokens: int
    failure: dict | None


class AirlineInstructionRanker:
    """Rank optional airline rules with a frozen local embedding model."""

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
        representation: AirlineRepresentation,
        top_k: int,
    ) -> tuple[list[str], dict]:
        mandatory = _mandatory_lines(representation)
        optional = _optional_lines(representation)
        if top_k >= len(optional):
            selected = list(range(len(optional)))
        else:
            key = tuple(optional)
            vectors = self._cache.get(key)
            if vectors is None:
                vectors = self.model.encode(
                    optional, normalize_embeddings=True
                )
                self._cache[key] = vectors
            query_vector = self.model.encode(
                query, normalize_embeddings=True
            )
            scores = vectors @ query_vector
            ranked = sorted(
                range(len(optional)),
                key=lambda index: float(scores[index]),
                reverse=True,
            )
            selected = sorted(ranked[: max(0, top_k)])
        return mandatory + [optional[index] for index in selected], {
            "top_k": top_k,
            "optional_count": len(optional),
            "selected_optional_indices": selected,
        }


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    cleaned = tuple(item.strip() for item in value if isinstance(item, str))
    if len(cleaned) != len(value) or any(not item for item in cleaned):
        raise ValueError(f"{field} must contain non-empty strings")
    return cleaned


def parse_representation(text: str) -> AirlineRepresentation:
    """Parse one strict parent-produced airline rule representation."""
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
            f"airline representation is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(value, dict) or set(value) != set(
        REPRESENTATION_FIELDS
    ):
        raise ValueError("airline representation keys do not match the schema")
    for field in ("objective", "output_contract"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    return AirlineRepresentation(
        objective=value["objective"].strip(),
        output_contract=value["output_contract"].strip(),
        fare_and_route_rules=_string_list(
            value["fare_and_route_rules"], "fare_and_route_rules"
        ),
        allowance_rules=_string_list(
            value["allowance_rules"], "allowance_rules"
        ),
        checked_bag_fee_rules=_string_list(
            value["checked_bag_fee_rules"], "checked_bag_fee_rules"
        ),
        overweight_oversize_rules=_string_list(
            value["overweight_oversize_rules"],
            "overweight_oversize_rules",
        ),
        exception_and_precedence_rules=_string_list(
            value["exception_and_precedence_rules"],
            "exception_and_precedence_rules",
        ),
        calculation_checks=_string_list(
            value["calculation_checks"], "calculation_checks"
        ),
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
            "name": "airline_task_knowledge",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string", "minLength": 1},
                    "output_contract": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "fare_and_route_rules": array,
                    "allowance_rules": array,
                    "checked_bag_fee_rules": array,
                    "overweight_oversize_rules": array,
                    "exception_and_precedence_rules": array,
                    "calculation_checks": array,
                },
                "required": list(REPRESENTATION_FIELDS),
                "additionalProperties": False,
            },
        },
    }


def _load_airline_generator():
    path = RULEARENA_ROOT / "airline" / "gen_questions.py"
    spec = importlib.util.spec_from_file_location(
        "_taskcompiler_rulearena_airline_generator",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the official airline generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_scenarios(
    seed: int,
    count: int,
    *,
    excluded_prompts: set[str] | None = None,
) -> list[RuleArenaExample]:
    """Generate unique Airline cases with the official source generator."""
    excluded = set(excluded_prompts or ())
    generator = _load_airline_generator()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    examples = []
    generation_index = 0
    try:
        while len(examples) < count:
            prompt, info = generator.gen_question(0)
            generation_index += 1
            if prompt in excluded:
                continue
            source = {"prompt": prompt, "info": info}
            digest = hashlib.sha256(
                json.dumps(source, sort_keys=True).encode("utf-8")
            ).hexdigest()
            examples.append(
                RuleArenaExample(
                    item_id=(
                        f"airline-generated-{seed}-{generation_index:04d}-"
                        f"{digest[:10]}"
                    ),
                    domain="airline",
                    complexity=0,
                    source_index=generation_index,
                    source=source,
                    target=_airline_target(source),
                )
            )
            excluded.add(prompt)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return examples


def _mandatory_lines(
    representation: AirlineRepresentation,
) -> list[str]:
    return [
        f"Objective: {representation.objective}",
        f"Output contract: {representation.output_contract}",
        *[
            f"Calculation check: {rule}"
            for rule in representation.calculation_checks
        ],
    ]


def _optional_lines(
    representation: AirlineRepresentation,
) -> list[str]:
    return [
        *[
            f"Fare and route rule: {rule}"
            for rule in representation.fare_and_route_rules
        ],
        *[
            f"Allowance rule: {rule}"
            for rule in representation.allowance_rules
        ],
        *[
            f"Checked bag fee rule: {rule}"
            for rule in representation.checked_bag_fee_rules
        ],
        *[
            f"Weight and size rule: {rule}"
            for rule in representation.overweight_oversize_rules
        ],
        *[
            f"Exception and precedence rule: {rule}"
            for rule in representation.exception_and_precedence_rules
        ],
    ]


def render_representation(
    representation: AirlineRepresentation,
) -> list[str]:
    """Render a deterministic compact airline manual."""
    return _mandatory_lines(representation) + _optional_lines(representation)


def _compiled_prompt(example: RuleArenaExample, rules: list[str]) -> str:
    return (
        "Use this compiled American Airlines task knowledge:\n\n"
        + "\n".join(f"- {rule}" for rule in rules)
        + "\n\nPassenger request:\n"
        + example.source["prompt"]
        + "\n\nCompute the total cost, including the flight ticket and every "
        "applicable baggage fee. Show enough arithmetic to verify the result "
        'and end with exactly "The total cost is $xxx."'
    )


def _compiler_prompt(reference_rules: str) -> str:
    return (
        "Public task: compute a passenger's total American Airlines trip cost "
        "from the ticket price, route, cabin class, bags, and official policies."
        "\n\nOfficial public policies:\n"
        f"{reference_rules}\n\n"
        "Produce a compact but complete reusable manual. Keep every table entry "
        "as an explicit textual rule. State how route direction and region are "
        "chosen, which items qualify as carry-on or personal items, how free "
        "bags are assigned, how checked-bag positions are counted, how cabin "
        "benefits alter allowances, how overweight and oversize bands apply, "
        "when fees combine or only the larger fee applies, and every embargo or "
        "maximum. End with ordered arithmetic and output checks."
    )


def _teacher_call(
    teacher: TeacherClient,
    config: RunConfig,
    recorder: RunRecorder,
    prompt: str,
    *,
    seed: int,
) -> TeacherResponse:
    response = teacher.call(
        COMPILER_SYSTEM_PROMPT,
        prompt,
        max_tokens=max(1200, min(config.token_budget, 5000)),
        response_format=(
            _response_format()
            if config.teacher_backend == OPENROUTER_TEACHER
            else None
        ),
        seed=seed,
    )
    recorder.add_teacher_tokens(
        response.input_tokens, response.output_tokens
    )
    return response


def _ask(
    student: StudentModel,
    example: RuleArenaExample,
    condition: str,
    representation: AirlineRepresentation | None,
    ranker: AirlineInstructionRanker,
    top_k: int,
) -> tuple[Any, dict | None, str]:
    if condition in {RULEARENA_ZERO_SHOT, RULEARENA_ONE_SHOT}:
        system, prompt = render_prompt(example, condition)
        routing = None
    else:
        system, _ = render_prompt(example, RULEARENA_ZERO_SHOT)
        if representation is None:
            system, prompt = render_prompt(
                example, RULEARENA_ONE_SHOT
            )
            routing = {"fallback": RULEARENA_ONE_SHOT}
        else:
            if condition == "validated_compiled_routed":
                rules, routing = ranker.select(
                    example.source["prompt"],
                    representation,
                    top_k,
                )
            else:
                rules = render_representation(representation)
                routing = None
            prompt = _compiled_prompt(example, rules)
    return student.ask(prompt, system=system), routing, prompt


def _scoreable(response: str) -> bool:
    return bool(
        re.search(
            r"The total cost is\s*\$[\d,]+(?:\.\d+)?",
            response.replace("**", ""),
            flags=re.IGNORECASE,
        )
    )


def _evaluate_condition(
    examples: list[RuleArenaExample],
    condition: str,
    representation: AirlineRepresentation | None,
    student: StudentModel,
    ranker: AirlineInstructionRanker,
    config: RunConfig,
    recorder: RunRecorder,
) -> list[dict]:
    rows = []
    for example in examples:
        response, routing, prompt = _ask(
            student,
            example,
            condition,
            representation,
            ranker,
            config.top_k_rules,
        )
        recorder.add_student_tokens(
            response.input_tokens, response.output_tokens
        )
        rows.append(
            {
                "item_id": example.item_id,
                "target": example.target,
                "response": response.text,
                "correct": score_response(example, response.text),
                "scoreable": _scoreable(response.text),
                "routing": routing,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        )
    return rows


def _accuracy(rows: list[dict]) -> float:
    return sum(row["correct"] for row in rows) / len(rows)


def _compile(
    reference_rules: str,
    build: list[RuleArenaExample],
    teacher: TeacherClient,
    student: StudentModel,
    ranker: AirlineInstructionRanker,
    config: RunConfig,
    recorder: RunRecorder,
) -> AirlineCompilation:
    response = _teacher_call(
        teacher,
        config,
        recorder,
        _compiler_prompt(reference_rules),
        seed=config.seed,
    )
    calls = 1
    input_tokens = response.input_tokens
    output_tokens = response.output_tokens
    try:
        initial = parse_representation(response.text)
    except ValueError as exc:
        return AirlineCompilation(
            initial=None,
            final=None,
            initial_text=response.text,
            final_text=response.text,
            initial_build=[],
            final_build=[],
            repairs=[],
            teacher_calls=calls,
            teacher_input_tokens=input_tokens,
            teacher_output_tokens=output_tokens,
            failure={
                "stage": "representation_schema",
                "message": str(exc),
                "fallback": RULEARENA_ONE_SHOT,
            },
        )
    initial_build = _evaluate_condition(
        build,
        "parent_one_shot_compiled",
        initial,
        student,
        ranker,
        config,
        recorder,
    )
    current = initial
    current_text = response.text
    current_build = initial_build
    protected = {
        row["item_id"] for row in current_build if row["correct"]
    }
    repairs = []
    max_rounds = min(
        2,
        max(
            0,
            config.max_teacher_repair_rounds
            if config.max_teacher_repair_rounds is not None
            else config.max_repair_rounds,
        ),
    )
    for round_index in range(1, max_rounds + 1):
        failures = [row for row in current_build if not row["correct"]]
        if not failures:
            break
        failure_payload = []
        by_id = {example.item_id: example for example in build}
        for row in failures:
            example = by_id[row["item_id"]]
            failure_payload.append(
                {
                    "item_id": row["item_id"],
                    "passenger_request": example.source["prompt"],
                    "official_total": example.target,
                    "student_response": row["response"],
                }
            )
        repair_prompt = (
            f"{_compiler_prompt(reference_rules)}\n\n"
            f"Current representation:\n{current_text}\n\n"
            "Generated build failures, independently scored by the official "
            "program:\n"
            + json.dumps(failure_payload, ensure_ascii=False, indent=2)
            + "\n\nReturn a complete corrected JSON manual. Generalize each "
            "correction to its source policy. Do not add passenger-specific "
            "rules or memorize a total."
        )
        repair_response = _teacher_call(
            teacher,
            config,
            recorder,
            repair_prompt,
            seed=config.seed + round_index,
        )
        calls += 1
        input_tokens += repair_response.input_tokens
        output_tokens += repair_response.output_tokens
        attempt = {
            "round": round_index,
            "raw_text": repair_response.text,
            "accepted": False,
            "parse_error": None,
            "build": [],
        }
        try:
            proposed = parse_representation(repair_response.text)
        except ValueError as exc:
            attempt["parse_error"] = str(exc)
            repairs.append(attempt)
            continue
        proposed_build = _evaluate_condition(
            build,
            "validated_compiled_full",
            proposed,
            student,
            ranker,
            config,
            recorder,
        )
        attempt["build"] = proposed_build
        proposed_correct = {
            row["item_id"] for row in proposed_build if row["correct"]
        }
        if (
            len(proposed_correct) > len(protected)
            and protected <= proposed_correct
        ):
            attempt["accepted"] = True
            current = proposed
            current_text = repair_response.text
            current_build = proposed_build
            protected = proposed_correct
        repairs.append(attempt)
    return AirlineCompilation(
        initial=initial,
        final=current,
        initial_text=response.text,
        final_text=current_text,
        initial_build=initial_build,
        final_build=current_build,
        repairs=repairs,
        teacher_calls=calls,
        teacher_input_tokens=input_tokens,
        teacher_output_tokens=output_tokens,
        failure=None,
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


def _scenario_manifest(
    build: list[RuleArenaExample],
    audit: list[RuleArenaExample],
) -> dict:
    def rows(examples):
        return [
            {
                "item_id": example.item_id,
                "prompt": example.source["prompt"],
                "info": example.source["info"],
                "target": example.target,
                "sha256": hashlib.sha256(
                    json.dumps(
                        example.source, sort_keys=True
                    ).encode("utf-8")
                ).hexdigest(),
            }
            for example in examples
        ]

    payload = {
        "benchmark": "RuleArena",
        "source_commit": RULEARENA_COMMIT,
        "domain": "airline",
        "complexity": 0,
        "build_seed": BUILD_SEED,
        "audit_seed": AUDIT_SEED,
        "build": rows(build),
        "audit": rows(audit),
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def execute(
    config: RunConfig,
    recorder: RunRecorder,
    run_dir: Path,
) -> dict:
    """Run the frozen generated-scenario Airline development matrix."""
    if config.baseline != RULEARENA_COMPILER_MATRIX:
        raise ValueError(
            f"expected baseline={RULEARENA_COMPILER_MATRIX}"
        )
    if config.benchmark_task != "airline" or config.benchmark_complexity != 0:
        raise ValueError("the frozen pilot is Airline complexity 0")
    if config.teacher_backend != OPENROUTER_TEACHER:
        raise ValueError("the frozen pilot uses an OpenRouter parent")
    if config.student_backend not in {
        OLLAMA_BACKEND,
        OLLAMA_OPENROUTER_BACKEND,
        OPENROUTER_BACKEND,
    }:
        raise ValueError("the pilot requires Ollama or OpenRouter")

    with recorder.stage("generate_and_verify_source_scenarios"):
        official = [
            *load_examples("airline", 0),
            *load_examples("airline", 1),
            *load_examples("airline", 2),
        ]
        excluded = {
            example.source["prompt"] for example in official
        }
        build = generate_scenarios(
            BUILD_SEED,
            SCENARIOS_PER_SPLIT,
            excluded_prompts=excluded,
        )
        excluded.update(example.source["prompt"] for example in build)
        audit = generate_scenarios(
            AUDIT_SEED,
            SCENARIOS_PER_SPLIT,
            excluded_prompts=excluded,
        )
        manifest = _scenario_manifest(build, audit)
        (
            run_dir / "artifacts" / "generated_scenarios.json"
        ).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    student = _student(config)
    teacher = _teacher(config)
    ranker = AirlineInstructionRanker(
        config.retriever_model,
        revision=config.retriever_revision,
    )
    recorder.set_model_identity("student", student.identity)
    recorder.set_model_identity("teacher", teacher.identity)
    recorder.set_model_identity("router", ranker.identity)
    reference_rules = (
        RULEARENA_ROOT / "airline" / "reference_rules.txt"
    ).read_text(encoding="utf-8")

    with recorder.stage("compile_and_validate_on_generated_build"):
        compilation = _compile(
            reference_rules,
            build,
            teacher,
            student,
            ranker,
            config,
            recorder,
        )
        artifact = {
            "domain": "airline",
            "source_commit": RULEARENA_COMMIT,
            "parent_received_build_scenarios": True,
            "parent_received_audit_scenarios": False,
            "parent_received_official_problems": False,
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
            "initial_build": compilation.initial_build,
            "final_build": compilation.final_build,
            "repairs": compilation.repairs,
            "failure": compilation.failure,
            "teacher_calls": compilation.teacher_calls,
            "teacher_input_tokens": compilation.teacher_input_tokens,
            "teacher_output_tokens": compilation.teacher_output_tokens,
        }
        artifact["artifact_sha256"] = hashlib.sha256(
            json.dumps(artifact, sort_keys=True).encode("utf-8")
        ).hexdigest()
        (run_dir / "artifacts" / "airline_compilation.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    phase_results = {"build": {}, "audit": {}}
    predictions = []
    with recorder.stage("evaluate_generated_conditions"):
        for phase, examples in (("build", build), ("audit", audit)):
            for condition in CONDITIONS:
                if condition == "parent_one_shot_compiled":
                    representation = compilation.initial
                elif condition.startswith("validated_compiled"):
                    representation = compilation.final
                else:
                    representation = None
                rows = _evaluate_condition(
                    examples,
                    condition,
                    representation,
                    student,
                    ranker,
                    config,
                    recorder,
                )
                phase_results[phase][condition] = {
                    "accuracy": _accuracy(rows),
                    "correct": sum(row["correct"] for row in rows),
                    "scoreable": sum(row["scoreable"] for row in rows),
                    "total": len(rows),
                }
                predictions.extend(
                    {
                        "phase": phase,
                        "condition": condition,
                        **row,
                    }
                    for row in rows
                )

    with (run_dir / "artifacts" / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit_results = phase_results["audit"]
    strongest_official = max(
        audit_results[RULEARENA_ZERO_SHOT]["accuracy"],
        audit_results[RULEARENA_ONE_SHOT]["accuracy"],
    )
    full_accuracy = audit_results["validated_compiled_full"]["accuracy"]
    routed = audit_results["validated_compiled_routed"]
    gate = {
        "routed_exceeds_both_official": (
            routed["accuracy"] > strongest_official
        ),
        "routed_gain_at_least_10_points": (
            routed["accuracy"] - strongest_official >= 0.10
        ),
        "full_not_below_strongest_official": (
            full_accuracy >= strongest_official
        ),
        "routed_scoreable_at_least_10": routed["scoreable"] >= 10,
        "artifact_schema_valid": compilation.final is not None,
        "audit_hidden_from_parent": True,
    }
    gate["passed"] = all(gate.values())
    metrics = {
        "dataset": "rulearena",
        "study": "verifier_backed_compiler_development",
        "domain": "airline",
        "complexity": 0,
        "official_problems_evaluated": 0,
        "generated_build_scenarios": len(build),
        "generated_audit_scenarios": len(audit),
        "manifest_sha256": manifest["manifest_sha256"],
        "conditions": list(CONDITIONS),
        "results": phase_results,
        "development_gate": gate,
        "teacher_calls": compilation.teacher_calls,
        "compilation_failure": compilation.failure,
        "parent_received_audit_scenarios": False,
        "parent_received_official_problems": False,
        "final_answers_generated_locally": True,
    }
    recorder.set_task_metrics(metrics)
    return metrics
