"""Run task-level and question-level guidance experiments."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from taskcompiler.config import (
    ANTHROPIC_TEACHER,
    DATA_DIR,
    FREEFORM_GUIDANCE,
    GUIDANCE_BASELINES,
    GUIDANCE_MODES,
    LEGACY_SPECIFICATION_GUIDANCE,
    LINEAGE_COMPACT_GUIDANCE,
    LINEAGE_SCRATCHPAD_GUIDANCE,
    LINEAGE_VERIFIED_SCRATCHPAD,
    NONPARENT_VERIFIED_SCRATCHPAD,
    GENERIC_VERIFIED_SCRATCHPAD,
    QUESTION_LEVEL_GUIDANCE,
    RAW_TASK,
    RULEARENA_COMPILER_MATRIX,
    STRUCTURED_GUIDANCE,
    TASK_LEVEL_GUIDANCE,
    TEACHER_ANSWER,
    RunConfig,
)
from taskcompiler.metrics import RunRecorder
from taskcompiler.student import StudentModel
from taskcompiler.teacher import TeacherClient
from taskcompiler.guidance_track.humaneval_exec import (
    extract_code_block,
    inspect_docker_image,
    run_candidate,
)


def _teacher_client(config: RunConfig) -> TeacherClient:
    """Build the configured teacher while preserving the legacy default."""
    if config.teacher_backend == ANTHROPIC_TEACHER:
        return TeacherClient(config.teacher_model)
    return TeacherClient(
        config.teacher_model,
        backend=config.teacher_backend,
        base_url=config.openrouter_base_url,
        timeout_seconds=config.teacher_timeout_seconds,
        provider=config.teacher_openrouter_provider,
    )

STRUCTURED_FIELDS = (
    "task_type",
    "relevant_principles",
    "reasoning_plan",
    "verification_checks",
    "uncertainties",
)
FREEFORM_SYSTEM_PROMPT = (
    "You produce answer-free reasoning guidance for a frozen, smaller language "
    "model. Return 3 to 8 concise bullet points. Describe principles, a process, "
    "verification checks, and uncertainty handling. Never state, imply, or guess "
    "the final answer. Do not include a worked solution."
)

STRUCTURED_SYSTEM_PROMPT = (
    "You produce an answer-free reasoning scaffold for a frozen, smaller language "
    "model. Return only one JSON object with exactly these keys: task_type "
    "(string), relevant_principles (array of strings), reasoning_plan (array of "
    "strings), verification_checks (array of strings), and uncertainties (array "
    "of strings). Never include an answer, conclusion, worked solution, answer "
    "choice, or a field not listed above."
)

TEACHER_ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question directly and accurately. Follow the public task "
    "specification, explain only what is needed, and state material uncertainty. "
    "Do not mention the guidance experiment."
)

CODE_ANSWER_SYSTEM_PROMPT = (
    "Complete the supplied Python function. Return only Python code that can be "
    "placed after the given prompt. Do not use Markdown fences, repeat the "
    "problem, or discuss the solution."
)

STUDENT_SYSTEM_PROMPT = (
    "You are the final answer model. Use any supplied guidance only as a reasoning "
    "process. Solve the question independently, do not discuss the guidance, and "
    "state uncertainty rather than inventing unsupported facts."
)

_FINAL_ANSWER_MARKER = re.compile(
    r"\b(?:(?:the\s+)?(?:final|correct)\s+answer|answer)\s*(?:is|:)",
    re.IGNORECASE,
)


def _teacher_call(
    teacher: TeacherClient,
    system: str,
    prompt: str,
    recorder: RunRecorder,
    max_tokens: int,
    response_format: dict | None = None,
) -> str:
    call_options: dict[str, Any] = {"max_tokens": max_tokens}
    if response_format is not None and getattr(
        teacher,
        "backend",
        None,
    ) == "openrouter":
        call_options["response_format"] = response_format
    response = teacher.call(system, prompt, **call_options)
    recorder.add_teacher_tokens(response.input_tokens, response.output_tokens)
    return response.text.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"structured guidance is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("structured guidance must be a JSON object")
    return value


def _validate_structured_guidance(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != set(STRUCTURED_FIELDS):
        missing = sorted(set(STRUCTURED_FIELDS) - set(value))
        extra = sorted(set(value) - set(STRUCTURED_FIELDS))
        raise ValueError(
            f"structured guidance keys do not match schema; missing={missing}, extra={extra}"
        )
    task_type = value["task_type"]
    if not isinstance(task_type, str) or not task_type.strip():
        raise ValueError("task_type must be a non-empty string")
    cleaned: dict[str, Any] = {"task_type": task_type.strip()}
    for field in STRUCTURED_FIELDS[1:]:
        items = value[field]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{field} must be a non-empty array")
        if not all(isinstance(item, str) and item.strip() for item in items):
            raise ValueError(f"{field} entries must be non-empty strings")
        cleaned[field] = [item.strip() for item in items]
    return cleaned


def _render_structured_guidance(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _validate_answer_free_text(text: str) -> None:
    if not text.strip():
        raise ValueError("guidance is empty")
    if _FINAL_ANSWER_MARKER.search(text):
        raise ValueError("guidance contains an explicit final-answer marker")


def _guidance_prompt(
    task_description: str,
    mode: str,
    question: str | None,
    correction: str | None = None,
) -> str:
    prompt = [
        "Public task specification:",
        task_description.strip(),
        "",
    ]
    if mode == TASK_LEVEL_GUIDANCE:
        prompt.extend(
            [
                "Create reusable guidance from the public specification only.",
                "No evaluation question, reference answer, or benchmark target is available.",
                "Make the reasoning plan operational: state exact transformations, "
                "state transitions, boundary rules, and output checks whenever "
                "the public specification supports them. Avoid generic advice.",
            ]
        )
    elif mode == QUESTION_LEVEL_GUIDANCE:
        if not question:
            raise ValueError("question-level guidance requires a question")
        prompt.extend(
            [
                "Incoming question:",
                question.strip(),
                "",
                "Create a question-specific reasoning scaffold, but do not solve the "
                "question or reveal its conclusion. No reference answer is available.",
            ]
        )
    else:
        raise ValueError(f"unknown guidance mode: {mode}")
    if correction:
        prompt.extend(["", f"Previous output was rejected: {correction}", "Return a corrected artifact."])
    return "\n".join(prompt)


def _generate_guidance(
    teacher: TeacherClient,
    task_description: str,
    recorder: RunRecorder,
    *,
    mode: str = TASK_LEVEL_GUIDANCE,
    style: str = FREEFORM_GUIDANCE,
    question: str | None = None,
    max_tokens: int = 800,
    max_retries: int = 1,
    use_response_format: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    if style == LEGACY_SPECIFICATION_GUIDANCE:
        style = FREEFORM_GUIDANCE
    if style not in {FREEFORM_GUIDANCE, STRUCTURED_GUIDANCE}:
        raise ValueError(f"baseline does not generate guidance: {style}")

    system = (
        STRUCTURED_SYSTEM_PROMPT if style == STRUCTURED_GUIDANCE else FREEFORM_SYSTEM_PROMPT
    )
    correction = None
    for attempt in range(max_retries + 1):
        text = _teacher_call(
            teacher,
            system,
            _guidance_prompt(task_description, mode, question, correction),
            recorder,
            max_tokens=max(64, min(max_tokens, 8000)),
            response_format=(
                _structured_guidance_response_format()
                if style == STRUCTURED_GUIDANCE and use_response_format
                else None
            ),
        )
        try:
            if style == STRUCTURED_GUIDANCE:
                structured = _validate_structured_guidance(_extract_json_object(text))
                rendered = _render_structured_guidance(structured)
                _validate_answer_free_text(rendered)
                return rendered, structured
            _validate_answer_free_text(text)
            return text, None
        except ValueError as exc:
            correction = str(exc)
            if attempt == max_retries:
                raise ValueError(
                    f"teacher failed answer-free guidance validation after "
                    f"{attempt + 1} attempt(s): {exc}"
                ) from exc
    raise AssertionError("unreachable guidance generation state")


def _structured_guidance_response_format() -> dict[str, Any]:
    """Constrain structured guidance to its validated public schema."""
    string_array = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_free_task_guidance",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string"},
                    "relevant_principles": string_array,
                    "reasoning_plan": string_array,
                    "verification_checks": string_array,
                    "uncertainties": string_array,
                },
                "required": list(STRUCTURED_FIELDS),
                "additionalProperties": False,
            },
        },
    }


def _normalise_answer(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _token_f1(predicted: str, reference: str) -> float:
    predicted_tokens = _normalise_answer(predicted).split()
    reference_tokens = _normalise_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _reference_overlap(guidance: str, reference: str | None) -> bool:
    """Detect obvious multi-token overlap with the reference answer."""
    if not reference:
        return False
    normalized_reference = _normalise_answer(reference)
    if len(normalized_reference.split()) < 3:
        return False
    return normalized_reference in _normalise_answer(guidance)


def _student_prompt(
    task_description: str,
    question: str,
    guidance: str,
    *,
    closed_choice: bool,
) -> str:
    parts = ["Public task specification:", task_description.strip(), ""]
    if guidance:
        parts.extend(["Answer-free guidance:", guidance.strip(), ""])
    parts.extend(["Question:", question.strip(), ""])
    if closed_choice:
        parts.append("Return only the option letter in parentheses, for example (A).")
    else:
        parts.append("Provide the final answer. Include concise supporting reasoning when useful.")
    return "\n".join(parts)


def _code_student_prompt(
    task_description: str,
    function_prompt: str,
    guidance: str,
) -> str:
    parts = ["Public task specification:", task_description.strip(), ""]
    if guidance:
        parts.extend(["Answer-free guidance:", guidance.strip(), ""])
    parts.extend(
        [
            "Python function prompt:",
            function_prompt.rstrip(),
            "",
            "Return only the Python continuation that completes this function. "
            "Do not repeat the signature or use Markdown fences.",
        ]
    )
    return "\n".join(parts)


def _completion_text(model_output: str, function_prompt: str) -> str:
    completion = extract_code_block(model_output)
    if completion.startswith(function_prompt):
        completion = completion[len(function_prompt) :]
    lines = completion.splitlines()
    first_nonempty = next((line for line in lines if line.strip()), "")
    if first_nonempty and not first_nonempty.startswith((" ", "\t")):
        if not first_nonempty.startswith(("def ", "async def ", "@")):
            completion = "\n".join(
                f"    {line}" if line.strip() else line for line in lines
            )
    return completion


def _load_bbh_rows(config: RunConfig) -> tuple[list[dict[str, str]], str, bool]:
    dataset = load_from_disk(str(DATA_DIR / "bbh"))["test"]
    rows = list(dataset)
    random.Random(config.seed).shuffle(rows)
    selected = rows[: config.num_examples]
    prepared = [
        {"question": str(row["input"]), "reference": str(row["target"])}
        for row in selected
    ]
    description = (
        "Answer a logical-deduction multiple-choice question by maintaining all "
        "ordering constraints and returning exactly one option letter."
    )
    return prepared, description, True


def _load_open_rows(config: RunConfig) -> tuple[list[dict[str, str | None]], str, bool]:
    questions = [question.strip() for question in config.open_questions if question.strip()]
    if not questions:
        raise ValueError(
            "open_ended guidance runs require at least one non-empty open question"
        )
    references = list(config.reference_answers)
    if references and len(references) != len(config.open_questions):
        raise ValueError(
            "reference_answers must be empty or aligned one-to-one with open_questions"
        )

    prepared: list[dict[str, str | None]] = []
    for index, raw_question in enumerate(config.open_questions):
        question = raw_question.strip()
        if not question:
            continue
        reference = references[index].strip() if references else ""
        prepared.append({"question": question, "reference": reference or None})
    return prepared[: config.num_examples], config.task_description, False


def _run_guidance(
    config: RunConfig,
    recorder: RunRecorder,
    run_dir: Path,
    rows: list[dict[str, str | None]],
    task_description: str,
    closed_choice: bool,
) -> dict[str, Any]:
    if config.guidance_mode not in GUIDANCE_MODES:
        raise ValueError(f"unknown guidance mode: {config.guidance_mode}")
    if config.baseline not in GUIDANCE_BASELINES:
        raise ValueError(f"unknown guidance baseline: {config.baseline}")
    if (
        config.baseline == LEGACY_SPECIFICATION_GUIDANCE
        and config.guidance_mode != TASK_LEVEL_GUIDANCE
    ):
        raise ValueError(
            "specification_only_guidance is a legacy task-level baseline; use "
            "freeform_guidance for question-level runs"
        )

    teacher: TeacherClient | None = None
    student: StudentModel | None = None
    task_guidance = ""
    task_guidance_structured: dict[str, Any] | None = None
    guidance_events: list[dict[str, Any]] = []

    if config.baseline != RAW_TASK:
        teacher = _teacher_client(config)
    if config.baseline != TEACHER_ANSWER:
        student = StudentModel(
            config.student_model,
            revision=config.student_revision,
            max_new_tokens=config.max_answer_tokens,
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
        recorder.set_model_identity(
            "student",
            getattr(
                student,
                "identity",
                {"backend": config.student_backend, "model": config.student_model},
            ),
        )

    if config.baseline in {
        FREEFORM_GUIDANCE,
        STRUCTURED_GUIDANCE,
        LEGACY_SPECIFICATION_GUIDANCE,
    } and config.guidance_mode == TASK_LEVEL_GUIDANCE:
        assert teacher is not None
        with recorder.stage("generate_task_guidance"):
            task_guidance, task_guidance_structured = _generate_guidance(
                teacher,
                task_description,
                recorder,
                mode=TASK_LEVEL_GUIDANCE,
                style=config.baseline,
                max_tokens=config.token_budget,
                max_retries=config.max_guidance_retries,
                use_response_format=config.teacher_structured_outputs,
            )
        guidance_events.append(
            {
                "scope": TASK_LEVEL_GUIDANCE,
                "question_index": None,
                "guidance": task_guidance,
                "structured": task_guidance_structured,
            }
        )

    predictions: list[dict[str, Any]] = []
    overlap_flags = 0
    exact_scores: list[float] = []
    f1_scores: list[float] = []

    with recorder.stage("online_answering"):
        for index, row in enumerate(rows):
            question = str(row["question"])
            reference = row.get("reference")
            guidance = task_guidance
            structured = task_guidance_structured

            if config.baseline in {FREEFORM_GUIDANCE, STRUCTURED_GUIDANCE} and (
                config.guidance_mode == QUESTION_LEVEL_GUIDANCE
            ):
                assert teacher is not None
                guidance, structured = _generate_guidance(
                    teacher,
                    task_description,
                    recorder,
                    mode=QUESTION_LEVEL_GUIDANCE,
                    style=config.baseline,
                    question=question,
                    max_tokens=config.token_budget,
                    max_retries=config.max_guidance_retries,
                    use_response_format=config.teacher_structured_outputs,
                )
                guidance_events.append(
                    {
                        "scope": QUESTION_LEVEL_GUIDANCE,
                        "question_index": index,
                        "guidance": guidance,
                        "structured": structured,
                    }
                )

            overlap = _reference_overlap(guidance, reference)
            overlap_flags += int(overlap)

            if config.baseline == TEACHER_ANSWER:
                assert teacher is not None
                answer_prompt = (
                    f"Public task specification:\n{task_description}\n\n"
                    f"Question:\n{question}"
                )
                answer = _teacher_call(
                    teacher,
                    TEACHER_ANSWER_SYSTEM_PROMPT,
                    answer_prompt,
                    recorder,
                    max_tokens=max(64, min(config.max_answer_tokens, 2000)),
                )
            else:
                assert student is not None
                response = student.ask(
                    _student_prompt(
                        task_description,
                        question,
                        guidance,
                        closed_choice=closed_choice,
                    ),
                    system=STUDENT_SYSTEM_PROMPT,
                )
                recorder.add_student_tokens(response.input_tokens, response.output_tokens)
                answer = response.text.strip()

            exact = None
            token_f1 = None
            if reference:
                exact = float(_normalise_answer(answer) == _normalise_answer(reference))
                token_f1 = _token_f1(answer, reference)
                exact_scores.append(exact)
                f1_scores.append(token_f1)

            predictions.append(
                {
                    "question_index": index,
                    "question": question,
                    "guidance_scope": (
                        None if config.baseline in {RAW_TASK, TEACHER_ANSWER}
                        else config.guidance_mode
                    ),
                    "guidance": guidance or None,
                    "answer": answer,
                    "reference": reference,
                    "exact_match": exact,
                    "token_f1": token_f1,
                    "obvious_reference_overlap_in_guidance": overlap,
                }
            )

    artifact = {
        "schema_version": 1,
        "dataset": config.dataset,
        "baseline": config.baseline,
        "guidance_mode": config.guidance_mode,
        "task_description": task_description,
        "task_guidance": task_guidance or None,
        "task_guidance_structured": task_guidance_structured,
        "guidance_validation": (
            "schema_and_marker_validated"
            if config.baseline == STRUCTURED_GUIDANCE
            else "marker_validated"
            if config.baseline in {FREEFORM_GUIDANCE, LEGACY_SPECIFICATION_GUIDANCE}
            else "not_applicable"
        ),
        "teacher_never_received_references": True,
    }
    artifacts_dir = run_dir / "artifacts"
    (artifacts_dir / "instruction_bundle.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "guidance_events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in guidance_events),
        encoding="utf-8",
    )
    (artifacts_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in predictions),
        encoding="utf-8",
    )

    metrics: dict[str, Any] = {
        "dataset": config.dataset,
        "baseline": config.baseline,
        "guidance_mode": config.guidance_mode,
        "condition_class": (
            "direct_teacher_upper_comparator"
            if config.baseline == TEACHER_ANSWER
            else "no_teacher_guidance"
            if config.baseline == RAW_TASK
            else "offline_reusable_guidance"
            if config.guidance_mode == TASK_LEVEL_GUIDANCE
            else "online_question_specific_guidance"
        ),
        "num_examples": len(predictions),
        "num_references": len(exact_scores),
        "guidance_events": len(guidance_events),
        "obvious_reference_overlap_flags": overlap_flags,
        "teacher_access_to_questions": (
            config.baseline == TEACHER_ANSWER
            or (
                config.guidance_mode == QUESTION_LEVEL_GUIDANCE
                and config.baseline in {FREEFORM_GUIDANCE, STRUCTURED_GUIDANCE}
            )
        ),
        "student_used_for_final_answer": config.baseline != TEACHER_ANSWER,
        "online_teacher_access_to_questions": (
            config.guidance_mode == QUESTION_LEVEL_GUIDANCE
            and config.baseline in {FREEFORM_GUIDANCE, STRUCTURED_GUIDANCE}
        ),
    }
    if exact_scores:
        metrics["exact_match"] = sum(exact_scores) / len(exact_scores)
        metrics["token_f1"] = sum(f1_scores) / len(f1_scores)
        if closed_choice:
            metrics["accuracy"] = metrics["exact_match"]
    else:
        metrics["evaluation_status"] = (
            "answers generated; no references supplied, so quality requires "
            "blinded rubric, external judge, or human evaluation"
        )
    return metrics


def _run_humaneval(config: RunConfig, recorder: RunRecorder, run_dir: Path) -> dict:
    if not config.enable_docker_sandbox:
        raise RuntimeError(
            "HumanEval requires enable_docker_sandbox=true. No unsafe local "
            "execution fallback is available."
        )
    if config.guidance_mode not in GUIDANCE_MODES:
        raise ValueError(f"unknown guidance mode: {config.guidance_mode}")
    if config.baseline not in GUIDANCE_BASELINES:
        raise ValueError(f"unknown guidance baseline: {config.baseline}")
    if (
        config.baseline == LEGACY_SPECIFICATION_GUIDANCE
        and config.guidance_mode != TASK_LEVEL_GUIDANCE
    ):
        raise ValueError(
            "specification_only_guidance is a legacy task-level baseline"
        )

    with recorder.stage("docker_sandbox_preflight"):
        docker_image_id = inspect_docker_image(config.docker_image)

    task_description = (
        "Complete a Python function from its signature and docstring. Return a "
        "correct deterministic implementation that handles edge cases and uses "
        "only the Python standard library."
    )
    teacher: TeacherClient | None = None
    student: StudentModel | None = None
    task_guidance = ""
    task_structured: dict[str, Any] | None = None
    guidance_events: list[dict[str, Any]] = []

    if config.baseline != RAW_TASK:
        teacher = _teacher_client(config)
    if config.baseline != TEACHER_ANSWER:
        student = StudentModel(
            config.student_model,
            revision=config.student_revision,
            max_new_tokens=config.max_answer_tokens,
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
        recorder.set_model_identity(
            "student",
            getattr(
                student,
                "identity",
                {"backend": config.student_backend, "model": config.student_model},
            ),
        )

    # Task-level artifacts are built before HumanEval data is loaded.
    if config.baseline in {
        FREEFORM_GUIDANCE,
        STRUCTURED_GUIDANCE,
        LEGACY_SPECIFICATION_GUIDANCE,
    } and config.guidance_mode == TASK_LEVEL_GUIDANCE:
        assert teacher is not None
        with recorder.stage("generate_task_guidance"):
            task_guidance, task_structured = _generate_guidance(
                teacher,
                task_description,
                recorder,
                mode=TASK_LEVEL_GUIDANCE,
                style=config.baseline,
                max_tokens=config.token_budget,
                max_retries=config.max_guidance_retries,
                use_response_format=config.teacher_structured_outputs,
            )
        guidance_events.append(
            {
                "scope": TASK_LEVEL_GUIDANCE,
                "question_index": None,
                "guidance": task_guidance,
                "structured": task_structured,
            }
        )

    with recorder.stage("load_humaneval_data"):
        dataset = load_from_disk(str(DATA_DIR / "humaneval"))
        split = dataset["test"] if hasattr(dataset, "keys") and "test" in dataset else dataset
        rows = list(split)
        random.Random(config.seed).shuffle(rows)
        rows = rows[: config.num_examples]

    predictions: list[dict[str, Any]] = []
    passed = 0
    timed_out = 0

    with recorder.stage("generate_and_execute_code"):
        for index, row in enumerate(rows):
            function_prompt = str(row["prompt"])
            guidance = task_guidance
            structured = task_structured

            if config.baseline in {FREEFORM_GUIDANCE, STRUCTURED_GUIDANCE} and (
                config.guidance_mode == QUESTION_LEVEL_GUIDANCE
            ):
                assert teacher is not None
                guidance, structured = _generate_guidance(
                    teacher,
                    task_description,
                    recorder,
                    mode=QUESTION_LEVEL_GUIDANCE,
                    style=config.baseline,
                    question=function_prompt,
                    max_tokens=config.token_budget,
                    max_retries=config.max_guidance_retries,
                    use_response_format=config.teacher_structured_outputs,
                )
                guidance_events.append(
                    {
                        "scope": QUESTION_LEVEL_GUIDANCE,
                        "question_index": index,
                        "task_id": row.get("task_id"),
                        "guidance": guidance,
                        "structured": structured,
                    }
                )

            if config.baseline == TEACHER_ANSWER:
                assert teacher is not None
                raw_completion = _teacher_call(
                    teacher,
                    CODE_ANSWER_SYSTEM_PROMPT,
                    f"Python function prompt:\n{function_prompt}",
                    recorder,
                    max_tokens=max(64, min(config.max_answer_tokens, 2000)),
                )
            else:
                assert student is not None
                response = student.ask(
                    _code_student_prompt(
                        task_description,
                        function_prompt,
                        guidance,
                    ),
                    system=CODE_ANSWER_SYSTEM_PROMPT,
                )
                recorder.add_student_tokens(response.input_tokens, response.output_tokens)
                raw_completion = response.text

            completion = _completion_text(raw_completion, function_prompt)
            execution = run_candidate(
                function_prompt,
                completion,
                str(row["test"]),
                str(row["entry_point"]),
                enabled=True,
                image=config.docker_image,
                timeout_seconds=config.docker_timeout_seconds,
                memory_mb=config.docker_memory_mb,
                cpus=config.docker_cpus,
                pids_limit=config.docker_pids_limit,
            )
            passed += int(execution.passed)
            timed_out += int(execution.timed_out)
            predictions.append(
                {
                    "question_index": index,
                    "task_id": row.get("task_id"),
                    "prompt": function_prompt,
                    "guidance_scope": (
                        None if config.baseline in {RAW_TASK, TEACHER_ANSWER}
                        else config.guidance_mode
                    ),
                    "guidance": guidance or None,
                    "completion": completion,
                    "passed": execution.passed,
                    "timed_out": execution.timed_out,
                    "error": execution.error,
                    "docker_image": execution.image,
                }
            )

    artifacts_dir = run_dir / "artifacts"
    (artifacts_dir / "instruction_bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "humaneval",
                "baseline": config.baseline,
                "guidance_mode": config.guidance_mode,
                "task_description": task_description,
                "task_guidance": task_guidance or None,
                "task_guidance_structured": task_structured,
                "docker_sandbox": {
                    "enabled": True,
                    "image": config.docker_image,
                    "image_id": docker_image_id,
                    "network": "none",
                    "read_only": True,
                    "memory_mb": config.docker_memory_mb,
                    "cpus": config.docker_cpus,
                    "pids_limit": config.docker_pids_limit,
                    "timeout_seconds": config.docker_timeout_seconds,
                },
                "teacher_never_received_tests_or_canonical_solutions": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "guidance_events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in guidance_events),
        encoding="utf-8",
    )
    (artifacts_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in predictions),
        encoding="utf-8",
    )

    return {
        "dataset": "humaneval",
        "baseline": config.baseline,
        "guidance_mode": config.guidance_mode,
        "condition_class": (
            "direct_teacher_upper_comparator"
            if config.baseline == TEACHER_ANSWER
            else "no_teacher_guidance"
            if config.baseline == RAW_TASK
            else "offline_reusable_guidance"
            if config.guidance_mode == TASK_LEVEL_GUIDANCE
            else "online_question_specific_guidance"
        ),
        "num_examples": len(predictions),
        "pass_at_1": passed / len(predictions) if predictions else 0.0,
        "passed": passed,
        "timed_out": timed_out,
        "guidance_events": len(guidance_events),
        "teacher_received_hidden_tests": False,
        "docker_sandbox_enabled": True,
        "docker_image": config.docker_image,
        "docker_image_id": docker_image_id,
    }


def execute(config: RunConfig, recorder: RunRecorder, run_dir: Path) -> dict:
    if config.dataset == "slr_bench":
        from taskcompiler.guidance_track.slr_bench import execute as execute_slr

        return execute_slr(config, recorder, run_dir)
    if config.dataset == "superni":
        from taskcompiler.guidance_track.superni import execute as execute_superni

        return execute_superni(config, recorder, run_dir)
    if config.dataset == "rulearena":
        if config.baseline == RULEARENA_COMPILER_MATRIX:
            from taskcompiler.guidance_track.rulearena_compiler import (
                execute as execute_rulearena_compiler,
            )

            return execute_rulearena_compiler(config, recorder, run_dir)
        from taskcompiler.guidance_track.rulearena import execute as execute_rulearena

        return execute_rulearena(config, recorder, run_dir)
    if config.dataset == "bbh" and config.baseline in {
        "raw_task",
        "zero_shot_cot",
        "few_shot_cot",
        "textgrad_published_prompt",
        "textgrad_tgd",
        "gepa",
        "structured_guidance",
        LINEAGE_COMPACT_GUIDANCE,
        LINEAGE_SCRATCHPAD_GUIDANCE,
        LINEAGE_VERIFIED_SCRATCHPAD,
        NONPARENT_VERIFIED_SCRATCHPAD,
        GENERIC_VERIFIED_SCRATCHPAD,
    }:
        from taskcompiler.guidance_track.bbh import execute as execute_bbh

        return execute_bbh(config, recorder, run_dir)
    if config.dataset == "humaneval":
        metrics = _run_humaneval(config, recorder, run_dir)
        recorder.set_task_metrics(metrics)
        return metrics

    with recorder.stage("load_data"):
        if config.dataset == "bbh":
            rows, task_description, closed_choice = _load_bbh_rows(config)
        elif config.dataset == "open_ended":
            rows, task_description, closed_choice = _load_open_rows(config)
        else:
            raise ValueError(f"unknown guidance_track dataset: {config.dataset}")

    metrics = _run_guidance(
        config,
        recorder,
        run_dir,
        rows,
        task_description,
        closed_choice,
    )
    recorder.set_task_metrics(metrics)
    return metrics
