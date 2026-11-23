"""Adapt APE instruction search to exact-label policy tasks."""

from __future__ import annotations

import json
from pathlib import Path

from taskcompiler.config import RunConfig
from taskcompiler.metrics import RunRecorder
from taskcompiler.student import StudentModel
from taskcompiler.teacher import TeacherClient
from taskcompiler.translate import normalise_student_label


APE_SOURCE = "keirp/automatic_prompt_engineer"


def _candidate_schema(count: int) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ape_candidates",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "maxLength": 4000,
                        },
                        "minItems": count,
                        "maxItems": count,
                    }
                },
                "required": ["candidates"],
                "additionalProperties": False,
            },
        },
    }


def _parse_candidates(text: str) -> list[str]:
    source = text.strip()
    if source.startswith("```"):
        lines = source.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        source = "\n".join(lines).strip()
    payload = json.loads(source)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("APE returned no candidate list")
    return [
        item.strip()
        for item in candidates
        if isinstance(item, str) and item.strip()
    ]


def optimize_label_prompt(
    config: RunConfig,
    recorder: RunRecorder,
    run_dir: Path,
    *,
    task_id: str,
    task_description: str,
    rows: list[dict],
    starting_prompt: str,
    labels: set[str],
    student: StudentModel,
    teacher: TeacherClient,
) -> tuple[str, dict]:
    """Generate candidate instructions and select them on development cases."""
    if not rows:
        raise ValueError("APE requires labeled development cases")
    candidate_count = min(
        8,
        config.optimizer_metric_budget // len(rows),
    )
    if candidate_count < 2:
        raise ValueError(
            "APE metric budget must cover two full development passes"
        )

    demonstrations = "\n".join(
        json.dumps(
            {"input": row["input"], "output": row["answer"]},
            sort_keys=True,
        )
        for row in rows
    )
    requested = candidate_count - 1
    prompt_context = "\n\n".join(
        [
            "Task description:",
            task_description,
            "Observed input-output pairs:",
            demonstrations,
        ]
    )
    responses = []
    generation_errors = []
    proposed = []
    max_generation_calls = 1 + 2 * requested
    while len(proposed) < requested and len(responses) < max_generation_calls:
        remaining = 1
        retry_note = (
            "\n\nAlready proposed instructions:\n"
            + "\n".join(f"- {item}" for item in proposed)
            if proposed
            else ""
        )
        request = (
            f"\n\nPropose {remaining} distinct reusable instructions for "
            "this task. Each instruction must solve new inputs, state all "
            "decision rules and exceptions, and return only one of: "
            f"{', '.join(sorted(labels))}. Use at most 250 words for each "
            "instruction. Do not copy an input."
        )
        response = teacher.call(
            "Generate reusable task instructions from demonstrations.",
            prompt_context + retry_note + request,
            max_tokens=max(800, config.token_budget),
            response_format=_candidate_schema(remaining),
            seed=config.seed + len(responses),
        )
        recorder.add_teacher_tokens(
            response.input_tokens,
            response.output_tokens,
        )
        responses.append(response.text)
        try:
            parsed = _parse_candidates(response.text)
        except (json.JSONDecodeError, RuntimeError) as exc:
            generation_errors.append(
                {
                    "call": len(responses),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        for candidate in parsed:
            if candidate != starting_prompt and candidate not in proposed:
                proposed.append(candidate)
                if len(proposed) == requested:
                    break
    candidates = [starting_prompt]
    candidates.extend(proposed)
    if len(candidates) != candidate_count:
        raise RuntimeError(
            f"APE produced {len(candidates)} unique candidates; "
            f"expected {candidate_count}"
        )

    scores = []
    predictions = []
    student_input_tokens = 0
    student_output_tokens = 0
    calls = 0
    for candidate_index, candidate in enumerate(candidates):
        correct = 0
        for row in rows:
            answer = student.ask(
                f"Input: {row['input']}",
                system=candidate,
            )
            calls += 1
            student_input_tokens += answer.input_tokens
            student_output_tokens += answer.output_tokens
            parsed = normalise_student_label(answer.text, labels)
            matched = parsed == row["answer"]
            correct += int(matched)
            predictions.append(
                {
                    "candidate": candidate_index,
                    "item_id": row["item_id"],
                    "gold": row["answer"],
                    "predicted": parsed,
                    "correct": matched,
                }
            )
            if calls % 25 == 0:
                recorder.log(f"APE development rollout {calls}")
        scores.append(correct / len(rows))

    recorder.add_student_tokens(
        student_input_tokens,
        student_output_tokens,
    )
    best_index = min(
        range(len(candidates)),
        key=lambda index: (
            -scores[index],
            len(candidates[index]),
            index,
        ),
    )
    protocol = {
        "implementation": "official_algorithm_adaptation",
        "source": APE_SOURCE,
        "candidate_count": candidate_count,
        "development_examples": len(rows),
        "metric_call_budget": config.optimizer_metric_budget,
        "task_model_calls": calls,
        "teacher_calls": len(responses),
        "generation_failures": len(generation_errors),
        "maximum_generation_calls": max_generation_calls,
        "best_candidate_index": best_index,
        "best_build_accuracy": scores[best_index],
        "selection_tie_break": "shorter_prompt_then_candidate_order",
        "final_access_during_optimization": False,
        "seed": config.seed,
    }
    artifact = {
        "protocol": protocol,
        "candidates": candidates,
        "generation_responses": responses,
        "generation_errors": generation_errors,
        "development_scores": scores,
        "development_predictions": predictions,
    }
    path = run_dir / "artifacts" / f"{task_id}_ape_result.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return candidates[best_index], protocol
