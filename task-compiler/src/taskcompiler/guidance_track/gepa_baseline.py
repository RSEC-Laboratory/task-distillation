"""Run the official GEPA optimizer on the locked BBH split."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from taskcompiler.config import RunConfig
from taskcompiler.metrics import RunRecorder
from taskcompiler.student import StudentModel
from taskcompiler.teacher import TeacherClient


GEPA_VERSION = "0.1.4"


def _gold(task_name: str, reference: str) -> str:
    if task_name == "object_counting":
        return str(int(reference))
    return " ".join(reference.casefold().split())


class _TaskModel:
    def __init__(self, student: StudentModel, recorder: RunRecorder):
        self.student = student
        self.recorder = recorder
        self.calls = 0

    def __call__(self, messages: Sequence[dict[str, str]]) -> str:
        system = None
        prompts = []
        for message in messages:
            if message["role"] == "system":
                system = message["content"]
            else:
                prompts.append(message["content"])
        response = self.student.ask("\n\n".join(prompts), system=system)
        self.recorder.add_student_tokens(
            response.input_tokens,
            response.output_tokens,
        )
        self.calls += 1
        if self.calls % 10 == 0:
            self.recorder.log(f"GEPA metric rollout {self.calls}")
        return response.text


class _ReflectionModel:
    def __init__(
        self,
        teacher: TeacherClient,
        recorder: RunRecorder,
        max_tokens: int,
    ):
        self.teacher = teacher
        self.recorder = recorder
        self.max_tokens = max_tokens
        self.calls = 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        system = "You improve reusable task instructions from evaluation traces."
        if isinstance(prompt, list):
            user_parts = []
            for message in prompt:
                content = str(message.get("content", ""))
                if message.get("role") == "system":
                    system = content
                else:
                    user_parts.append(content)
            user_prompt = "\n\n".join(user_parts)
        else:
            user_prompt = prompt
        response = self.teacher.call(
            system,
            user_prompt,
            max_tokens=self.max_tokens,
        )
        self.recorder.add_teacher_tokens(
            response.input_tokens,
            response.output_tokens,
        )
        self.calls += 1
        return response.text


def optimize_prompt(
    config: RunConfig,
    recorder: RunRecorder,
    run_dir: Path,
    train_rows: list[dict],
    validation_rows: list[dict],
    starting_prompt: str,
) -> tuple[str, dict]:
    """Optimize one reusable prompt with the official GEPA package."""
    import gepa
    from gepa.adapters.default_adapter.default_adapter import (
        DefaultAdapter,
        EvaluationResult,
    )

    if config.optimizer_metric_budget < len(validation_rows):
        raise ValueError(
            "GEPA metric budget must cover at least one full validation pass"
        )

    task_name = config.benchmark_task
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
        context_tokens=config.student_context_tokens,
        enable_thinking=config.student_thinking,
        seed=config.student_seed,
    )
    recorder.set_model_identity("student", student.identity)
    teacher = TeacherClient(
        config.teacher_model,
        backend=config.teacher_backend,
        base_url=config.openrouter_base_url,
        timeout_seconds=config.teacher_timeout_seconds,
        provider=config.teacher_openrouter_provider,
    )
    recorder.set_model_identity(
        "optimizer",
        teacher.identity,
    )

    def evaluator(data: dict, response: str) -> EvaluationResult:
        from taskcompiler.guidance_track.bbh import parse_answer

        expected = data["answer"]
        parsed = parse_answer(task_name, response)
        correct = parsed == expected
        if correct:
            feedback = f"Correct. Parsed final answer: {parsed}."
        else:
            feedback = (
                f"Incorrect. Parsed final answer: {parsed!r}. "
                f"Expected: {expected!r}. Improve the reusable reasoning "
                "instructions without copying this example."
            )
        return EvaluationResult(float(correct), feedback, None)

    def prepare(rows: list[dict]) -> list[dict]:
        return [
            {
                "input": row["question"],
                "answer": _gold(task_name, row["reference"]),
                "additional_context": {"item_id": row["item_id"]},
            }
            for row in rows
        ]

    adapter = DefaultAdapter(
        model=_TaskModel(student, recorder),
        evaluator=evaluator,
    )
    reflection_model = _ReflectionModel(
        teacher,
        recorder,
        max(2000, config.token_budget),
    )
    with recorder.stage("gepa_optimization"):
        result = gepa.optimize(
            seed_candidate={"system_prompt": starting_prompt},
            trainset=prepare(train_rows),
            valset=prepare(validation_rows),
            adapter=adapter,
            reflection_lm=reflection_model,
            reflection_minibatch_size=3,
            max_metric_calls=config.optimizer_metric_budget,
            candidate_selection_strategy="pareto",
            use_merge=False,
            val_evaluation_policy="full_eval",
            track_best_outputs=False,
            display_progress_bar=False,
            logger=recorder,
            seed=config.seed,
            raise_on_exception=True,
        )

    best_candidate = result.best_candidate
    if not isinstance(best_candidate, dict):
        raise RuntimeError("GEPA returned an invalid prompt candidate")
    optimized = best_candidate["system_prompt"]
    protocol = {
        "implementation": "official_gepa_package",
        "gepa_version": GEPA_VERSION,
        "metric_call_budget": config.optimizer_metric_budget,
        "metric_calls_used": result.total_metric_calls,
        "num_candidates": result.num_candidates,
        "best_candidate_index": result.best_idx,
        "best_validation_accuracy": result.val_aggregate_scores[result.best_idx],
        "reflection_minibatch_size": 3,
        "candidate_selection": "pareto",
        "merge": False,
        "validation_policy": "full_eval",
        "test_access_during_optimization": False,
        "reflection_calls": reflection_model.calls,
        "seed": config.seed,
    }
    artifact = {
        "protocol": protocol,
        "best_prompt": optimized,
        "candidate_prompts": result.candidates,
        "candidate_validation_scores": result.val_aggregate_scores,
        "parents": result.parents,
        "discovery_eval_counts": result.discovery_eval_counts,
    }
    (run_dir / "artifacts" / "gepa_result.json").write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )
    return optimized, protocol


def optimize_label_prompt(
    config: RunConfig,
    recorder: RunRecorder,
    run_dir: Path,
    *,
    task_id: str,
    rows: list[dict],
    starting_prompt: str,
    labels: set[str],
    student: StudentModel,
    teacher: TeacherClient,
) -> tuple[str, dict]:
    """Optimize a classification prompt over labeled build examples.

    Args:
        config: Run and optimizer settings.
        recorder: Token and progress recorder.
        run_dir: Directory that receives the optimizer trace.
        task_id: Stable task identifier used in the artifact name.
        rows: Build inputs with `input`, `answer`, and `item_id` fields.
        starting_prompt: Rendered policy shown before each input.
        labels: Declared output labels.
        student: Frozen task model.
        teacher: Reflection model.
    """
    import gepa
    from gepa.adapters.default_adapter.default_adapter import (
        DefaultAdapter,
        EvaluationResult,
    )
    from taskcompiler.translate import normalise_student_label

    if config.optimizer_metric_budget < len(rows):
        raise ValueError(
            "GEPA metric budget must cover at least one full build evaluation"
        )

    def evaluator(data: dict, response: str) -> EvaluationResult:
        parsed = normalise_student_label(response, labels)
        expected = data["answer"]
        correct = parsed == expected
        if correct:
            feedback = f"Correct. Parsed label: {parsed}."
        else:
            feedback = (
                f"Incorrect. Parsed label: {parsed!r}. Expected: "
                f"{expected!r}. Improve the reusable policy instructions "
                "without copying this input."
            )
        return EvaluationResult(float(correct), feedback, None)

    prepared = [
        {
            "input": row["input"],
            "answer": row["answer"],
            "additional_context": {"item_id": row["item_id"]},
        }
        for row in rows
    ]
    task_model = _TaskModel(student, recorder)
    reflection_model = _ReflectionModel(
        teacher,
        recorder,
        max(2000, config.token_budget),
    )
    adapter = DefaultAdapter(model=task_model, evaluator=evaluator)
    with recorder.stage(f"gepa_optimization_{task_id}"):
        result = gepa.optimize(
            seed_candidate={"system_prompt": starting_prompt},
            trainset=prepared,
            valset=prepared,
            adapter=adapter,
            reflection_lm=reflection_model,
            reflection_minibatch_size=3,
            max_metric_calls=config.optimizer_metric_budget,
            candidate_selection_strategy="pareto",
            use_merge=False,
            val_evaluation_policy="full_eval",
            track_best_outputs=False,
            display_progress_bar=False,
            logger=recorder,
            seed=config.seed,
            raise_on_exception=True,
        )

    best_candidate = result.best_candidate
    if not isinstance(best_candidate, dict):
        raise RuntimeError("GEPA returned an invalid prompt candidate")
    optimized = best_candidate["system_prompt"]
    protocol = {
        "implementation": "official_gepa_package",
        "gepa_version": GEPA_VERSION,
        "metric_call_budget": config.optimizer_metric_budget,
        "metric_calls_used": result.total_metric_calls,
        "task_model_calls": task_model.calls,
        "reflection_calls": reflection_model.calls,
        "num_candidates": result.num_candidates,
        "best_candidate_index": result.best_idx,
        "best_build_accuracy": result.val_aggregate_scores[result.best_idx],
        "reflection_minibatch_size": 3,
        "candidate_selection": "pareto",
        "merge": False,
        "validation_policy": "full_eval_on_build",
        "audit_access_during_optimization": False,
        "final_access_during_optimization": False,
        "seed": config.seed,
    }
    artifact = {
        "protocol": protocol,
        "best_prompt": optimized,
        "candidate_prompts": result.candidates,
        "candidate_build_scores": result.val_aggregate_scores,
        "parents": result.parents,
        "discovery_eval_counts": result.discovery_eval_counts,
    }
    artifact_path = run_dir / "artifacts" / f"{task_id}_gepa_result.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )
    return optimized, protocol
