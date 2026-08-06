"""Run TextGrad's published TGD loop on the locked BBH split."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from taskcompiler.config import PROJECT_ROOT, RunConfig
from taskcompiler.metrics import RunRecorder
from taskcompiler.student import StudentModel
from taskcompiler.teacher import TeacherClient


TEXTGRAD_COMMIT = "75e912e210864b61999781778cdf756d4468120f"
PUBLISHED_BATCH_SIZE = 3
PUBLISHED_EPOCHS = 3
PUBLISHED_STEPS_PER_EPOCH = 4


def published_batches(
    row_count: int,
    seed: int,
) -> list[list[int]]:
    """Return the batches visited by the published training loop."""
    indices = np.arange(row_count)
    random_state = np.random.RandomState(seed)
    batches = []
    for _ in range(PUBLISHED_EPOCHS):
        random_state.shuffle(indices)
        for step in range(PUBLISHED_STEPS_PER_EPOCH):
            start = step * PUBLISHED_BATCH_SIZE
            batches.append(indices[start : start + PUBLISHED_BATCH_SIZE].tolist())
    return batches


def _load_official_textgrad():
    source_root = PROJECT_ROOT / "data" / "textgrad_official"
    if not source_root.exists():
        raise FileNotFoundError("The cached official TextGrad source is missing")
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

    # The cached package imports this optional backend even when it is unused.
    module_name = "textgrad.engine_experimental.litellm"
    if module_name not in sys.modules:
        stub = types.ModuleType(module_name)
        stub.LiteLLMEngine = object
        sys.modules[module_name] = stub

    import textgrad as tg
    from textgrad.autograd.string_based_ops import StringBasedFunction
    from textgrad.tasks.big_bench_hard import string_based_equality_fn

    return tg, StringBasedFunction, string_based_equality_fn


class _StudentEngine:
    def __init__(self, student: StudentModel, recorder: RunRecorder):
        self.model_string = student.model_name
        self.student = student
        self.recorder = recorder

    def __call__(self, prompt, system_prompt=None, **_kwargs):
        response = self.student.ask(str(prompt), system=system_prompt)
        self.recorder.add_student_tokens(
            response.input_tokens,
            response.output_tokens,
        )
        return response.text

    generate = __call__


class _TeacherEngine:
    def __init__(
        self,
        teacher: TeacherClient,
        recorder: RunRecorder,
        max_tokens: int,
    ):
        self.model_string = teacher.model
        self.teacher = teacher
        self.recorder = recorder
        self.max_tokens = max_tokens

    def __call__(self, prompt, system_prompt=None, **_kwargs):
        response = self.teacher.call(
            system_prompt or "You are a precise optimization assistant.",
            str(prompt),
            max_tokens=self.max_tokens,
        )
        self.recorder.add_teacher_tokens(
            response.input_tokens,
            response.output_tokens,
        )
        return response.text

    generate = __call__


def optimize_prompt(
    config: RunConfig,
    recorder: RunRecorder,
    run_dir: Path,
    train_rows: list[dict],
    validation_rows: list[dict],
    starting_prompt: str,
) -> tuple[str, dict]:
    """Optimize one reusable prompt with the official TextGrad implementation."""
    if config.benchmark_task != "object_counting":
        raise ValueError(
            "The faithful TextGrad TGD rerun currently supports object_counting only"
        )

    tg, string_function, equality_function = _load_official_textgrad()
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
    recorder.set_model_identity("teacher", teacher.identity)
    student_engine = _StudentEngine(student, recorder)
    teacher_engine = _TeacherEngine(
        teacher,
        recorder,
        max(2000, config.token_budget),
    )
    tg.set_backward_engine(teacher_engine, override=True)

    prompt = tg.Variable(
        starting_prompt,
        requires_grad=True,
        role_description=(
            "structured system prompt to a somewhat capable language model "
            "that specifies the behavior and strategies for the QA task"
        ),
    )
    model = tg.BlackboxLLM(student_engine, prompt)
    optimizer = tg.TextualGradientDescent(
        engine=teacher_engine,
        parameters=[prompt],
    )
    evaluator = string_function(
        equality_function,
        function_purpose=(
            "The runtime of string-based function that checks if the "
            "prediction is correct."
        ),
    )

    batch_plan = published_batches(len(train_rows), config.seed)

    def validation_accuracy() -> float:
        scores = []
        for row in validation_rows:
            question = tg.Variable(
                row["question"],
                requires_grad=False,
                role_description="query to the language model",
            )
            answer = tg.Variable(
                row["reference"],
                requires_grad=False,
                role_description="correct answer for the query",
            )
            response = model(question)
            scores.append(equality_function(response, answer))
        return sum(scores) / len(scores)

    with recorder.stage("textgrad_initial_validation"):
        current_validation = validation_accuracy()
    history = [
        {
            "step": 0,
            "prompt": prompt.value,
            "batch_item_ids": [],
            "validation_accuracy": current_validation,
            "accepted": True,
        }
    ]
    with recorder.stage("textgrad_tgd_optimization"):
        for step, batch in enumerate(batch_plan, start=1):
            optimizer.zero_grad()
            losses = []
            for index in batch:
                row = train_rows[index]
                question = tg.Variable(
                    row["question"],
                    requires_grad=False,
                    role_description="query to the language model",
                )
                answer = tg.Variable(
                    row["reference"],
                    requires_grad=False,
                    role_description="correct answer for the query",
                )
                response = model(question)
                losses.append(
                    evaluator(
                        inputs={
                            "prediction": response,
                            "ground_truth_answer": answer,
                        }
                    )
                )
            tg.sum(losses).backward()
            optimizer.step()
            candidate_validation = validation_accuracy()
            accepted = candidate_validation >= current_validation
            if accepted:
                current_validation = candidate_validation
            else:
                prompt.set_value(history[-1]["prompt"])
            history.append(
                {
                    "step": step,
                    "prompt": prompt.value,
                    "batch_item_ids": [
                        train_rows[index]["item_id"] for index in batch
                    ],
                    "candidate_validation_accuracy": candidate_validation,
                    "validation_accuracy": current_validation,
                    "accepted": accepted,
                }
            )
            recorder.log(
                f"TextGrad update {step}/{len(batch_plan)} completed "
                f"(validation={candidate_validation:.3f}, "
                f"accepted={accepted})"
            )

    protocol = {
        "implementation": "official_textgrad_cached_source",
        "textgrad_commit": TEXTGRAD_COMMIT,
        "batch_size": PUBLISHED_BATCH_SIZE,
        "epochs": PUBLISHED_EPOCHS,
        "updates_per_epoch": PUBLISHED_STEPS_PER_EPOCH,
        "updates": len(batch_plan),
        "seed": config.seed,
        "test_access_during_optimization": False,
        "validation_selection": True,
        "validation_examples": len(validation_rows),
        "validation_evaluations": len(batch_plan) + 1,
        "best_validation_accuracy": current_validation,
        "deviation_from_published_script": (
            "The test split is evaluated once after optimization instead of "
            "after every update."
        ),
    }
    (run_dir / "artifacts" / "textgrad_history.json").write_text(
        json.dumps({"protocol": protocol, "history": history}, indent=2),
        encoding="utf-8",
    )
    return prompt.value, protocol
