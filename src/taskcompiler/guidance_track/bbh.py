"""Official BBH evaluation with published prompt baselines."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from taskcompiler.config import (
    DATA_DIR,
    FEW_SHOT_COT,
    GEPA_OPTIMIZER,
    GENERIC_VERIFIED_SCRATCHPAD,
    LINEAGE_COMPACT_GUIDANCE,
    LINEAGE_SCRATCHPAD_GUIDANCE,
    LINEAGE_VERIFIED_SCRATCHPAD,
    NONPARENT_VERIFIED_SCRATCHPAD,
    RAW_TASK,
    STRUCTURED_GUIDANCE,
    TASK_LEVEL_GUIDANCE,
    TEXTGRAD_PUBLISHED_PROMPT,
    TEXTGRAD_TGD,
    ZERO_SHOT_COT,
    RunConfig,
)
from taskcompiler.metrics import RunRecorder
from taskcompiler.lineage import validate_lineage_config
from taskcompiler.statistics import wilson_accuracy
from taskcompiler.student import StudentModel


TRACKING_TASKS = (
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
)
TRANSFER_SCREEN_TASKS = (
    "boolean_expressions",
    "date_understanding",
    "formal_fallacies",
)
TRACKING_TRACE_SCHEMA = {
    "type": "object",
    "properties": {
        "initial": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "person": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["person", "object"],
            },
        },
        "swaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "first": {"type": "string"},
                    "second": {"type": "string"},
                },
                "required": ["order", "first", "second"],
            },
        },
        "query": {"type": "string"},
        "options": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["initial", "swaps", "query", "options"],
}
DATE_TRACE_SCHEMA = {
    "type": "object",
    "properties": {
        "today": {"type": "string"},
        "operation": {
            "type": "string",
            "enum": ["add", "subtract"],
        },
        "amount": {"type": "integer"},
        "unit": {
            "type": "string",
            "enum": ["day", "week", "month", "year"],
        },
        "options": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["today", "operation", "amount", "unit", "options"],
}
SUPPORTED_TASKS = (
    "object_counting",
    "word_sorting",
    "navigate",
    *TRACKING_TASKS,
    *TRANSFER_SCREEN_TASKS,
)
TEXTGRAD_START_PROMPT = (
    "You will answer a reasoning question. Think step by step. The last line "
    "of your response should be of the following format: 'Answer: $VALUE' "
    "where VALUE is a numerical value."
)
TEXTGRAD_PROMPTS = {
    "object_counting": (
        "You will answer a reasoning question. List each item and its quantity "
        "in a clear and consistent format, such as '- Item: Quantity'. Sum the "
        "values directly from the list and provide a concise summation. Ensure "
        "the final answer is clearly indicated in the format: 'Answer: $VALUE' "
        "where VALUE is a numerical value. Verify the relevance of each item to "
        "the context of the query and handle potential errors or ambiguities in "
        "the input. Double-check the final count to ensure accuracy."
    ),
    "word_sorting": (
        "You will answer a reasoning question. Think step by step. Clearly state "
        "the final answer in the format expected by the question. Ensure that "
        "your final answer matches the format of the expected answer, whether it "
        "is a list, a numerical value, or another type of response. When sorting "
        "or performing any ordered task, perform lexicographical comparison, "
        "which involves comparing each character in sequence until a difference "
        "is found. Clearly explain the criteria and steps involved, including how "
        "to handle ties or similar cases. For tasks involving multiple steps, "
        "break down each step in detail and explain the reasoning behind each "
        "action. Only include steps that are necessary for the task at hand, and "
        "avoid mentioning irrelevant or redundant steps. After arriving at your "
        "final answer, include a verification step that compares each element in "
        "the final output against the expected result to ensure they match. "
        "Include intermediate steps or iterations of the process to verify the "
        "accuracy of each step. Consider and address potential edge cases or "
        "assumptions, such as handling different capitalizations, special "
        "characters, or spaces, in your reasoning to demonstrate a robust "
        "understanding of the task. Ensure consistency in the presentation of "
        "information, such as using the same format for lists or other structured "
        "data throughout your response. Explicitly state the final prediction in "
        "a clear and unambiguous manner, using a formal structure such as 'Final "
        "Answer: [item1, item2, item3, ...]'. Provide a detailed explanation of "
        "your reasoning process, justifying each step, and avoid including "
        "unnecessary or incorrect information. Include a step for error handling "
        "or specify what to do if an unexpected input is encountered."
    ),
}
TASK_DESCRIPTIONS = {
    "object_counting": (
        "Count only the objects that belong to the category asked about, "
        "including stated quantities, and return the numerical total."
    ),
    "word_sorting": (
        "Sort every supplied word in lexicographic alphabetical order and "
        "return the complete space-separated list."
    ),
    "navigate": (
        "Follow every movement and turn instruction from the origin, track "
        "the current facing direction, and answer whether the final position "
        "returns to the origin."
    ),
    "boolean_expressions": (
        "Evaluate the fully specified Boolean expression using standard not, "
        "and, and or semantics, then return True or False."
    ),
    "date_understanding": (
        "Resolve the stated calendar-date relations and select the option "
        "whose date satisfies the question."
    ),
    "formal_fallacies": (
        "Decide whether the stated conclusion follows deductively from the "
        "given premises, then return valid or invalid."
    ),
    **{
        task_name: (
            "Track which object each named person holds while pairs swap "
            "objects in sequence, then select the option matching the queried "
            "person's final object."
        )
        for task_name in TRACKING_TASKS
    },
}


def _paths(task_name: str) -> tuple[Path, Path]:
    root = DATA_DIR / "bbh_official"
    return (
        root / "bbh" / f"{task_name}.json",
        root / "cot-prompts" / f"{task_name}.txt",
    )


def load_textgrad_split(task_name: str, split: str) -> list[dict]:
    """Load TextGrad's exact positional 50/100/100 split."""
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(
            f"BBH task must be one of {', '.join(SUPPORTED_TASKS)}"
        )
    data_path, _ = _paths(task_name)
    if not data_path.exists():
        raise FileNotFoundError(
            "Official BBH cache is missing. Run scripts/download_data.py."
        )
    examples = json.loads(data_path.read_text(encoding="utf-8"))["examples"]
    bounds = {"train": (0, 50), "validation": (50, 150), "test": (150, 250)}
    if split not in bounds:
        raise ValueError(f"unknown BBH split: {split}")
    start, end = bounds[split]
    prepared = []
    for source_index, example in enumerate(examples[start:end], start=start):
        question = str(example["input"])
        item_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        prepared.append(
            {
                "task_name": task_name,
                "source_index": source_index,
                "item_id": f"{task_name}-{source_index:03d}-{item_hash[:10]}",
                "question": question,
                "reference": str(example["target"]),
            }
        )
    return prepared


def _manifest(task_name: str) -> dict:
    data_path, prompt_path = _paths(task_name)
    payload = {
        "protocol": "textgrad-bbh-50-100-100",
        "task_name": task_name,
        "source_repository": "suzgunmirac/BIG-Bench-Hard",
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "few_shot_prompt_sha256": hashlib.sha256(
            prompt_path.read_bytes()
        ).hexdigest(),
        "splits": {
            split: [row["item_id"] for row in load_textgrad_split(task_name, split)]
            for split in ("train", "validation", "test")
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def parse_answer(task_name: str, text: str) -> str | None:
    """Parse the task answer from direct or chain-of-thought output."""
    candidate = text.strip()
    markers = list(
        re.finditer(
            r"(?:final\s+answer|answer|answer\s+is)\s*:\s*",
            candidate,
            flags=re.IGNORECASE,
        )
    )
    if markers:
        remainder = candidate[markers[-1].end() :].strip()
        candidate = remainder.splitlines()[0].strip() if remainder else ""
    elif "so the answer is" in candidate.casefold():
        remainder = re.split(
            r"so\s+the\s+answer\s+is",
            candidate,
            flags=re.IGNORECASE,
        )[-1].strip()
        candidate = remainder.splitlines()[0].strip() if remainder else ""
    elif candidate.splitlines():
        candidate = candidate.splitlines()[-1].strip()

    if task_name == "object_counting":
        numbers = re.findall(r"-?\d+", candidate)
        if not numbers:
            numbers = re.findall(r"-?\d+", text)
        return str(int(numbers[-1])) if numbers else None
    if task_name == "navigate":
        answers = re.findall(r"\b(?:yes|no)\b", candidate.casefold())
        if not answers:
            answers = re.findall(r"\b(?:yes|no)\b", text.casefold())
        return answers[-1] if answers else None
    if task_name in TRACKING_TASKS:
        options = re.findall(r"\(?\b([A-G])\b\)?", candidate.upper())
        if not options:
            options = re.findall(
                r"(?:answer|option)\s*(?:is|:)?\s*\(?([A-G])\)?",
                text,
                flags=re.IGNORECASE,
            )
        return options[-1].casefold() if options else None

    words = re.findall(r"[A-Za-z]+", candidate.casefold())
    return " ".join(words) if words else None


def _few_shot_prefix(task_name: str) -> str:
    _, prompt_path = _paths(task_name)
    text = prompt_path.read_text(encoding="utf-8")
    _, separator, body = text.partition("-----")
    return body.strip() if separator else text.strip()


def _prompt(task_name: str, baseline: str, question: str) -> tuple[str, str | None]:
    if baseline == RAW_TASK:
        if task_name == "word_sorting":
            words = question.partition("List:")[2].strip()
            return (
                "Sort these words alphabetically and return only the complete "
                f"space-separated sequence, with no label: {words}",
                None,
            )
        if task_name == "date_understanding":
            return (
                f"Question: {question}\n"
                "Return only the letter of the correct option.",
                None,
            )
        return (
            f"Question: {question}\nReturn only the answer value.",
            None,
        )
    if baseline == ZERO_SHOT_COT:
        return (
            f"Question: {question}",
            TEXTGRAD_START_PROMPT,
        )
    if baseline == TEXTGRAD_PUBLISHED_PROMPT:
        return (
            f"Question: {question}",
            TEXTGRAD_PROMPTS[task_name],
        )
    if baseline == FEW_SHOT_COT:
        return (
            f"{_few_shot_prefix(task_name)}\n\nQ: {question}\nA: "
            "Let's think step by step.",
            None,
        )
    raise ValueError(f"unsupported BBH baseline: {baseline}")


def execute(config: RunConfig, recorder: RunRecorder, run_dir: Path) -> dict:
    """Evaluate a published BBH prompt condition on TextGrad's test split."""
    if config.baseline not in {
        RAW_TASK,
        ZERO_SHOT_COT,
        FEW_SHOT_COT,
        TEXTGRAD_PUBLISHED_PROMPT,
        TEXTGRAD_TGD,
        GEPA_OPTIMIZER,
        STRUCTURED_GUIDANCE,
        LINEAGE_COMPACT_GUIDANCE,
        LINEAGE_SCRATCHPAD_GUIDANCE,
        LINEAGE_VERIFIED_SCRATCHPAD,
        NONPARENT_VERIFIED_SCRATCHPAD,
        GENERIC_VERIFIED_SCRATCHPAD,
    }:
        raise ValueError(
            "Official BBH mode supports raw_task, zero_shot_cot, "
            "few_shot_cot, textgrad_published_prompt, textgrad_tgd, "
            "gepa, structured_guidance, lineage_compact_guidance, and "
            "lineage_scratchpad_guidance."
        )
    task_name = config.benchmark_task
    lineage = validate_lineage_config(config)
    guidance = None
    optimized_prompt = None
    optimizer_protocol = None
    if config.baseline in {
        STRUCTURED_GUIDANCE,
        LINEAGE_COMPACT_GUIDANCE,
        LINEAGE_SCRATCHPAD_GUIDANCE,
        LINEAGE_VERIFIED_SCRATCHPAD,
        NONPARENT_VERIFIED_SCRATCHPAD,
    }:
        if config.guidance_mode != TASK_LEVEL_GUIDANCE:
            raise ValueError(
                "BBH structured_guidance is restricted to offline task-level use"
            )
        from taskcompiler.guidance_track.pipeline import _generate_guidance
        from taskcompiler.teacher import TeacherClient

        if config.compiled_guidance_path:
            with recorder.stage("load_offline_task_guidance"):
                structured_guidance = _load_lineage_guidance(
                    Path(config.compiled_guidance_path),
                    config,
                )
                from taskcompiler.guidance_track.pipeline import (
                    _render_structured_guidance,
                )

                guidance = _render_structured_guidance(
                    structured_guidance
                )
        else:
            teacher = TeacherClient(
                config.teacher_model,
                backend=config.teacher_backend,
                base_url=config.openrouter_base_url,
                timeout_seconds=config.teacher_timeout_seconds,
                provider=config.teacher_openrouter_provider,
            )
            recorder.set_model_identity("teacher", teacher.identity)
            with recorder.stage("generate_offline_task_guidance"):
                guidance, structured_guidance = _generate_guidance(
                    teacher,
                    TASK_DESCRIPTIONS[task_name],
                    recorder,
                    mode=TASK_LEVEL_GUIDANCE,
                    style=STRUCTURED_GUIDANCE,
                    max_tokens=config.token_budget,
                    max_retries=config.max_guidance_retries,
                    use_response_format=config.teacher_structured_outputs,
                )
        (run_dir / "artifacts" / "offline_guidance.txt").write_text(
            guidance, encoding="utf-8"
        )
        (run_dir / "artifacts" / "offline_guidance.json").write_text(
            json.dumps(structured_guidance, indent=2),
            encoding="utf-8",
        )
        (run_dir / "artifacts" / "offline_guidance.manifest.json").write_text(
            json.dumps(
                {
                    "teacher_selection": config.teacher_selection,
                    "teacher_backend": config.teacher_backend,
                    "teacher_model": config.teacher_model,
                    "student_model": config.student_model,
                    "lineage": lineage.to_dict() if lineage else None,
                    "teacher_received_evaluation_examples": False,
                    "student_weights_updated": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    manifest = _manifest(task_name)
    (run_dir / "artifacts" / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if config.baseline == TEXTGRAD_TGD:
        from taskcompiler.guidance_track.textgrad_baseline import optimize_prompt

        with recorder.stage("load_locked_textgrad_development_splits"):
            train_rows = load_textgrad_split(task_name, "train")
            validation_rows = load_textgrad_split(task_name, "validation")
        optimized_prompt, optimizer_protocol = optimize_prompt(
            config,
            recorder,
            run_dir,
            train_rows,
            validation_rows,
            TEXTGRAD_START_PROMPT,
        )
    elif config.baseline == GEPA_OPTIMIZER:
        from taskcompiler.guidance_track.gepa_baseline import optimize_prompt

        with recorder.stage("load_locked_gepa_development_splits"):
            train_rows = load_textgrad_split(task_name, "train")
            validation_rows = load_textgrad_split(task_name, "validation")
        optimized_prompt, optimizer_protocol = optimize_prompt(
            config,
            recorder,
            run_dir,
            train_rows,
            validation_rows,
            TEXTGRAD_START_PROMPT,
        )
    evaluation_split = (
        "validation"
        if config.benchmark_split in {"validation", "dev"}
        else config.benchmark_split
    )
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("BBH split must be validation, dev, or test")
    with recorder.stage(f"load_textgrad_{evaluation_split}_split"):
        rows = load_textgrad_split(
            task_name,
            evaluation_split,
        )[: config.num_examples]
    if config.baseline != TEXTGRAD_TGD:
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
        recorder.set_model_identity("student", student.identity)
    else:
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

    predictions = []
    predictions_path = run_dir / "artifacts" / "predictions.partial.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    correct = 0
    parse_failures = 0
    with recorder.stage("final_evaluation"):
        for row in rows:
            if config.baseline == STRUCTURED_GUIDANCE:
                prompt = (
                    f"Question: {row['question']}\n\n"
                    "Use the reusable guidance and end with 'Answer: $VALUE'."
                )
                system = (
                    f"Task: {TASK_DESCRIPTIONS[task_name]}\n\n"
                    f"Reusable guidance:\n{guidance}"
                )
            elif config.baseline == LINEAGE_COMPACT_GUIDANCE:
                if task_name == "word_sorting":
                    words = row["question"].partition("List:")[2].strip()
                    prompt = (
                        "Sort these words using the transferred procedure. "
                        "Return only the complete space-separated sequence, "
                        f"with no label: {words}"
                    )
                elif task_name == "date_understanding":
                    prompt = (
                        f"Question: {row['question']}\n"
                        "Use the transferred procedure and return only the "
                        "letter of the correct option."
                    )
                else:
                    prompt = (
                        f"Question: {row['question']}\n"
                        "Return only the answer value."
                    )
                system = _render_compact_lineage_guidance(
                    TASK_DESCRIPTIONS[task_name],
                    structured_guidance,
                )
            elif config.baseline == LINEAGE_SCRATCHPAD_GUIDANCE:
                prompt = (
                    f"Question: {row['question']}\n"
                    "Follow the transferred procedure. End with "
                    "'Answer: $VALUE'."
                )
                system = _render_scratchpad_lineage_guidance(
                    task_name,
                    TASK_DESCRIPTIONS[task_name],
                    structured_guidance,
                )
            elif config.baseline in {
                LINEAGE_VERIFIED_SCRATCHPAD,
                NONPARENT_VERIFIED_SCRATCHPAD,
            }:
                prompt = _verified_scratchpad_prompt(
                    task_name,
                    row["question"],
                    parent_transferred=True,
                )
                system = _render_scratchpad_lineage_guidance(
                    task_name,
                    TASK_DESCRIPTIONS[task_name],
                    structured_guidance,
                )
            elif config.baseline == GENERIC_VERIFIED_SCRATCHPAD:
                prompt = _verified_scratchpad_prompt(
                    task_name,
                    row["question"],
                    parent_transferred=False,
                )
                system = _render_generic_scratchpad(
                    task_name,
                    TASK_DESCRIPTIONS[task_name]
                )
            elif config.baseline in {TEXTGRAD_TGD, GEPA_OPTIMIZER}:
                prompt = row["question"]
                system = optimized_prompt
            else:
                prompt, system = _prompt(
                    task_name, config.baseline, row["question"]
                )
            response = student.ask(
                prompt,
                system=system,
                response_schema=(
                    TRACKING_TRACE_SCHEMA
                    if task_name in TRACKING_TASKS
                    and config.baseline in {
                        LINEAGE_VERIFIED_SCRATCHPAD,
                        NONPARENT_VERIFIED_SCRATCHPAD,
                        GENERIC_VERIFIED_SCRATCHPAD,
                    }
                    else DATE_TRACE_SCHEMA
                    if task_name == "date_understanding"
                    and config.baseline in {
                        LINEAGE_VERIFIED_SCRATCHPAD,
                        NONPARENT_VERIFIED_SCRATCHPAD,
                        GENERIC_VERIFIED_SCRATCHPAD,
                    }
                    else None
                ),
            )
            recorder.add_student_tokens(
                response.input_tokens, response.output_tokens
            )
            initial_response = response
            parsed = parse_answer(task_name, initial_response.text)
            verified_ledger = None
            correction_response = None
            if config.baseline in {
                LINEAGE_VERIFIED_SCRATCHPAD,
                NONPARENT_VERIFIED_SCRATCHPAD,
                GENERIC_VERIFIED_SCRATCHPAD,
            }:
                verified_ledger = _verified_scratchpad_answer(
                    task_name,
                    initial_response.text,
                )
                if verified_ledger is not None and verified_ledger != parsed:
                    correction_response = student.ask(
                        (
                            "A local verifier checked your structured trace "
                            f"and obtained {verified_ledger}. Return exactly "
                            f"`Answer: {verified_ledger}`."
                        ),
                        system=(
                            "Return the locally verified answer exactly as "
                            "requested. Do not add reasoning."
                        ),
                    )
                    recorder.add_student_tokens(
                        correction_response.input_tokens,
                        correction_response.output_tokens,
                    )
                    parsed = parse_answer(
                        task_name,
                        correction_response.text,
                    )
            if task_name == "object_counting":
                gold = str(int(row["reference"]))
            elif task_name == "navigate":
                gold = row["reference"].strip().casefold()
            elif task_name in TRACKING_TASKS:
                gold = row["reference"].strip("() \t\r\n").casefold()
            else:
                gold = " ".join(
                    re.findall(r"[A-Za-z]+", row["reference"].casefold())
                )
            matched = parsed == gold
            correct += int(matched)
            parse_failures += int(parsed is None)
            prediction = {
                **row,
                "raw_output": (
                    correction_response.text
                    if correction_response
                    else initial_response.text
                ),
                "initial_output": initial_response.text,
                "verified_ledger": verified_ledger,
                "local_correction_used": correction_response is not None,
                "parsed_output": parsed,
                "gold": gold,
                "correct": matched,
                "input_tokens": initial_response.input_tokens
                + (
                    correction_response.input_tokens
                    if correction_response
                    else 0
                ),
                "output_tokens": initial_response.output_tokens
                + (
                    correction_response.output_tokens
                    if correction_response
                    else 0
                ),
                "latency_seconds": initial_response.latency_seconds
                + (
                    correction_response.latency_seconds
                    if correction_response
                    else 0
                ),
                "finish_reason": (
                    correction_response.finish_reason
                    if correction_response
                    else initial_response.finish_reason
                ),
            }
            predictions.append(prediction)
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(prediction) + "\n")
            recorder.log(
                f"BBH example {len(predictions)}/{len(rows)}: "
                f"{'passed' if matched else 'failed'}"
            )

    (run_dir / "artifacts" / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in predictions),
        encoding="utf-8",
    )
    interval = wilson_accuracy(correct, len(rows))
    metrics = {
        "dataset": "bbh",
        "benchmark_task": task_name,
        "protocol": "textgrad-bbh-50-100-100",
        "baseline": config.baseline,
        "accuracy": correct / len(rows),
        "correct": correct,
        "accuracy_wilson_95": {
            "low": interval.low,
            "high": interval.high,
            "descriptive_only": True,
        },
        "num_examples": len(rows),
        "parse_failures": parse_failures,
        "local_corrections": sum(
            row.get("local_correction_used", False)
            for row in predictions
        ),
        "verified_scratchpad": config.baseline in {
            LINEAGE_VERIFIED_SCRATCHPAD,
            NONPARENT_VERIFIED_SCRATCHPAD,
            GENERIC_VERIFIED_SCRATCHPAD,
        },
        "compiled_guidance_path": config.compiled_guidance_path,
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_split": evaluation_split,
        "development_only": evaluation_split != "test",
        "teacher_backend": config.teacher_backend,
        "teacher_model": config.teacher_model,
        "teacher_selection": config.teacher_selection,
        "lineage": lineage.to_dict() if lineage else None,
        "student_weights_updated": False,
        "transfer_target": "frozen_task_artifact",
        "training_labels_used": config.baseline in {
            TEXTGRAD_TGD,
            GEPA_OPTIMIZER,
        },
        "validation_labels_used": _uses_validation_labels(config.baseline),
        "published_prompt_transfer": (
            config.baseline == TEXTGRAD_PUBLISHED_PROMPT
        ),
        "faithful_optimizer_rerun": config.baseline in {
            TEXTGRAD_TGD,
            GEPA_OPTIMIZER,
        },
        "optimizer_protocol": optimizer_protocol,
        "condition_class": (
            "offline_reusable_guidance"
            if config.baseline in {
                STRUCTURED_GUIDANCE,
                LINEAGE_COMPACT_GUIDANCE,
                LINEAGE_SCRATCHPAD_GUIDANCE,
                LINEAGE_VERIFIED_SCRATCHPAD,
            }
            else "capacity_matched_nonparent_control"
            if config.baseline == NONPARENT_VERIFIED_SCRATCHPAD
            else "verified_scratchpad_control"
            if config.baseline == GENERIC_VERIFIED_SCRATCHPAD
            else "published_textual_gradient_optimizer"
            if config.baseline == TEXTGRAD_TGD
            else "official_gepa_optimizer"
            if config.baseline == GEPA_OPTIMIZER
            else "published_or_raw_prompt_baseline"
        ),
        "inferential_unit": "BBH example (descriptive pilot)",
    }
    recorder.set_task_metrics(metrics)
    return metrics


def _load_lineage_guidance(
    path: Path,
    config: RunConfig,
) -> dict:
    """Load a public parent artifact with matching lineage provenance."""
    from taskcompiler.guidance_track.pipeline import (
        _validate_structured_guidance,
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    guidance = _validate_structured_guidance(value)
    manifest_path = path.with_name("offline_guidance.manifest.json")
    if not manifest_path.exists():
        raise ValueError(
            "lineage guidance requires offline_guidance.manifest.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "teacher_model": config.teacher_model,
        "student_model": config.student_model,
        "teacher_selection": config.teacher_selection,
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise ValueError(f"guidance manifest mismatch for {field}")
    return guidance


def _render_compact_lineage_guidance(
    task_description: str,
    guidance: dict,
) -> str:
    """Render parent knowledge without requesting a visible rationale."""
    return "\n".join(
        (
            f"Task: {task_description}",
            "Parent-transferred procedure:",
            *(f"- {step}" for step in guidance["reasoning_plan"]),
            "Final checks:",
            *(f"- {check}" for check in guidance["verification_checks"]),
            "Apply this procedure internally. Output only the answer value.",
        )
    )


def _render_scratchpad_lineage_guidance(
    task_name: str,
    task_description: str,
    guidance: dict,
) -> str:
    """Compile parent guidance into a task-specific checkable scratchpad."""
    parent_steps = guidance["reasoning_plan"]
    parent_checks = guidance["verification_checks"]
    if task_name == "navigate":
        return "\n".join(
            (
                f"Task: {task_description}",
                "Teacher-compiled execution contract:",
                *(f"- Procedure: {step}" for step in parent_steps),
                "- Start at (0,0), facing north.",
                "- Resolve each movement using the current facing direction.",
                "- Write one `trace 1: dx,dy` line per movement, using sequential integers.",
                "- Turns update facing but do not create a move line.",
                *(f"- Verify: {check}" for check in parent_checks),
                "End with exactly `Answer: Yes` only at (0,0), otherwise No.",
            )
        )
    if task_name in TRACKING_TASKS:
        return "\n".join(
            (
                f"Task: {task_description}",
                "Teacher-compiled execution contract:",
                *(f"- Procedure: {step}" for step in parent_steps),
                "- Preserve the complete initial holder-to-object mapping.",
                "- Apply swaps in their stated order; a swap exchanges objects.",
                "- Transcribe the typed state program without executing it.",
                *(f"- Verify: {check}" for check in parent_checks),
                "- A local runtime executes the state program and selects the option.",
            )
        )
    if task_name == "date_understanding":
        return "\n".join(
            (
                f"Task: {task_description}",
                "Teacher-compiled execution contract:",
                *(f"- Procedure: {step}" for step in parent_steps),
                "- Resolve the narrative's calendar date for today.",
                "- Represent the requested change as add or subtract, an integer amount, and a day, week, month, or year unit.",
                "- Copy every letter-to-date option exactly.",
                *(f"- Verify: {check}" for check in parent_checks),
                "- Transcribe the typed calendar program without executing it.",
                "- A local runtime executes the program and selects the option.",
            )
        )
    return "\n".join(
        (
            f"Task: {task_description}",
            "Teacher-compiled execution contract:",
            f"- Category step: {parent_steps[0]}",
            "- Write one `item: quantity` line for each relevant mention.",
            f"- Inclusion step: {parent_steps[2]}",
            f"- Sum step: {parent_steps[3]}",
            f"- Verify: {parent_checks[0]}",
            f"- Verify: {parent_checks[1]}",
            "End with exactly `Answer: N`, where N is the verified sum.",
        )
    )


def _render_generic_scratchpad(
    task_name: str,
    task_description: str,
) -> str:
    """Render the same structured interface without teacher knowledge."""
    if task_name == "navigate":
        return "\n".join(
            (
                f"Task: {task_description}",
                "Generic execution contract:",
                "- Track the position and facing direction.",
                "- Write one `trace 1: dx,dy` line per movement, using sequential integers.",
                "- Turns update facing but do not create a move line.",
                "- Check whether the final displacement is zero.",
                "End with exactly `Answer: Yes` or `Answer: No`.",
            )
        )
    if task_name in TRACKING_TASKS:
        return "\n".join(
            (
                f"Task: {task_description}",
                "Generic execution contract:",
                "- Preserve the complete initial holder-to-object mapping.",
                "- Preserve every swap in the stated order.",
                "- Transcribe the typed state program without executing it.",
                "- A local runtime executes the state program and selects the option.",
            )
        )
    if task_name == "date_understanding":
        return "\n".join(
            (
                f"Task: {task_description}",
                "Generic execution contract:",
                "- Resolve the narrative's calendar date for today.",
                "- Represent the requested change as add or subtract, an integer amount, and a day, week, month, or year unit.",
                "- Copy every letter-to-date option exactly.",
                "- Transcribe the typed calendar program without executing it.",
                "- A local runtime executes the program and selects the option.",
            )
        )
    return "\n".join(
        (
            f"Task: {task_description}",
            "Generic execution contract:",
            "- Identify the category requested by the question.",
            "- Write one `item: quantity` line for each relevant mention.",
            "- Include every stated quantity and use one when unstated.",
            "- Sum the ledger and check category membership.",
            "End with exactly `Answer: N`, where N is the verified sum.",
        )
    )


def _verified_scratchpad_prompt(
    task_name: str,
    question: str,
    *,
    parent_transferred: bool,
) -> str:
    """Request the task-specific structured output without extra prose."""
    if task_name == "navigate":
        contract = (
            "the transferred execution contract"
            if parent_transferred
            else "the execution contract"
        )
        return (
            f"Question: {question}\n"
            f"Follow {contract}. Output only one `trace 1: dx,dy` line "
            "for each movement command, in order, followed by one "
            "`Answer: Yes` or `Answer: No` line. Do not output prose."
        )
    if task_name in TRACKING_TASKS:
        contract = (
            "the transferred execution contract"
            if parent_transferred
            else "the execution contract"
        )
        return (
            f"Question: {question}\n"
            f"Follow {contract}. Do not solve the question or update states. "
            "Copy its facts into the required JSON state program. Include every "
            "initial person-object pair, every ordered swap, the queried person, "
            "and every letter-to-object option. Do not output an answer."
        )
    if task_name == "date_understanding":
        contract = (
            "the transferred execution contract"
            if parent_transferred
            else "the execution contract"
        )
        return (
            f"Question: {question}\n"
            f"Follow {contract}. Do not calculate the requested answer. "
            "Copy the resolved date for today, the requested calendar operation, "
            "and every letter-to-date option into the required JSON program. "
            "Use amount zero with add and day when the question asks for today. "
            "Do not output an answer."
        )
    return (
        f"Question: {question}\n"
        f"Follow {'the transferred ' if parent_transferred else 'the '}"
        "execution contract. End with 'Answer: $VALUE'."
    )


def _verified_ledger_sum(text: str) -> str | None:
    """Sum unique entries from explicit item-quantity ledger sections."""
    entries: dict[str, int] = {}
    active = False
    for line in text.splitlines():
        normalized = line.casefold()
        if "item: quantity" in normalized:
            active = True
            continue
        if active and (
            "inclusion step:" in normalized
            or "sum step:" in normalized
        ):
            active = False
            continue
        if not active:
            continue
        match = re.match(
            r"^\s*-\s*([^:]+):\s*(\d+)\b",
            line,
        )
        if not match:
            continue
        name = " ".join(match.group(1).casefold().split())
        quantity = int(match.group(2))
        entries[name] = max(entries.get(name, 0), quantity)
    return str(sum(entries.values())) if entries else None


def _verified_navigation_answer(text: str) -> str | None:
    """Sum a student-produced movement trace from the local scratchpad."""
    moves = []
    for line in text.splitlines():
        match = re.match(
            r"^\s*(?:-\s*)?(?:trace|move)\s+\d+\s*:\s*"
            r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            moves.append((int(match.group(1)), int(match.group(2))))
    if not moves:
        return None
    x = sum(move[0] for move in moves)
    y = sum(move[1] for move in moves)
    return "yes" if (x, y) == (0, 0) else "no"


def _normalized_trace_value(value: str) -> str:
    """Normalize a name or object copied into a typed trace."""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _verified_tracking_answer(text: str) -> str | None:
    """Execute a student-produced ownership and swap trace."""
    state: dict[str, str] = {}
    swaps: list[tuple[int, str, str]] = []
    options: dict[str, str] = {}
    query = None
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        for entry in payload.get("initial", []):
            if not isinstance(entry, dict):
                continue
            holder = _normalized_trace_value(str(entry.get("person", "")))
            item = _normalized_trace_value(str(entry.get("object", "")))
            if holder and item:
                state[holder] = item
        for entry in payload.get("swaps", []):
            if not isinstance(entry, dict):
                continue
            try:
                order = int(entry.get("order"))
            except (TypeError, ValueError):
                continue
            swaps.append(
                (
                    order,
                    _normalized_trace_value(str(entry.get("first", ""))),
                    _normalized_trace_value(str(entry.get("second", ""))),
                )
            )
        query = _normalized_trace_value(str(payload.get("query", "")))
        raw_options = payload.get("options", {})
        if isinstance(raw_options, dict):
            options = {
                str(letter).casefold(): _normalized_trace_value(str(item))
                for letter, item in raw_options.items()
            }
    for line in text.splitlines():
        typed_initial = re.match(
            r"^\s*INITIAL\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if typed_initial:
            holder = _normalized_trace_value(typed_initial.group(1))
            item = _normalized_trace_value(typed_initial.group(2))
            if holder and item:
                state[holder] = item
            continue
        typed_swap = re.match(
            r"^\s*SWAP\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if typed_swap:
            swaps.append(
                (
                    int(typed_swap.group(1)),
                    _normalized_trace_value(typed_swap.group(2)),
                    _normalized_trace_value(typed_swap.group(3)),
                )
            )
            continue
        typed_query = re.match(
            r"^\s*QUERY\s*\|\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if typed_query:
            query = _normalized_trace_value(typed_query.group(1))
            continue
        typed_option = re.match(
            r"^\s*OPTION\s*\|\s*([A-G])\s*\|\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if typed_option:
            options[typed_option.group(1).casefold()] = _normalized_trace_value(
                typed_option.group(2)
            )
            continue
        initial = re.match(
            r"^\s*(?:-\s*)?initial\s*:\s*([^=]+?)\s*=\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if initial:
            holder = _normalized_trace_value(initial.group(1))
            item = _normalized_trace_value(initial.group(2))
            if holder and item:
                state[holder] = item
            continue
        swap = re.match(
            r"^\s*(?:-\s*)?swap\s+(\d+)\s*:\s*(.+?)\s*<->\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if swap:
            swaps.append(
                (
                    int(swap.group(1)),
                    _normalized_trace_value(swap.group(2)),
                    _normalized_trace_value(swap.group(3)),
                )
            )
            continue
        query_match = re.match(
            r"^\s*(?:-\s*)?query\s*:\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if query_match:
            query = _normalized_trace_value(query_match.group(1))
            continue
        option = re.match(
            r"^\s*(?:-\s*)?option\s+([A-G])\s*:\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if option:
            options[option.group(1).casefold()] = _normalized_trace_value(
                option.group(2)
            )
    if not state or not swaps or query not in state or not options:
        return None
    for _, first, second in sorted(swaps):
        if first not in state or second not in state:
            return None
        state[first], state[second] = state[second], state[first]
    matches = [
        letter
        for letter, item in options.items()
        if item == state[query]
    ]
    return matches[0] if len(matches) == 1 else None


def _parse_calendar_date(value: str):
    """Parse a student-transcribed calendar date."""
    import datetime

    normalized = value.strip()
    for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.datetime.strptime(normalized, pattern).date()
        except ValueError:
            continue
    return None


def _shift_calendar_date(date, operation: str, amount: int, unit: str):
    """Execute a typed calendar operation."""
    import calendar
    import datetime

    signed = amount if operation == "add" else -amount
    if unit == "day":
        return date + datetime.timedelta(days=signed)
    if unit == "week":
        return date + datetime.timedelta(weeks=signed)
    if unit in {"month", "year"}:
        months = signed if unit == "month" else signed * 12
        month_index = date.year * 12 + date.month - 1 + months
        year, zero_based_month = divmod(month_index, 12)
        month = zero_based_month + 1
        day = min(date.day, calendar.monthrange(year, month)[1])
        return datetime.date(year, month, day)
    return None


def _verified_date_answer(text: str) -> str | None:
    """Execute a student-produced calendar program."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    today = _parse_calendar_date(str(payload.get("today", "")))
    operation = str(payload.get("operation", "")).casefold()
    unit = str(payload.get("unit", "")).casefold()
    try:
        amount = int(payload.get("amount"))
    except (TypeError, ValueError):
        return None
    if today is None or operation not in {"add", "subtract"} or amount < 0:
        return None
    target = _shift_calendar_date(today, operation, amount, unit)
    options = payload.get("options")
    if target is None or not isinstance(options, dict):
        return None
    matches = []
    for letter, value in options.items():
        option_date = _parse_calendar_date(str(value))
        if option_date == target:
            matches.append(str(letter).casefold())
    return matches[0] if len(matches) == 1 else None


def _verified_scratchpad_answer(task_name: str, text: str) -> str | None:
    """Verify the structured representation for a supported BBH task."""
    if task_name == "object_counting":
        return _verified_ledger_sum(text)
    if task_name == "navigate":
        return _verified_navigation_answer(text)
    if task_name in TRACKING_TASKS:
        return _verified_tracking_answer(text)
    if task_name == "date_understanding":
        return _verified_date_answer(text)
    raise ValueError(
        f"verified scratchpad does not support {task_name}"
    )


def _uses_validation_labels(baseline: str) -> bool:
    """Return whether a condition selects its prompt on validation labels."""
    return baseline in {TEXTGRAD_TGD, GEPA_OPTIMIZER}
