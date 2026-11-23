"""Run SLR-Bench Basic and evaluate rules with a local symbolic judge."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from taskcompiler.config import (
    DATA_DIR,
    RAW_TASK,
    SLR_COMPILED_GUIDANCE,
    SLR_LOCAL_EVIDENCE,
    STRUCTURED_GUIDANCE,
    TASK_LEVEL_GUIDANCE,
    ZERO_SHOT_COT,
    RunConfig,
)
from taskcompiler.metrics import RunRecorder
from taskcompiler.statistics import wilson_accuracy
from taskcompiler.student import StudentModel
from taskcompiler.teacher import TeacherClient


@dataclass(frozen=True)
class Atom:
    predicate: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class HornRule:
    head: Atom
    body: tuple[Atom, ...]


@dataclass(frozen=True)
class RuleEvaluation:
    valid: bool
    semantically_correct: bool
    positive_correct: int
    positive_total: int
    negative_correct: int
    negative_total: int
    error: str | None = None


_ATOM_PATTERN = re.compile(r"([a-z][A-Za-z0-9_]*)\s*\(([^()]*)\)")
_RULE_PATTERN = re.compile(
    r"eastbound\s*\([^)]*\)\s*:-\s*[^.]+\.",
    re.IGNORECASE,
)

SLR_BASELINES = (
    RAW_TASK,
    ZERO_SHOT_COT,
    STRUCTURED_GUIDANCE,
    SLR_COMPILED_GUIDANCE,
    SLR_LOCAL_EVIDENCE,
)
COMPILED_FIELDS = (
    "induction_strategy",
    "candidate_construction",
    "variable_binding_rules",
    "minimality_rules",
    "counterfactual_checks",
    "output_contract",
)
PUBLIC_TASK_SPECIFICATION = """Induce one shortest conjunctive Prolog rule from
labeled positive and negative examples. Each example describes a train through
ground facts over has_car/2 and property predicates such as car_color/2,
car_len/2, car_num/2, and has_wall/2. The rule head must be eastbound(Train).
The body may use only predicates declared in the incoming problem. It must cover
every positive train and no negative train. The result must generalize by using
variables for train and car entities. Return one rule of the form
eastbound(Train) :- Body."""
COMPILED_SYSTEM_PROMPT = """Compile reusable, answer-free task knowledge for a
frozen small model that induces conjunctive Prolog rules. Return one JSON object
with exactly these array-of-string fields: induction_strategy,
candidate_construction, variable_binding_rules, minimality_rules,
counterfactual_checks, and output_contract. Do not solve or invent an instance.
Do not use concrete train or car constants, labels from an example, Markdown, or
additional fields. Put exactly two concise strings of at most 18 words in each
array."""
STUDENT_SYSTEM_PROMPT = """Return only one complete Prolog rule and no prose.
Use this exact shape: eastbound(Train) :- has_car(Train, Car),
property_name(Car, property_value). Add another property literal only when one
property cannot separate the labels. Replace property_name and property_value
with evidence values. Never copy example identifiers."""
_INSTANCE_CONSTANT = re.compile(
    r"\b(?:my)?(?:train|car)[_-]?\d",
    re.IGNORECASE,
)


def extract_rule(text: str) -> str | None:
    """Extract the final eastbound Horn rule from model output.

    Args:
        text: Raw model output that may contain prose or code fences.
    """
    matches = list(_RULE_PATTERN.finditer(text))
    return matches[-1].group(0).strip() if matches else None


def _parse_atom(text: str) -> Atom:
    match = _ATOM_PATTERN.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"invalid atom: {text.strip()!r}")
    arguments = tuple(
        argument.strip()
        for argument in match.group(2).split(",")
    )
    if not arguments or any(not argument for argument in arguments):
        raise ValueError("atom arguments must be non-empty")
    return Atom(match.group(1), arguments)


def parse_rule(text: str) -> HornRule:
    """Parse one non-recursive Horn rule.

    Args:
        text: Prolog rule with an `eastbound` head and non-empty body.
    """
    source = text.strip().removesuffix(".")
    if ":-" not in source:
        raise ValueError("rule must contain :-")
    head_text, body_text = source.split(":-", 1)
    head = _parse_atom(head_text)
    if head.predicate != "eastbound" or len(head.arguments) != 1:
        raise ValueError("rule head must be eastbound with one argument")
    atoms = tuple(
        _parse_atom(match.group(0))
        for match in _ATOM_PATTERN.finditer(body_text)
    )
    residual = _ATOM_PATTERN.sub("", body_text)
    if residual.replace(",", "").strip():
        raise ValueError("rule body contains unsupported syntax")
    if not atoms:
        raise ValueError("rule body must contain at least one atom")
    return HornRule(head, atoms)


def parse_facts(program: str) -> list[Atom]:
    """Parse ground facts from an SLR validation program.

    Args:
        program: Validation program containing one fact per line.
    """
    facts = []
    for raw_line in program.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.endswith("."):
            raise ValueError(f"fact must end with a period: {line!r}")
        facts.append(_parse_atom(line[:-1]))
    return facts


def _is_variable(value: str) -> bool:
    return bool(value) and (value[0].isupper() or value[0] == "_")


def _unify(
    pattern: Atom,
    fact: Atom,
    binding: dict[str, str],
) -> dict[str, str] | None:
    if (
        pattern.predicate != fact.predicate
        or len(pattern.arguments) != len(fact.arguments)
    ):
        return None
    updated = dict(binding)
    for expected, actual in zip(pattern.arguments, fact.arguments):
        if _is_variable(expected):
            bound = updated.get(expected)
            if bound is not None and bound != actual:
                return None
            updated[expected] = actual
        elif expected != actual:
            return None
    return updated


def rule_holds(
    rule: HornRule,
    train: str,
    facts: list[Atom],
) -> bool:
    """Evaluate a rule for one train using conjunctive joins.

    Args:
        rule: Candidate non-recursive Horn rule.
        train: Ground train constant to bind to the rule head.
        facts: Ground facts in the validation program.
    """
    head_argument = rule.head.arguments[0]
    if _is_variable(head_argument):
        bindings = [{head_argument: train}]
    elif head_argument == train:
        bindings = [{}]
    else:
        return False
    for atom in rule.body:
        next_bindings = []
        for binding in bindings:
            for fact in facts:
                unified = _unify(atom, fact, binding)
                if unified is not None:
                    next_bindings.append(unified)
        bindings = next_bindings
        if not bindings:
            return False
    return True


def evaluate_rule(
    candidate: str,
    validation_program: str,
) -> RuleEvaluation:
    """Check whether a candidate separates all positive and negative trains.

    Args:
        candidate: Candidate Prolog rule.
        validation_program: Hidden ground facts with eastbound/westbound labels.
    """
    try:
        rule = parse_rule(candidate)
        facts = parse_facts(validation_program)
    except ValueError as exc:
        return RuleEvaluation(False, False, 0, 0, 0, 0, str(exc))

    positives = [
        fact.arguments[0]
        for fact in facts
        if fact.predicate == "eastbound"
    ]
    negatives = [
        fact.arguments[0]
        for fact in facts
        if fact.predicate == "westbound"
    ]
    background = [
        fact
        for fact in facts
        if fact.predicate not in {"eastbound", "westbound"}
    ]
    allowed = {fact.predicate for fact in background}
    if any(atom.predicate not in allowed for atom in rule.body):
        return RuleEvaluation(
            False,
            False,
            0,
            len(positives),
            0,
            len(negatives),
            "rule uses a predicate absent from the validation program",
        )

    positive_correct = sum(
        rule_holds(rule, train, background)
        for train in positives
    )
    negative_correct = sum(
        not rule_holds(rule, train, background)
        for train in negatives
    )
    semantically_correct = (
        bool(positives)
        and bool(negatives)
        and positive_correct == len(positives)
        and negative_correct == len(negatives)
    )
    return RuleEvaluation(
        True,
        semantically_correct,
        positive_correct,
        len(positives),
        negative_correct,
        len(negatives),
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(
            r"^```(?:json)?\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(r"\s*```$", "", candidate)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("compiled guidance must be a JSON object")
    return value


def validate_compiled_guidance(value: dict[str, Any]) -> dict[str, list[str]]:
    """Validate the instance-free compiler artifact.

    Args:
        value: Parsed teacher JSON.
    """
    if set(value) != set(COMPILED_FIELDS):
        missing = sorted(set(COMPILED_FIELDS) - set(value))
        extra = sorted(set(value) - set(COMPILED_FIELDS))
        raise ValueError(
            f"compiled guidance keys do not match; missing={missing}, extra={extra}"
        )
    cleaned = {}
    for field in COMPILED_FIELDS:
        items = value[field]
        if (
            not isinstance(items, list)
            or len(items) != 2
            or not all(isinstance(item, str) and item.strip() for item in items)
        ):
            raise ValueError(f"{field} must contain exactly two strings")
        if any(len(item.split()) > 18 for item in items):
            raise ValueError(f"{field} entries must contain at most 18 words")
        cleaned[field] = [item.strip() for item in items]
    rendered = json.dumps(cleaned, ensure_ascii=False)
    if _INSTANCE_CONSTANT.search(rendered):
        raise ValueError("compiled guidance contains an instance constant")
    return cleaned


def _compile_guidance(
    teacher: TeacherClient,
    recorder: RunRecorder,
    config: RunConfig,
) -> tuple[str, dict[str, list[str]], str]:
    prompt = (
        "Public task specification:\n"
        f"{PUBLIC_TASK_SPECIFICATION}\n\n"
        "Compile reusable task knowledge from this specification only. No "
        "benchmark instance, target rule, or held-out validation program is "
        "available."
    )
    error = None
    for attempt in range(config.max_guidance_retries + 1):
        request = prompt
        if error:
            request += (
                f"\n\nThe previous artifact failed validation: {error}. "
                "Return a corrected JSON object."
            )
        response = teacher.call(
            COMPILED_SYSTEM_PROMPT,
            request,
            max_tokens=max(128, min(config.token_budget, 2000)),
        )
        recorder.add_teacher_tokens(response.input_tokens, response.output_tokens)
        try:
            value = validate_compiled_guidance(
                _extract_json_object(response.text)
            )
            rendered = json.dumps(value, ensure_ascii=False, indent=2)
            return rendered, value, prompt
        except (ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
            if attempt == config.max_guidance_retries:
                raise ValueError(
                    "teacher failed compiled-guidance validation after "
                    f"{attempt + 1} attempt(s): {error}"
                ) from exc
    raise AssertionError("unreachable")


def _load_rows(config: RunConfig) -> tuple[list[dict], dict]:
    cache = DATA_DIR / "slr_bench_basic"
    if not cache.exists():
        raise FileNotFoundError(
            "SLR-Bench Basic is missing. Download AIML-TUDA/SLR-Bench and "
            "save its Basic configuration to data/slr_bench_basic."
        )
    dataset = load_from_disk(str(cache))
    if config.benchmark_split not in dataset:
        raise ValueError(f"unknown SLR-Bench split: {config.benchmark_split}")
    source = dataset[config.benchmark_split]
    count = min(config.num_examples, len(source))
    rows = []
    for source_index in range(count):
        row = dict(source[source_index])
        prompt_hash = hashlib.sha256(
            row["prompt"].encode("utf-8")
        ).hexdigest()
        row["source_index"] = source_index
        row["item_id"] = (
            f"slr-basic-{config.benchmark_split}-{source_index:04d}-"
            f"{prompt_hash[:10]}"
        )
        rows.append(row)
    manifest = {
        "dataset": "AIML-TUDA/SLR-Bench",
        "configuration": "Basic",
        "evaluation_split": config.benchmark_split,
        "dataset_fingerprint": source._fingerprint,
        "item_ids": [row["item_id"] for row in rows],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return rows, manifest


def render_local_evidence(prompt: str) -> str:
    """Normalize one private instance without solving it.

    Args:
        prompt: Official SLR-Bench prompt with labeled ground facts.
    """
    fact_pattern = re.compile(
        r"(?m)^\s*([a-z][A-Za-z0-9_]*\s*\([^()\n]*\))\.\s*$"
    )
    facts = [
        _parse_atom(match.group(1))
        for match in fact_pattern.finditer(prompt)
    ]
    labels = [
        (fact.arguments[0], fact.predicate)
        for fact in facts
        if fact.predicate in {"eastbound", "westbound"}
    ]
    car_links = [
        fact for fact in facts
        if fact.predicate == "has_car"
    ]
    properties = [
        fact for fact in facts
        if fact.predicate not in {"eastbound", "westbound", "has_car"}
    ]
    if not labels or not car_links:
        raise ValueError("SLR prompt does not contain labeled train facts")

    lines = [
        "Locally normalized private evidence:",
        "positive means eastbound; negative means westbound.",
    ]
    for example_index, (train, label) in enumerate(labels, start=1):
        polarity = "positive" if label == "eastbound" else "negative"
        lines.append(f"{polarity} example E{example_index}:")
        cars = [
            fact.arguments[1]
            for fact in car_links
            if fact.arguments[0] == train
        ]
        for car_index, car in enumerate(cars, start=1):
            values = [
                f"{fact.predicate}={fact.arguments[1]}"
                for fact in properties
                if fact.arguments[0] == car and len(fact.arguments) == 2
            ]
            lines.append(f"  car C{car_index}: {'; '.join(values)}")
    lines.extend(
        [
            "",
            "Construct the shortest distinguishing conjunction. In Prolog, "
            "connect each selected car property with has_car(Train, Car).",
        ]
    )
    return "\n".join(lines)


def _student_prompt(
    row: dict,
    guidance: str | None,
    *,
    local_evidence: bool = False,
) -> tuple[str, str | None]:
    prompt = (
        render_local_evidence(row["prompt"])
        if local_evidence
        else row["prompt"].strip()
    )
    if guidance:
        prompt += (
            "\n\nReusable task knowledge from the offline compiler:\n"
            f"{guidance}"
        )
    prompt += (
        "\n\nReturn only one complete eastbound(Train) Prolog rule ending "
        "in a period."
    )
    system = None if guidance is None else STUDENT_SYSTEM_PROMPT
    return prompt, system


def _level_metrics(predictions: list[dict]) -> list[dict]:
    levels = sorted({int(row["curriculum_level"]) for row in predictions})
    metrics = []
    for level in levels:
        subset = [
            row for row in predictions
            if int(row["curriculum_level"]) == level
        ]
        correct = sum(row["semantically_correct"] for row in subset)
        metrics.append(
            {
                "curriculum_level": level,
                "correct": correct,
                "num_examples": len(subset),
                "semantic_accuracy": correct / len(subset),
            }
        )
    return metrics


def execute(config: RunConfig, recorder: RunRecorder, run_dir: Path) -> dict:
    """Evaluate task-level knowledge transfer on SLR-Bench Basic."""
    if config.baseline not in SLR_BASELINES:
        raise ValueError(
            "SLR-Bench supports raw_task, zero_shot_cot, "
            "structured_guidance, slr_local_evidence, and "
            "slr_compiled_guidance"
        )
    if config.guidance_mode != TASK_LEVEL_GUIDANCE:
        raise ValueError("SLR-Bench permits offline task-level guidance only")

    with recorder.stage("load_slr_bench_basic"):
        rows, manifest = _load_rows(config)
    artifacts = run_dir / "artifacts"
    (artifacts / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    guidance = None
    teacher_prompt = None
    teacher_calls = 0
    guidance_source = None
    if config.baseline == STRUCTURED_GUIDANCE:
        from taskcompiler.guidance_track.pipeline import _generate_guidance

        teacher = TeacherClient(
            config.teacher_model,
            backend=config.teacher_backend,
            base_url=config.openrouter_base_url,
            timeout_seconds=config.teacher_timeout_seconds,
            provider=config.teacher_openrouter_provider,
        )
        with recorder.stage("generate_offline_task_guidance"):
            guidance, _ = _generate_guidance(
                teacher,
                PUBLIC_TASK_SPECIFICATION,
                recorder,
                mode=TASK_LEVEL_GUIDANCE,
                style=STRUCTURED_GUIDANCE,
                max_tokens=config.token_budget,
                max_retries=config.max_guidance_retries,
                use_response_format=config.teacher_structured_outputs,
            )
        teacher_prompt = PUBLIC_TASK_SPECIFICATION
        teacher_calls = 1
        guidance_source = "generated_for_run"
    elif config.baseline == SLR_COMPILED_GUIDANCE:
        if config.compiled_guidance_path:
            source_path = Path(config.compiled_guidance_path)
            with recorder.stage("load_offline_task_knowledge"):
                value = validate_compiled_guidance(
                    _extract_json_object(
                        source_path.read_text(encoding="utf-8")
                    )
                )
                guidance = json.dumps(value, ensure_ascii=False, indent=2)
            teacher_prompt = PUBLIC_TASK_SPECIFICATION
            guidance_source = str(source_path.resolve())
        else:
            teacher = TeacherClient(
                config.teacher_model,
                backend=config.teacher_backend,
                base_url=config.openrouter_base_url,
                timeout_seconds=config.teacher_timeout_seconds,
                provider=config.teacher_openrouter_provider,
            )
            with recorder.stage("compile_offline_task_knowledge"):
                guidance, _, teacher_prompt = _compile_guidance(
                    teacher,
                    recorder,
                    config,
                )
            teacher_calls = 1
            guidance_source = "generated_for_run"
    if guidance is not None:
        (artifacts / "offline_guidance.txt").write_text(
            guidance,
            encoding="utf-8",
        )
        privacy_audit = {
            "teacher_received_benchmark_instances": False,
            "teacher_received_target_rules": False,
            "teacher_received_validation_programs": False,
            "teacher_prompt_sha256": hashlib.sha256(
                (teacher_prompt or "").encode("utf-8")
            ).hexdigest(),
            "instance_constant_detected": bool(
                _INSTANCE_CONSTANT.search(teacher_prompt or "")
            ),
            "guidance_sha256": hashlib.sha256(
                guidance.encode("utf-8")
            ).hexdigest(),
            "guidance_source": guidance_source,
        }
        (artifacts / "teacher_privacy_audit.json").write_text(
            json.dumps(privacy_audit, indent=2),
            encoding="utf-8",
        )

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

    predictions = []
    partial_path = artifacts / "predictions.partial.jsonl"
    partial_path.write_text("", encoding="utf-8")
    with recorder.stage("evaluate_slr_rules"):
        for index, row in enumerate(rows, start=1):
            use_local_evidence = config.baseline in {
                SLR_LOCAL_EVIDENCE,
                SLR_COMPILED_GUIDANCE,
            }
            if config.baseline == ZERO_SHOT_COT:
                prompt, _ = _student_prompt(row, None)
                system = STUDENT_SYSTEM_PROMPT
            else:
                prompt, system = _student_prompt(
                    row,
                    guidance,
                    local_evidence=use_local_evidence,
                )
                if use_local_evidence and system is None:
                    system = STUDENT_SYSTEM_PROMPT
            response = student.ask(prompt, system=system)
            recorder.add_student_tokens(
                response.input_tokens,
                response.output_tokens,
            )
            candidate = extract_rule(response.text)
            evaluation = evaluate_rule(
                candidate or "",
                row["validation program"],
            )
            prediction = {
                "item_id": row["item_id"],
                "source_index": row["source_index"],
                "curriculum_level": int(row["curriculum level"]),
                "raw_output": response.text,
                "candidate_rule": candidate,
                "ground_truth_rule": row["ground-truth rule"],
                "syntax_valid": evaluation.valid,
                "semantically_correct": evaluation.semantically_correct,
                "positive_correct": evaluation.positive_correct,
                "positive_total": evaluation.positive_total,
                "negative_correct": evaluation.negative_correct,
                "negative_total": evaluation.negative_total,
                "evaluation_error": evaluation.error,
            }
            predictions.append(prediction)
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(prediction) + "\n")
            recorder.log(
                f"SLR example {index}/{len(rows)}: "
                f"{'passed' if evaluation.semantically_correct else 'failed'}"
            )

    (artifacts / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in predictions),
        encoding="utf-8",
    )
    correct = sum(row["semantically_correct"] for row in predictions)
    valid = sum(row["syntax_valid"] for row in predictions)
    interval = wilson_accuracy(correct, len(predictions))
    metrics = {
        "dataset": "slr_bench",
        "configuration": "Basic",
        "baseline": config.baseline,
        "semantic_accuracy": correct / len(predictions),
        "semantic_accuracy_wilson_95": {
            "low": interval.low,
            "high": interval.high,
            "descriptive_only": True,
        },
        "syntax_validity": valid / len(predictions),
        "correct": correct,
        "num_examples": len(predictions),
        "level_metrics": _level_metrics(predictions),
        "evaluation_split": config.benchmark_split,
        "manifest_sha256": manifest["manifest_sha256"],
        "teacher_received_benchmark_instances": False,
        "teacher_received_target_rules": False,
        "teacher_received_validation_programs": False,
        "teacher_calls": teacher_calls,
        "guidance_source": guidance_source,
        "condition_class": (
            "certificate_oriented_task_knowledge"
            if config.baseline == SLR_COMPILED_GUIDANCE
            else "teacher_free_local_evidence"
            if config.baseline == SLR_LOCAL_EVIDENCE
            else "generic_structured_guidance"
            if config.baseline == STRUCTURED_GUIDANCE
            else "no_teacher_guidance"
        ),
        "judge_scope": "non-recursive conjunctive SLR-Bench Basic",
        "inferential_unit": "SLR-Bench induction problem (descriptive pilot)",
    }
    recorder.set_task_metrics(metrics)
    return metrics
