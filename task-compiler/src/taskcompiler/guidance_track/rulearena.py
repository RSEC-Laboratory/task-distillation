"""Evaluate the official RuleArena tasks with their published prompts."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import types
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from taskcompiler.config import (
    DATA_DIR,
    RULEARENA_ONE_SHOT,
    RULEARENA_ZERO_SHOT,
    RunConfig,
)
from taskcompiler.metrics import RunRecorder
from taskcompiler.statistics import wilson_accuracy
from taskcompiler.student import StudentModel


RULEARENA_ROOT = DATA_DIR / "rulearena"
RULEARENA_COMMIT = "3b9e2256294644beca66732babc5e1055855a576"
DOMAINS = ("airline", "nba", "tax")
COMPLEXITIES = (0, 1, 2)


@dataclass(frozen=True)
class RuleArenaExample:
    item_id: str
    domain: str
    complexity: int
    source_index: int
    source: dict
    target: str | float


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} is missing from {path}")


@lru_cache(maxsize=1)
def _airline_evaluator():
    root = RULEARENA_ROOT / "airline"
    path = root / "compute_answer.py"
    spec = importlib.util.spec_from_file_location(
        "_taskcompiler_rulearena_airline",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the official airline evaluator")
    module = importlib.util.module_from_spec(spec)
    previous = Path.cwd()
    try:
        os.chdir(root)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    return module


@lru_cache(maxsize=1)
def _tax_evaluator():
    root = RULEARENA_ROOT / "tax"
    previous = Path.cwd()
    old_path = list(sys.path)
    old_modules = {
        name: sys.modules.get(name)
        for name in ("prompt", "structured_forms", "micro_evaluation")
    }
    added_openai_stub = "openai" not in sys.modules
    if added_openai_stub:
        sys.modules["openai"] = types.ModuleType("openai")
    try:
        os.chdir(root)
        sys.path.insert(0, str(root))
        for name in old_modules:
            sys.modules.pop(name, None)
        structured_forms = importlib.import_module("structured_forms")
        evaluator = importlib.import_module("micro_evaluation")
    finally:
        os.chdir(previous)
        sys.path[:] = old_path
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if added_openai_stub:
            sys.modules.pop("openai", None)
    return structured_forms.TaxPayer, evaluator.compute_answer


def _airline_target(source: dict) -> float:
    evaluator = _airline_evaluator()
    value, _ = evaluator.compute_answer(
        **source["info"],
        check_base_tables=evaluator.check_base_tables,
    )
    return float(value)


def _nba_target(source: dict) -> str:
    if not source["answer"]:
        return "Answer: False"
    return (
        "Answer: True. "
        f"Illegal Operation: {source['illegal_operation']}. "
        f"Problematic Team: {source['problematic_team']}"
    )


def _tax_target(source: dict) -> float:
    tax_payer_class, compute_answer = _tax_evaluator()
    value, _ = compute_answer(tax_payer_class(**source["pydantic"]))
    return float(value)


def load_examples(domain: str, complexity: int) -> list[RuleArenaExample]:
    """Load one official RuleArena domain and difficulty level."""
    if domain not in DOMAINS:
        raise ValueError(f"RuleArena domain must be one of {', '.join(DOMAINS)}")
    if complexity not in COMPLEXITIES:
        raise ValueError("RuleArena complexity must be 0, 1, or 2")

    if domain == "airline":
        path = RULEARENA_ROOT / domain / "synthesized_problems" / f"comp_{complexity}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        target = _airline_target
    elif domain == "nba":
        path = RULEARENA_ROOT / domain / "annotated_problems" / f"comp_{complexity}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        target = _nba_target
    else:
        path = RULEARENA_ROOT / domain / "synthesized_problems" / f"comp_{complexity}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        target = _tax_target

    examples = []
    for index, row in enumerate(rows):
        digest = hashlib.sha256(
            json.dumps(row, sort_keys=True).encode("utf-8")
        ).hexdigest()
        examples.append(
            RuleArenaExample(
                item_id=f"{domain}-c{complexity}-{index:03d}-{digest[:10]}",
                domain=domain,
                complexity=complexity,
                source_index=index,
                source=row,
                target=target(row),
            )
        )
    return examples


def _nba_question(source: dict) -> str:
    return (
        "Team Situations:\n"
        + "\n".join(source["team_situations"])
        + "\n\nPlayer Situations:\n"
        + "\n".join(source["player_situations"])
        + "\n\nOperations:\n"
        + "\n".join(source["operations"])
    )


def _tax_forms(source: dict) -> str:
    tax_root = RULEARENA_ROOT / "tax"
    forms = [
        _literal_assignment(tax_root / "prompt.py", "basic_forms"),
    ]
    payer = copy.deepcopy(source["dict"])
    if payer["itemized"]:
        forms.append(_literal_assignment(tax_root / "prompt.py", "itemized_forms"))
    if payer["self_employed"]:
        forms.append(_literal_assignment(tax_root / "prompt.py", "self_employ_forms"))
    if payer["has_student_loans_or_education_expenses"]:
        forms.append(_literal_assignment(tax_root / "prompt.py", "edu_forms"))
    if payer["child_and_dependent"]:
        forms.append(_literal_assignment(tax_root / "prompt.py", "schedule_8812"))

    rendered = "".join(forms)
    for field, value in payer["data"].items():
        replacement = f"${value:,}" if not isinstance(value, str) else value
        rendered = rendered.replace(f"${field}", replacement)
    rendered = rendered.replace("$TBD", "[__]")
    replacements = {
        "$name": payer["name"],
        "$age": str(payer["age"]),
        "$spouse_age": str(payer["spouse_age"]),
        "$blind": str(payer["blind"]),
        "$spouse_blind": str(payer["spouse_blind"]),
        "$filing_status": payer["filing_status"],
        "$itemized": str(payer["itemized"]),
        "$num_qualifying_children": str(payer["num_qualifying_children"]),
        "$num_other_dependents": str(payer["num_other_dependents"]),
    }
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    return rendered


def render_prompt(
    example: RuleArenaExample,
    baseline: str,
) -> tuple[str, str]:
    """Render the official zero-shot or one-shot RuleArena prompt."""
    if baseline not in {RULEARENA_ZERO_SHOT, RULEARENA_ONE_SHOT}:
        raise ValueError(f"unsupported RuleArena baseline: {baseline}")
    use_example = baseline == RULEARENA_ONE_SHOT
    root = RULEARENA_ROOT / example.domain
    auto_test = root / "auto_test.py"
    system = _literal_assignment(auto_test, "system_prompt")
    prompt = _literal_assignment(auto_test, "prompt_template")

    if example.domain == "airline":
        rules = (root / "reference_rules.txt").read_text(encoding="utf-8")
        one_shot = _literal_assignment(auto_test, "example") if use_example else ""
        prompt = prompt.replace("$reference_rules", rules)
        prompt = prompt.replace("$example_prompt", one_shot)
        prompt = prompt.replace("$question_prompt", example.source["prompt"])
    elif example.domain == "nba":
        rules = (root / "reference_rules.txt").read_text(encoding="utf-8")
        one_shot = (
            _literal_assignment(auto_test, "prompt_example") if use_example else ""
        )
        prompt = prompt.replace("$reference_rules", rules)
        prompt = prompt.replace("$example", one_shot)
        prompt = prompt.replace("$question", _nba_question(example.source))
    else:
        one_shot = (
            _literal_assignment(root / "one_shot_example.py", "example_qwen")
            if use_example
            else ""
        )
        prompt = prompt.replace("$example", one_shot)
        prompt = prompt.replace("$forms", _tax_forms(example.source))
    return system, prompt


def score_response(example: RuleArenaExample, response: str) -> bool:
    """Apply the official problem-level answer criterion."""
    cleaned = response.replace("**", "")
    if example.domain == "airline":
        start = cleaned.find("The total cost is")
        if start == -1:
            return False
        conclusion = cleaned[start:]
        value_index = conclusion.find("$")
        if value_index == -1:
            return False
        value = conclusion[value_index + 1 :].replace(",", "").replace(".", "")
        return value.isnumeric() and int(value) == int(float(example.target))

    if example.domain == "nba":
        return str(example.target) in cleaned

    pattern = (
        r"The total tax (owed|overpaid) is "
        r"\$((?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)."
    )
    match = re.search(pattern, cleaned)
    if not match:
        return False
    value = float(match.group(2).replace(",", ""))
    if match.group(1) == "overpaid":
        value = -value
    return bool(np.isclose(value, float(example.target)))


def _manifest(domain: str, complexity: int, examples: list[RuleArenaExample]) -> dict:
    root = RULEARENA_ROOT / domain
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    payload = {
        "benchmark": "RuleArena",
        "source_repository": "SkyRiver-2000/RuleArena",
        "source_commit": RULEARENA_COMMIT,
        "domain": domain,
        "complexity": complexity,
        "files": {
            str(path.relative_to(RULEARENA_ROOT)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in files
        },
        "items": [example.item_id for example in examples],
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
    """Run an official RuleArena prompt condition."""
    if config.baseline not in {RULEARENA_ZERO_SHOT, RULEARENA_ONE_SHOT}:
        raise ValueError(f"unsupported RuleArena baseline: {config.baseline}")
    with recorder.stage("load_rulearena"):
        complete = load_examples(
            config.benchmark_task,
            config.benchmark_complexity,
        )
        examples = complete[: min(config.num_examples, len(complete))]
        manifest = _manifest(
            config.benchmark_task,
            config.benchmark_complexity,
            complete,
        )
        (run_dir / "artifacts" / "suite_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

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
        max_new_tokens=config.max_answer_tokens,
        context_tokens=config.student_context_tokens,
        enable_thinking=config.student_thinking,
        seed=config.student_seed,
    )
    recorder.set_model_identity("student", student.identity)

    predictions = []
    correct = 0
    with recorder.stage("rulearena_inference"):
        for example in examples:
            system, prompt = render_prompt(example, config.baseline)
            response = student.ask(prompt, system=system)
            matched = score_response(example, response.text)
            correct += int(matched)
            recorder.add_student_tokens(
                response.input_tokens,
                response.output_tokens,
            )
            predictions.append(
                {
                    "item_id": example.item_id,
                    "domain": example.domain,
                    "complexity": example.complexity,
                    "source_index": example.source_index,
                    "target": example.target,
                    "response": response.text,
                    "correct": matched,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                }
            )
    with (run_dir / "artifacts" / "predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction) + "\n")

    accuracy = correct / len(examples) if examples else 0.0
    interval = wilson_accuracy(correct, len(examples))
    metrics = {
        "dataset": "rulearena",
        "domain": config.benchmark_task,
        "complexity": config.benchmark_complexity,
        "baseline": config.baseline,
        "accuracy": accuracy,
        "accuracy_wilson_95": asdict(interval),
        "correct": correct,
        "num_examples": len(examples),
        "official_problem_order": True,
        "official_answer_criterion": True,
        "manifest_sha256": manifest["manifest_sha256"],
    }
    recorder.set_task_metrics(metrics)
    return metrics
