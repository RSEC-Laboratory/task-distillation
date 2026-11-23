"""Recompute every released result from saved records with Python alone.

This command performs no inference or network access. It does not copy
metrics from supplied summaries: predictions are rescored, budgets checked, and saved DSL reexecuted.
Use run_student.py separately to generate new student responses.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "evaluation"

from . import adapters, controls, matched_gepa, programs, submitted
from .common import VerificationError, require, scoring_functions
from .layout import PROJECT_ROOT, prepare_records


# Output names describe experiments; dictionary keys retain the original report
# interface so the refactoring can be checked by exact JSON comparison.
PANEL_FILES = {
    "submitted": "submitted",
    "current_native_gepa": "matched_gepa",
    "qwen_adapters": "qwen_adapters",
    "new_rule_ablation": "instruction_ablation",
    "new_common_checker": "counting_control",
    "public_example_lookup": "public_lookup",
    "reserved_nlp": "reserved_nlp",
    "program_repair_replay": "program_repairs",
    "teacher_compilation_divergence": "teacher_divergence",
}


def metric_rows(data, prefix=""):
    """Flatten all accuracy records while retaining their full metric paths."""
    if isinstance(data, dict):
        if set(data) >= {"correct", "total", "accuracy"}:
            yield [prefix, data["correct"], data["total"], data["accuracy"]]
        for key, value in data.items():
            yield from metric_rows(value, prefix + ("/" if prefix else "") + str(key))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from metric_rows(value, prefix + "/" + str(index))


def write_results(output: Path, name: str, data: dict) -> None:
    """Write complete JSON and a CSV only when accuracy records are present."""
    (output / (name + ".json")).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = list(metric_rows(data))
    csv_path = output / (name + ".csv")
    if not metrics:
        csv_path.unlink(missing_ok=True)
        return
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["section", "correct", "total", "accuracy"])
        writer.writerows(metrics)


def write_study_tables(output: Path, studies: dict) -> None:
    """Expose the main submitted comparisons without nested metric paths."""
    columns = ["study", "condition", "correct", "total", "accuracy"]
    with (output / "main_studies.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for study in ("main_llama", "main_gemma", "llama_counting"):
            for condition, values in studies[study]["conditions"].items():
                writer.writerow(
                    [
                        study,
                        condition,
                        values["correct"],
                        values["total"],
                        values["accuracy"],
                    ]
                )
    routing = studies["two_task_routing"]
    with (output / "routing.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for condition, correct in routing["correct"].items():
            writer.writerow(
                [
                    "two_task_routing",
                    condition,
                    correct,
                    routing["cases"],
                    correct / routing["cases"],
                ]
            )


def reproduce_all(root: Path, output: Path) -> dict:
    """Validate saved evidence and return the complete, unchanged result schema."""
    root, output = Path(root).resolve(), Path(output).resolve()
    require(
        (root / "src/taskcompiler").is_dir(), "--root must name the project directory"
    )
    prepare_records(root)
    score, hashes = scoring_functions(root)
    result = {
        "status": "passed",
        "mode": "offline saved-record rescoring; no inference or network",
        "scoring_source_sha256": hashes,
    }
    checks = [
        ("submitted", lambda: submitted.reproduce(root)),
        ("current_native_gepa", lambda: matched_gepa.current_gepa(root)),
        ("qwen_adapters", lambda: adapters.qwen_adapters(root, score)),
        ("new_rule_ablation", lambda: controls.original_ablation(root, score)),
        ("new_common_checker", lambda: controls.counting_control(root, score)),
        ("public_example_lookup", lambda: controls.lookup_control(root)),
        ("reserved_nlp", lambda: adapters.reserved(root, score)),
        ("program_repair_replay", lambda: programs.program_repairs(root)),
        ("teacher_compilation_divergence", lambda: programs.teacher_divergence(root)),
    ]
    output.mkdir(parents=True, exist_ok=True)
    for name, check in checks:
        result[name] = check()
        write_results(output, PANEL_FILES[name], result[name])
        print("Passed: " + PANEL_FILES[name], flush=True)
    write_study_tables(output, result["submitted"])
    write_results(output, "results", result)
    # Preserve the aggregate CSV filename used by the existing release interface.
    (output / "results.csv").replace(output / "accuracy.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "evaluation/results/reproduced"
    )
    args = parser.parse_args()
    result = reproduce_all(args.root, args.output)
    print(
        json.dumps(
            {
                "status": "passed",
                "model_generation_calls": 0,
                "output": str(args.output.resolve()),
                "upstream_nlp_source_validation": result["reserved_nlp"][
                    "source_instance_validation"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (VerificationError, FileNotFoundError, KeyError) as error:
        print("Verification failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
