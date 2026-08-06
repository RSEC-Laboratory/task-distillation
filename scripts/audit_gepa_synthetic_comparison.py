"""Recompute the retained synthetic GEPA comparison."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from statistics import mean, stdev

from taskcompiler.statistics import task_bootstrap_difference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    PROJECT_ROOT
    / "protocols"
    / "baseline_artifacts"
    / "gepa_synthetic_development_2026-07-25"
)
MATRIX_ROOT = EVIDENCE_ROOT / "matrices"
RUN_ROOT = EVIDENCE_ROOT / "runs"

SELECTIONS = {
    "compiler": [
        (
            "matrix_2026-07-25_131032_synthetic.json",
            "full_method",
        ),
        (
            "matrix_2026-07-25_134129_synthetic_ablations.json",
            "full_method_cap3",
        ),
        (
            "matrix_2026-07-25_143702_synthetic.json",
            "full_method",
        ),
    ],
    "gepa": [
        ("matrix_2026-07-25_152849_synthetic.json", "gepa"),
        ("matrix_2026-07-25_160112_synthetic.json", "gepa"),
        ("matrix_2026-07-25_163705_synthetic.json", "gepa"),
    ],
}


def _read_json(path: Path) -> dict:
    """Read one retained JSON record."""
    return json.loads(path.read_text(encoding="utf-8"))


def _read_predictions(path: Path) -> list[dict]:
    """Read final prediction records from one retained run."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select_record(matrix_name: str, condition: str) -> dict:
    """Return and validate one condition from a study matrix."""
    matrix = _read_json(MATRIX_ROOT / matrix_name)
    matches = [
        row for row in matrix["results"] if row["condition"] == condition
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{matrix_name} contains {len(matches)} rows for {condition}"
        )
    row = matches[0]
    run_name = PureWindowsPath(row["run_dir"]).name
    run_dir = RUN_ROOT / run_name
    summary = _read_json(run_dir / "summary.json")
    predictions = _read_predictions(
        run_dir / "artifacts" / "predictions.jsonl"
    )

    if summary["task_metrics"] != row["metrics"]:
        raise RuntimeError(f"summary mismatch for {run_name}")
    if len(predictions) != row["metrics"]["num_final_examples"]:
        raise RuntimeError(f"prediction count mismatch for {run_name}")

    correct = sum(bool(item["correct"]) for item in predictions)
    accuracy = correct / len(predictions)
    if abs(accuracy - row["metrics"]["macro_task_accuracy"]) > 1e-12:
        raise RuntimeError(f"prediction accuracy mismatch for {run_name}")

    prediction_scores = {}
    for task in row["metrics"]["task_metrics"]:
        task_rows = [
            item for item in predictions if item["task_id"] == task["task_id"]
        ]
        task_accuracy = sum(bool(item["correct"]) for item in task_rows) / len(
            task_rows
        )
        if abs(task_accuracy - task["accuracy"]) > 1e-12:
            raise RuntimeError(
                f"task accuracy mismatch for {run_name}/{task['task_id']}"
            )
        prediction_scores[task["task_id"]] = task_accuracy

    return {
        "matrix": matrix_name,
        "condition": condition,
        "run": run_name,
        "correct": correct,
        "total": len(predictions),
        "accuracy": accuracy,
        "task_scores": prediction_scores,
        "manifest_sha256": row["metrics"]["manifest_sha256"],
        "student_digest": row["model_identity"]["digest"],
        "student_model": row["model_identity"]["model"],
    }


def main() -> None:
    """Validate raw records and print the recomputed comparison."""
    records = {
        system: [
            _select_record(matrix_name, condition)
            for matrix_name, condition in selections
        ]
        for system, selections in SELECTIONS.items()
    }
    all_records = [row for rows in records.values() for row in rows]

    manifests = {row["manifest_sha256"] for row in all_records}
    student_digests = {row["student_digest"] for row in all_records}
    student_models = {row["student_model"] for row in all_records}
    task_sets = {
        tuple(sorted(row["task_scores"]))
        for row in all_records
    }
    if not (
        len(manifests)
        == len(student_digests)
        == len(student_models)
        == len(task_sets)
        == 1
    ):
        raise RuntimeError("the compared records are not matched")

    task_means = {
        system: {
            task_id: mean(row["task_scores"][task_id] for row in rows)
            for task_id in rows[0]["task_scores"]
        }
        for system, rows in records.items()
    }
    interval = task_bootstrap_difference(
        task_means["compiler"],
        task_means["gepa"],
        samples=10_000,
        seed=0,
    )

    output = {
        "evidence_root": str(EVIDENCE_ROOT.relative_to(PROJECT_ROOT)),
        "matched": {
            "manifest_sha256": next(iter(manifests)),
            "student_model": next(iter(student_models)),
            "student_digest": next(iter(student_digests)),
            "tasks": list(next(iter(task_sets))),
            "runs_per_method": 3,
            "final_examples_per_run": all_records[0]["total"],
        },
        "systems": {
            system: {
                "run_correct": [row["correct"] for row in rows],
                "run_total": [row["total"] for row in rows],
                "run_accuracy": [row["accuracy"] for row in rows],
                "mean_accuracy": mean(row["accuracy"] for row in rows),
                "run_accuracy_sd": stdev(row["accuracy"] for row in rows),
                "mean_task_accuracy": task_means[system],
            }
            for system, rows in records.items()
        },
        "compiler_minus_gepa": {
            "estimate": interval.estimate,
            "bootstrap_95": [interval.low, interval.high],
            "bootstrap_unit": "policy task after averaging three runs",
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 0,
        },
        "source_records": records,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
