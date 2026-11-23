"""Paired summaries of persisted independent evaluation runs.

All intended cases remain in the denominator. A missing prediction in a failed
or partial run is an operational failure, not a fabricated model response.
Bootstrap units are task IDs; repeated runs never become independent tasks.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .experiment import METHODS, OPTIMIZER_METHODS, normalized_input_hash, score_output, verify_bundle
from .schema import content_hash, strict_json_loads, validate_output


_USAGE_FIELDS = (
    "attempted_calls", "failed_calls", "input_tokens", "output_tokens",
    "known_input_tokens", "known_output_tokens", "token_usage_complete", "latency_seconds",
)


def _read(path: Path):
    return strict_json_loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _usage_columns(usage: dict | None, prefix: str, path: Path) -> dict:
    """Retain unknown costs as null; known token subtotals are lower bounds."""
    _require(usage is None or isinstance(usage, dict), f"{path}: malformed {prefix} usage")
    values = {key: (usage or {}).get(key) for key in _USAGE_FIELDS}
    for key, value in values.items():
        if value is None:
            continue
        if key == "token_usage_complete":
            valid = type(value) is bool
        elif key == "latency_seconds":
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        else:
            valid = type(value) is int and value >= 0
        _require(valid, f"{path}: invalid {prefix} {key}")
    if values["token_usage_complete"] is True:
        _require(values["input_tokens"] is not None and values["output_tokens"] is not None,
                 f"{path}: complete {prefix} token usage is missing totals")
    if values["token_usage_complete"] is False:
        _require(values["input_tokens"] is None and values["output_tokens"] is None,
                 f"{path}: incomplete {prefix} token usage cannot claim complete totals")
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _checked_run(path: Path, expected: list[str]) -> tuple[dict | None, dict | None]:
    manifest = _read(path / "manifest.json")
    _require(manifest.get("mode") == "live", f"{path}: only live runs are eligible")
    _require(manifest.get("status") in {"completed", "partial", "failed"},
             f"{path}: run must have a terminal status")
    _require(manifest.get("method") is None or manifest["method"] in expected,
             f"{path}: unexpected method in manifest")
    missing = [name for name in ("bundle.json", "evaluation.json") if not (path / name).exists()]
    if missing:
        _require(manifest["status"] == "failed", f"{path}: missing artifacts {missing}")
        return None, {"directory": str(path), "status": "unscorable",
                      "reason": f"Failed run lacks {', '.join(missing)}",
                      "task_id": manifest.get("task_id"), "method": manifest.get("method")}
    bundle, evaluation = _read(path / "bundle.json"), _read(path / "evaluation.json")
    verify_bundle(bundle)
    task, method, identity = bundle["task_id"], bundle["method"], bundle["student_identity"]
    compilation_status = bundle.get("compile_status")
    _require(compilation_status in {"completed", "partial", "failed", "budget_stopped"},
             f"{path}: unrecognized compilation status")
    _require(compilation_status != "budget_stopped" or method in OPTIMIZER_METHODS,
             f"{path}: budget_stopped is an official-optimizer compilation state")
    if "compilation_status" in manifest:
        _require(manifest["compilation_status"] == compilation_status,
                 f"{path}: manifest compilation_status mismatch")
    _require(method in expected, f"{path}: unexpected method {method!r}")
    _require(bundle.get("mode") == "live" and identity.get("mode") == "live"
             and identity.get("local_only") is True, f"{path}: fixture/nonlocal student is ineligible")
    for key, value in (("task_id", task), ("method", method),
                       ("student_identity", identity), ("bundle_hash", bundle["bundle_hash"])):
        _require(manifest.get(key) == value, f"{path}: manifest {key} mismatch")
    evaluation_hash = content_hash(evaluation)
    _require(manifest.get("evaluation_hash") == evaluation_hash, f"{path}: evaluation hash mismatch")
    _require(evaluation.get("task_id") == task and evaluation.get("reference_kind") == "independent"
             and bool(evaluation.get("source")), f"{path}: independent evaluation provenance required")
    cases = evaluation.get("cases")
    _require(isinstance(cases, list) and bool(cases), f"{path}: nonempty evaluation cases required")
    case_map, input_hashes = {}, set()
    for case in cases:
        _require(isinstance(case, dict) and isinstance(case.get("id"), str)
                 and isinstance(case.get("input"), str) and "reference" in case,
                 f"{path}: malformed evaluation case")
        _require(case["id"] not in case_map, f"{path}: duplicate evaluation case ID")
        _require(bool(case["input"].strip()), f"{path}: empty evaluation input")
        normalized = normalized_input_hash(case["input"])
        _require(normalized not in input_hashes, f"{path}: normalized duplicate evaluation input")
        input_hashes.add(normalized)
        validate_output(case["reference"], bundle["output_schema"])
        _require(normalized not in bundle["generated_input_hashes"],
                 f"{path}: evaluation overlaps generated inputs")
        case_map[case["id"]] = case
    _require(manifest.get("expected_case_count") == len(cases), f"{path}: expected case count mismatch")
    predictions_path, metrics_path = path / "predictions.json", path / "metrics.json"
    _require(predictions_path.exists() or manifest["status"] == "failed",
             f"{path}: predictions missing from a nonfailed run")
    predictions = _read(predictions_path) if predictions_path.exists() else []
    _require(isinstance(predictions, list), f"{path}: predictions must be a list")
    observed, correct, valid, failed = set(), 0, 0, 0
    case_scores = {}
    for prediction in predictions:
        _require(isinstance(prediction, dict), f"{path}: malformed prediction row")
        case_id = prediction.get("case_id")
        _require(case_id in case_map and case_id not in observed,
                 f"{path}: duplicate or unknown prediction case")
        observed.add(case_id)
        case = case_map[case_id]
        input_hash = hashlib.sha256(case["input"].encode("utf-8")).hexdigest()
        _require(prediction.get("input_hash") == input_hash, f"{path}: prediction input hash mismatch")
        _require(prediction.get("phase") == "external_evaluation"
                 and prediction.get("candidate_id") == bundle["selected"]["id"],
                 f"{path}: prediction condition mismatch")
        errored = prediction.get("error") is not None
        _require(errored or isinstance(prediction.get("output"), str),
                 f"{path}: successful prediction needs textual output")
        score = {"correct": False, "format_valid": False} if errored else score_output(
            prediction.get("output"), case["reference"], bundle["output_schema"]
        )
        _require(prediction.get("correct") is score["correct"]
                 and prediction.get("format_valid") is score["format_valid"],
                 f"{path}: prediction scores disagree with rescored output")
        correct += score["correct"]
        valid += score["format_valid"]
        failed += errored
        case_scores[case_id] = int(score["correct"])
    missing_count = len(cases) - len(observed)
    _require(manifest["status"] != "completed" or not (missing_count or failed),
             f"{path}: completed run contains missing/failed predictions")
    result = {}
    if metrics_path.exists():
        result = _read(metrics_path)
        for key, value in (("task_id", task), ("method", method),
                           ("bundle_hash", bundle["bundle_hash"]), ("evaluation_hash", evaluation_hash),
                           ("student_identity", identity), ("status", manifest["status"])):
            _require(result.get(key) == value, f"{path}: metrics {key} mismatch")
        metric = result["metrics"]
        _require(metric.get("mode") == "live" and metric.get("reference_kind") == "independent"
                 and metric.get("eligible_empirical_evidence") is True,
                 f"{path}: metrics are not eligible independent live evidence")
        for key, value in (("n", len(predictions)), ("correct", correct),
                           ("format_valid", valid), ("failed", failed)):
            _require(type(metric.get(key)) is int and metric[key] == value,
                     f"{path}: reported {key} disagrees with predictions")
        for key, value in (("exact_match", correct / len(predictions) if predictions else None),
                           ("format_validity", valid / len(predictions) if predictions else None)):
            reported = metric.get(key)
            _require(reported is None if value is None else
                     type(reported) in (int, float) and math.isclose(reported, value, abs_tol=1e-12),
                     f"{path}: reported {key} disagrees with predictions")
    else:
        _require(manifest["status"] == "failed", f"{path}: metrics missing from a nonfailed run")
    usage = result.get("usage")
    if usage is not None:
        _require(isinstance(usage, dict), f"{path}: malformed inference usage")
        for key, value in (("attempted_calls", len(predictions)), ("failed_calls", failed)):
            _require(usage.get(key) is None or usage[key] == value,
                     f"{path}: inference {key} disagrees with predictions")
    costs = {**_usage_columns(bundle.get("usage"), "compilation", path),
             **_usage_columns(usage, "inference", path)}
    for case_id in case_map.keys() - observed:
        case_scores[case_id] = 0
    student_hash = content_hash(identity)
    # Same task pack and compilation configuration are required for matched arms.
    repeat = content_hash({"task_pack_hash": bundle["task_pack_hash"], "config": bundle["config"]})
    unit = content_hash([task, evaluation_hash, student_hash, repeat])
    return {
        "directory": str(path), "task_id": task, "method": method,
        "student_hash": student_hash, "student_identity": identity,
        "comparison_unit": unit, "evaluation_hash": evaluation_hash,
        "bundle_hash": bundle["bundle_hash"], "status": manifest["status"],
        "compilation_status": compilation_status,
        "n": len(cases), "observed_predictions": len(predictions),
        "missing_predictions": missing_count, "failed_predictions": failed,
        "operational_failures": failed + missing_count, "correct": correct,
        "format_valid": valid, "exact_match": correct / len(cases),
        "format_validity": valid / len(cases), "_case_scores": case_scores, **costs,
    }, None


def _interval(values: list[float], seed: int, samples: int) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    def quantile(q):
        position = (len(draws) - 1) * q
        lower = math.floor(position)
        return draws[lower] + (draws[math.ceil(position)] - draws[lower]) * (position - lower)
    return [quantile(0.025), quantile(0.975)]


def build_report(evaluation_dirs: list[Path], expected_methods: list[str],
                 bootstrap_seed: int = 0, bootstrap_samples: int = 2000) -> dict:
    """Validate runs, re-score predictions, and aggregate repeats before tasks.

    Decoding seeds are part of the full student identity, so different seeds
    remain separate strata. Pairing additionally requires the same evaluation
    dataset, generated task pack, and compilation configuration.
    """
    _require(bool(evaluation_dirs), "At least one evaluation directory is required")
    _require(isinstance(expected_methods, list) and bool(expected_methods)
             and len(set(expected_methods)) == len(expected_methods)
             and set(expected_methods) <= {*METHODS, *OPTIMIZER_METHODS},
             "Expected methods must be distinct registered methods")
    _require(type(bootstrap_seed) is int and type(bootstrap_samples) is int
             and bootstrap_samples >= 100, "Bootstrap needs an integer seed and at least 100 samples")
    paths = [Path(path).resolve() for path in evaluation_dirs]
    _require(len(set(paths)) == len(paths), "Duplicate evaluation directories")
    runs, unscorable, units = [], [], defaultdict(dict)
    for path in paths:
        run, issue = _checked_run(path, expected_methods)
        if issue:
            unscorable.append(issue)
            continue
        methods = units[run["comparison_unit"]]
        _require(run["method"] not in methods, f"{path}: duplicate experiment condition")
        methods[run["method"]] = run
        runs.append(run)
    missing = []
    for unit, arms in sorted(units.items()):
        absent = [method for method in expected_methods if method not in arms]
        if absent:
            sample = next(iter(arms.values()))
            missing.append({"comparison_unit": unit, "task_id": sample["task_id"],
                            "student_hash": sample["student_hash"], "missing_methods": absent})
    groups = defaultdict(list)
    for run in runs:
        groups[(run["student_hash"], run["task_id"], run["method"])].append(run)
    per_task = [
        {"student_hash": student, "task_id": task, "method": method, "runs": len(rows),
         "exact_match": mean(row["exact_match"] for row in rows),
         "format_validity": mean(row["format_validity"] for row in rows),
         "operational_failures": sum(row["operational_failures"] for row in rows)}
        for (student, task, method), rows in sorted(groups.items())
    ]
    macro_groups = defaultdict(list)
    for row in per_task:
        macro_groups[(row["student_hash"], row["method"])].append(row)
    macro = [
        {"student_hash": student, "method": method, "task_count": len(rows),
         "task_ids": [row["task_id"] for row in rows],
         "exact_match": mean(row["exact_match"] for row in rows),
         "format_validity": mean(row["format_validity"] for row in rows)}
        for (student, method), rows in sorted(macro_groups.items())
    ]
    paired = []
    for student in sorted({run["student_hash"] for run in runs}):
        for left, right in itertools.combinations(expected_methods, 2):
            deltas = defaultdict(list)
            matched_units = 0
            for arms in units.values():
                if left not in arms or right not in arms or arms[left]["student_hash"] != student:
                    continue
                a, b = arms[left], arms[right]
                _require(a["_case_scores"].keys() == b["_case_scores"].keys(),
                         "Paired conditions have different case IDs")
                deltas[a["task_id"]].append(mean(
                    b["_case_scores"][case] - a["_case_scores"][case] for case in a["_case_scores"]
                ))
                matched_units += 1
            task_deltas = {task: mean(values) for task, values in sorted(deltas.items())}
            values = list(task_deltas.values())
            paired.append({
                "student_hash": student, "left_method": left, "right_method": right,
                "delta_definition": "right_minus_left", "matched_units": matched_units,
                "task_count": len(values), "per_task_deltas": task_deltas,
                "mean_delta": mean(values) if values else None,
                "ci95": _interval(values, bootstrap_seed, bootstrap_samples),
            })
    for run in runs:
        run.pop("_case_scores")
    return {
        "version": 1, "status": ("incomplete" if missing or unscorable else
                                  "partial" if any(
                                      row["status"] != "completed"
                                      or row["compilation_status"] in {"partial", "failed"}
                                      or (row["compilation_failed_calls"] or 0) > 0
                                      for row in runs
                                  ) else "complete"),
        "mode": "live", "reference_kind": "independent",
        "metric": "typed_exact_match", "expected_methods": expected_methods,
        "denominator_policy": "All intended evaluation cases; missing/error predictions count as operational failures and incorrect.",
        "bootstrap": {"cluster": "task_id", "seed": bootstrap_seed, "samples": bootstrap_samples,
                      "minimum_independent_tasks": 2},
        "runs": runs, "unscorable_runs": unscorable, "missing_conditions": missing,
        "per_task": per_task, "macro": macro, "paired_deltas": paired,
        "limitations": [
            "Reference independence is declared by the dataset source, not automatically established.",
            "Macro summaries may cover different tasks when conditions are missing; use matched paired deltas.",
            "Different student identities, including decoding seeds, are not pooled.",
            "Missing predictions are operational failures, not observed model responses.",
            "Compilation usage covers local student search; remote teacher preparation costs are not included.",
            "GEPA/MIPRO budget_stopped records planned search-budget termination; it is not an operational failure unless failed calls are recorded.",
        ],
    }


def write_report(report: dict, outdir: Path) -> None:
    """Write new report artifacts without overwriting an existing report directory."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = ["directory", "task_id", "method", "student_hash", "evaluation_hash",
               "bundle_hash", "comparison_unit", "status", "compilation_status", "n", "observed_predictions",
               "missing_predictions", "failed_predictions", "operational_failures",
               "correct", "format_valid", "exact_match", "format_validity"]
    columns += [f"{phase}_{field}" for phase in ("compilation", "inference")
                for field in _USAGE_FIELDS]
    with (outdir / "per_run.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["runs"])
