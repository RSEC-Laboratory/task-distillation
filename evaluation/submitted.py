"""Recompute the submitted historical table panels from retained predictions.

These are historical measurements, including the registered Gemma fallback
estimates and the documented NLP identical-prompt fallback correction. They are
kept separate from the later control experiments.
"""

from __future__ import annotations

from collections import defaultdict
import json
import re

from .bootstrap import historical_bootstrap as bootstrap, mcnemar
from .common import (
    INSTRUCTION_RECORDS,
    SUBMITTED_RECORDS,
    accuracy,
    read,
    require as need,
    rows as load_rows,
)
from .layout import resolve_path


def close(value: float, expected: float, tolerance: float = 1e-12) -> None:
    need(abs(value - expected) <= tolerance, f"{value} != {expected}")


def main_study(root) -> dict:
    rows = load_rows(
        resolve_path(root, root / SUBMITTED_RECORDS / "main_predictions.jsonl")
    )
    runs = read(resolve_path(root, root / SUBMITTED_RECORDS / "main_runs.json"))
    tasks = sorted({row["task"] for row in rows})
    conditions = ("raw", "one_shot", "full")
    need(len(tasks) == 24, "main task count changed")
    need(len(rows) == 24 * 3 * 27, "main prediction count changed")
    scores = defaultdict(int)
    grouped = defaultdict(list)
    for row in rows:
        need(
            bool(row["correct"]) == (row["predicted"] == row["gold"]),
            "main correctness field changed",
        )
        key = (row["task"], row["condition"])
        scores[key] += int(row["correct"])
        grouped[key].append(row)
    for task in tasks:
        reference = {
            (row["item_id"], json.dumps(row["input"], sort_keys=True), row["gold"])
            for row in grouped[task, "raw"]
        }
        for condition in conditions:
            values = grouped[task, condition]
            need(len(values) == 27, "main task-condition size changed")
            paired = {
                (row["item_id"], json.dumps(row["input"], sort_keys=True), row["gold"])
                for row in values
            }
            need(paired == reference, "main conditions are not paired")
    run_index = {(record["task"], record["condition"]): record for record in runs}
    need(len(run_index) == 72, "main run metadata changed")
    for task in tasks:
        records = [run_index[task, condition] for condition in conditions]
        need(
            len({record["manifest_sha256"] for record in records}) == 1,
            "main split manifest mismatch",
        )
        need(
            len({record["source_snapshot_sha256"] for record in records}) == 1,
            "main source snapshot mismatch",
        )
        need(
            len({record["student"]["digest"] for record in records}) == 1,
            "main student digest mismatch",
        )
        initial = {
            run_index[task, condition]["policy"]["initial_program_sha256"]
            for condition in ("one_shot", "full")
        }
        need(len(initial) == 1, "one-shot did not reuse the initial artifact")
        need(
            not run_index[task, "full"]["policy"]["teacher_received_final_cases"],
            "teacher received a final case",
        )
    totals = {
        condition: sum((scores[task, condition] for task in tasks))
        for condition in conditions
    }
    need(totals == {"raw": 107, "one_shot": 311, "full": 517}, "main totals changed")
    differences = {
        condition: [
            (scores[task, "full"] - scores[task, condition]) / 27 for task in tasks
        ]
        for condition in ("raw", "one_shot")
    }
    versus_raw = bootstrap(differences["raw"], 100000, 20260727)
    versus_one = bootstrap(differences["one_shot"], 100000, 20260727)
    for observed, expected in zip(
        versus_raw, (0.6327160493827161, 0.5509259259259259, 0.7021604938271605)
    ):
        close(observed, expected)
    for observed, expected in zip(
        versus_one, (0.3179012345679012, 0.26080246913580246, 0.3703703703703704)
    ):
        close(observed, expected)
    full_runs = [run_index[task, "full"] for task in tasks]
    certified = sum(
        (
            record["source_verification"]["build"] == 1.0
            and record["source_verification"]["audit"] == 1.0
            for record in full_runs
        )
    )
    wins = sum(
        (
            scores[task, "full"] > scores[task, "raw"]
            and scores[task, "full"] > scores[task, "one_shot"]
            for task in tasks
        )
    )
    corrections = sum(
        (record["source_verification"]["patches"] for record in full_runs)
    )
    corrected_tasks = sum(
        (record["source_verification"]["patches"] > 0 for record in full_runs)
    )
    teacher_calls = sum((record["teacher_calls"] for record in full_runs))
    student_calls = sum((sum(record["student_calls"].values()) for record in runs))
    fallback_tasks = [
        record["task"] for record in full_runs if record["compilation_failure"]
    ]
    need(
        (
            certified,
            wins,
            corrections,
            corrected_tasks,
            teacher_calls,
            student_calls,
            fallback_tasks,
        )
        == (23, 23, 11, 6, 61, 6372, [62]),
        "main process counts changed",
    )
    need(
        all((record["remote_student_calls"] == 0 for record in runs)),
        "a remote student call appears in the main records",
    )
    family_ranges = {}
    for condition in ("raw", "one_shot"):
        means = [
            sum(
                (
                    differences[condition][index]
                    for index, task in enumerate(tasks)
                    if task % 4 == family
                )
            )
            / sum((task % 4 == family for task in tasks))
            for family in range(4)
        ]
        family_ranges[condition] = [min(means), max(means)]
    return {
        "tasks": len(tasks),
        "final_cases_per_condition": 648,
        "correct": totals,
        "accuracy": {condition: totals[condition] / 648 for condition in conditions},
        "full_minus_raw": {
            "estimate": versus_raw[0],
            "bootstrap_95": list(versus_raw[1:]),
        },
        "full_minus_one_shot": {
            "estimate": versus_one[0],
            "bootstrap_95": list(versus_one[1:]),
        },
        "certified_tasks": certified,
        "full_higher_than_both": wins,
        "teacher_calls": teacher_calls,
        "deterministic_corrections": corrections,
        "tasks_with_corrections": corrected_tasks,
        "fallback_tasks": fallback_tasks,
        "student_calls": student_calls,
        "family_gain_ranges": family_ranges,
        "scope": "24 generated instances from four rule families",
    }


def gemma_study(root) -> dict:
    rows = load_rows(
        resolve_path(root, root / SUBMITTED_RECORDS / "gemma_predictions.jsonl")
    )
    records = read(resolve_path(root, root / SUBMITTED_RECORDS / "gemma_runs.json"))
    tasks = sorted({row["task"] for row in rows})
    failures = {
        record["task"] for record in records if record.get("status") == "syntax_failure"
    }
    need(len(tasks) == 18 and failures == {68, 72}, "Gemma task set changed")
    scores = defaultdict(int)
    totals = defaultdict(int)
    for row in rows:
        need(
            bool(row["correct"]) == (row["predicted"] == row["gold"]),
            "Gemma correctness field changed",
        )
        scores[row["task"], row["condition"]] += int(row["correct"])
        totals[row["task"], row["condition"]] += 1
    for task in failures:
        need(totals[task, "raw"] == 27, "Gemma fallback source changed")
        for condition in ("one_shot", "full"):
            scores[task, condition] = scores[task, "raw"]
            totals[task, condition] = totals[task, "raw"]
    for task in tasks:
        for condition in ("raw", "one_shot", "full"):
            need(totals[task, condition] == 27, "Gemma pairing changed")
    correct = {
        condition: sum((scores[task, condition] for task in tasks))
        for condition in ("raw", "one_shot", "full")
    }
    need(correct == {"raw": 202, "one_shot": 264, "full": 419}, "Gemma totals changed")
    intervals = {}
    for condition, expected in (
        ("raw", (0.4465020576131687, 0.33539094650205764, 0.5452674897119342)),
        ("one_shot", (0.31893004115226337, 0.23868312757201643, 0.3909465020576131)),
    ):
        differences = [
            (scores[task, "full"] - scores[task, condition]) / 27 for task in tasks
        ]
        interval = bootstrap(differences, 100000, 20260727)
        for observed, target in zip(interval, expected):
            close(observed, target)
        intervals[condition] = {
            "estimate": interval[0],
            "bootstrap_95": list(interval[1:]),
        }
    complete_full = [record for record in records if record.get("condition") == "full"]
    certified = sum(
        (
            record["source_verification"]["build"] == 1.0
            and record["source_verification"]["audit"] == 1.0
            for record in complete_full
        )
    )
    wins = sum(
        (
            scores[task, "full"] > scores[task, "raw"]
            and scores[task, "full"] > scores[task, "one_shot"]
            for task in tasks
        )
    )
    need((certified, wins) == (15, 15), "Gemma task counts changed")
    return {
        "tasks": 18,
        "complete_task_triplets": 16,
        "imputed_fallback_tasks": sorted(failures),
        "cases_per_condition_after_fallback": 486,
        "correct": correct,
        "accuracy": {condition: correct[condition] / 486 for condition in correct},
        "full_minus_controls": intervals,
        "certified_complete_tasks": certified,
        "full_higher_than_both": wins,
        "scope": "conservative registered-fallback sensitivity",
    }


def routing_study(root) -> dict:
    rows = load_rows(
        resolve_path(root, root / SUBMITTED_RECORDS / "routing_predictions.jsonl")
    )
    runs = read(resolve_path(root, root / SUBMITTED_RECORDS / "routing_runs.json"))
    scores = defaultdict(int)
    paired = defaultdict(set)
    delivered = []
    for row in rows:
        need(
            bool(row["correct"]) == (row["predicted"] == row["gold"]),
            "Routing rescoring differs",
        )
        scores[row["condition"]] += int(row["correct"])
        paired[row["condition"]].add(
            (
                row["task"],
                row["item_id"],
                json.dumps(row["input"], sort_keys=True),
                row["gold"],
            )
        )
        if row["condition"] == "routed":
            delivered.append(row["delivered_instructions"])
    need(paired["full_list"] == paired["routed"], "routing cases changed")
    need(scores == {"full_list": 25, "routed": 43}, "routing totals changed")
    for task in (0, 26):
        hashes = {record["bundle_sha256"] for record in runs if record["task"] == task}
        need(len(hashes) == 1, "routing bundles do not match")
    available = (
        sum(
            (
                record["available_instructions"]
                for record in runs
                if record["condition"] == "full_list"
            )
        )
        / 2
    )
    mean_delivered = sum(delivered) / len(delivered)
    reduction = 1 - mean_delivered / available
    close(available, 29.0)
    close(mean_delivered, 7.62962962962963)
    close(reduction, 0.7369093231162197)
    return {
        "tasks": 2,
        "cases": 54,
        "correct": dict(scores),
        "mean_available_instructions": available,
        "mean_delivered_instructions": mean_delivered,
        "instruction_reduction": reduction,
        "scope": "development ablation with identical bundles",
    }


def textgrad_study(root) -> dict:
    rows = load_rows(
        resolve_path(root, root / SUBMITTED_RECORDS / "textgrad_predictions.jsonl")
    )
    prompt_rows = load_rows(
        resolve_path(
            root,
            root / SUBMITTED_RECORDS / "textgrad_prompt_predictions.jsonl",
        )
    )
    metadata = read(
        resolve_path(root, root / SUBMITTED_RECORDS / "textgrad_metadata.json")
    )
    interpretation = metadata["interpretation"]
    need(
        interpretation["comparison_type"] == "matched local TextGrad adaptation",
        "TextGrad comparison type changed",
    )
    need(
        interpretation["not_a_published_score_reproduction"],
        "TextGrad score source changed",
    )
    close(interpretation["published_table_3"]["word_sorting"]["cot_zero_shot"], 0.767)
    close(interpretation["published_table_3"]["word_sorting"]["textgrad"], 0.798)
    close(
        interpretation["published_table_3"]["object_counting"]["cot_zero_shot"], 0.778
    )
    close(interpretation["published_table_3"]["object_counting"]["textgrad"], 0.919)
    need(len(rows) == 100, "Object Counting test size changed")
    need(len(prompt_rows) == 100, "prompt-transfer test size changed")
    method = [bool(row["method_correct"]) for row in rows]
    baseline = [bool(row["textgrad_correct"]) for row in rows]
    for row in rows:
        need(
            bool(row["method_correct"]) == (row["method_predicted"] == row["gold"]),
            "method scoring changed",
        )
        need(
            bool(row["textgrad_correct"]) == (row["textgrad_predicted"] == row["gold"]),
            "TextGrad scoring changed",
        )
    by_id = {row["item_id"]: row for row in rows}
    for row in prompt_rows:
        need(
            row["item_id"] in by_id
            and row["gold"] == by_id[row["item_id"]]["gold"]
            and (row["source_index"] == by_id[row["item_id"]]["source_index"]),
            "prompt-transfer case pairing changed",
        )
        need(
            bool(row["correct"]) == (row["predicted"] == row["gold"]),
            "prompt-transfer scoring changed",
        )
    gains, losses, probability = mcnemar(method, baseline)
    need(
        (sum(method), sum(baseline), gains, losses) == (85, 75, 17, 7),
        "Object Counting result changed",
    )
    close(probability, 0.06391465663909912)
    need(sum((row["correct"] for row in prompt_rows)) == 76, "prompt score changed")
    need(
        sum((row["input_tokens"] for row in prompt_rows)) == 15615
        and sum((row["output_tokens"] for row in prompt_rows)) == 11990,
        "prompt token record changed",
    )
    need(
        metadata["method_student"]["digest"] == metadata["textgrad_student"]["digest"],
        "Object Counting student weights differ",
    )
    need(
        metadata["prompt_transfer"]["student_digest"]
        == metadata["method_student"]["digest"],
        "prompt-transfer student weights differ",
    )
    need(
        metadata["method_tokens"] == {"teacher_total": 349, "student_test": 39800},
        "method token record changed",
    )
    need(
        metadata["textgrad_tokens"]["teacher_total"] == 145921
        and metadata["textgrad_tokens"]["student_total"] == 592241,
        "TextGrad token record changed",
    )
    need(
        metadata["protocol"]["test_split_previously_explored"],
        "Object Counting exposure disclosure changed",
    )
    return {
        "task": "BBH Object Counting",
        "test_examples": 100,
        "method_correct": sum(method),
        "textgrad_correct": sum(baseline),
        "difference": (sum(method) - sum(baseline)) / 100,
        "comparison_type": interpretation["comparison_type"],
        "published_textgrad_score": 0.919,
        "published_score_reproduction": False,
        "prompt_transfer_correct": sum((row["correct"] for row in prompt_rows)),
        "paired_gains": gains,
        "paired_losses": losses,
        "exact_mcnemar_two_sided": probability,
        "tokens": {
            "method_teacher": 349,
            "method_student_test": 39800,
            "textgrad_teacher": 145921,
            "textgrad_student_total": 592241,
        },
        "scope": "one previously explored test task; descriptive comparison",
    }


def normalise_label(value: str) -> str:
    cleaned = value.strip().strip("`\"' ")
    cleaned = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:|is|:)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.splitlines()[0].strip().strip("`\"' ")
    return re.sub(r"[\s.]+$", "", cleaned).casefold()


def superni_study(root) -> dict:
    rows = load_rows(
        resolve_path(root, root / SUBMITTED_RECORDS / "superni_predictions.jsonl")
    )
    metadata = read(
        resolve_path(root, root / SUBMITTED_RECORDS / "superni_metadata.json")
    )
    need(len(rows) == 800, "SuperNI prediction count changed")
    for row in rows:
        rescored = any(
            (
                normalise_label(row["response"]) == normalise_label(reference)
                for reference in row["references"]
            )
        )
        need(rescored == row["correct"], "SuperNI scoring changed")
    result = {}
    for family in ("llama", "gemma"):
        family_rows = [row for row in rows if row["student_family"] == family]
        family_meta = next((row for row in metadata if row["student_family"] == family))
        fallback = set(family_meta["fallback_tasks"])
        index = {
            (row["task_id"], row["item_id"], row["condition"]): row
            for row in family_rows
        }
        positive = sum(
            (
                row["correct"]
                for row in family_rows
                if row["condition"] == "positive_examples"
            )
        )
        recorded_routed = sum(
            (
                row["correct"]
                for row in family_rows
                if row["condition"] == "validated_routed"
            )
        )
        corrected_routed = sum(
            (
                (
                    index[task, item, "positive_examples"]
                    if task in fallback
                    else index[task, item, "validated_routed"]
                )["correct"]
                for task, item, condition in index
                if condition == "validated_routed"
            )
        )
        result[family] = {
            "tasks": 4,
            "cases": 80,
            "positive_examples_correct": positive,
            "recorded_routed_correct": recorded_routed,
            "identical_prompt_corrected_routed": corrected_routed,
            "uncertified_tasks": len(fallback),
        }
    need(
        result
        == {
            "llama": {
                "tasks": 4,
                "cases": 80,
                "positive_examples_correct": 38,
                "recorded_routed_correct": 34,
                "identical_prompt_corrected_routed": 35,
                "uncertified_tasks": 3,
            },
            "gemma": {
                "tasks": 4,
                "cases": 80,
                "positive_examples_correct": 53,
                "recorded_routed_correct": 50,
                "identical_prompt_corrected_routed": 50,
                "uncertified_tasks": 2,
            },
        },
        "SuperNI result changed",
    )
    return {
        "students": result,
        "pooled_positive_examples_correct": 91,
        "pooled_corrected_routed_correct": 85,
        "pooled_cases": 160,
        "scope": "negative four-task development gates; confirmation stayed sealed",
    }


def bbh_development_study(root) -> dict:
    rows = load_rows(
        resolve_path(
            root,
            root / SUBMITTED_RECORDS / "bbh_development_predictions.jsonl",
        )
    )
    metadata = read(
        resolve_path(
            root,
            root / SUBMITTED_RECORDS / "bbh_development_metadata.json",
        )
    )
    scores = defaultdict(int)
    totals = defaultdict(int)
    cases = defaultdict(set)
    for row in rows:
        need(
            bool(row["correct"]) == (row["predicted"] == row["gold"]),
            "BBH development scoring changed",
        )
        key = (row["student_family"], row["condition"], row["task"])
        scores[key] += int(row["correct"])
        totals[key] += 1
        cases[row["student_family"], row["task"], row["condition"]].add(
            (row["item_id"], row["gold"])
        )
    for family in ("llama", "gemma", "qwen"):
        for task in ("boolean_expressions", "date_understanding", "formal_fallacies"):
            reference = cases[family, task, "raw"]
            for condition in ("few_shot", "compiled"):
                need(
                    cases[family, task, condition] == reference,
                    "BBH development cases are not paired",
                )
    need(len(metadata) == 27 and len(rows) == 540, "BBH study size changed")
    aggregate = {}
    for family in ("llama", "gemma", "qwen"):
        aggregate[family] = {
            condition: sum(
                (
                    scores[family, condition, task]
                    for task in (
                        "boolean_expressions",
                        "date_understanding",
                        "formal_fallacies",
                    )
                )
            )
            for condition in ("raw", "few_shot", "compiled")
        }
    need(
        aggregate
        == {
            "llama": {"raw": 23, "few_shot": 16, "compiled": 26},
            "gemma": {"raw": 26, "few_shot": 15, "compiled": 34},
            "qwen": {"raw": 43, "few_shot": 47, "compiled": 43},
        },
        "BBH development totals changed",
    )
    pooled = {
        condition: sum(
            (aggregate[family][condition] for family in ("llama", "gemma", "qwen"))
        )
        for condition in ("raw", "few_shot", "compiled")
    }
    intervals = {}
    for baseline in ("raw", "few_shot"):
        differences = [
            scores[family, "compiled", task] / 20 - scores[family, baseline, task] / 20
            for family in ("llama", "gemma", "qwen")
            for task in (
                "boolean_expressions",
                "date_understanding",
                "formal_fallacies",
            )
        ]
        interval = bootstrap(differences, 10000, 20260727)
        intervals[baseline] = {
            "estimate": interval[0],
            "bootstrap_95": list(interval[1:]),
        }
    close(intervals["raw"]["bootstrap_95"][0], -0.027777777777777776)
    close(intervals["raw"]["bootstrap_95"][1], 0.17777777777777773)
    close(intervals["few_shot"]["bootstrap_95"][0], -0.03888888888888891)
    close(intervals["few_shot"]["bootstrap_95"][1], 0.3277777777777778)
    return {
        "student_totals": aggregate,
        "pooled_correct_of_180": pooled,
        "compiled_minus_controls": intervals,
        "scope": "exploratory development gate; intervals include zero",
    }


def reproduce(root):
    """Return the six submitted panels with their original reporting keys."""
    result = {
        "main_llama": main_study(root),
        "main_gemma": gemma_study(root),
        "two_task_routing": routing_study(root),
        "llama_counting": textgrad_study(root),
        "historical_nlp_adapters": superni_study(root),
        "additional_bbh": bbh_development_study(root),
        "scope": "Submitted historical records; failed Gemma tasks use the recorded fallback estimates. NLP diagnostics retain the documented identical-prompt fallback correction.",
    }
    for name, denominator_key in [
        ("main_llama", "final_cases_per_condition"),
        ("main_gemma", "cases_per_condition_after_fallback"),
    ]:
        result[name]["conditions"] = {
            k: accuracy(v, result[name][denominator_key])
            for k, v in result[name]["correct"].items()
        }
    result["llama_counting"]["conditions"] = {
        name: accuracy(result["llama_counting"][key], 100)
        for name, key in [
            ("adapter", "method_correct"),
            ("textgrad", "textgrad_correct"),
            ("published_prompt", "prompt_transfer_correct"),
        ]
    }
    for name, student in result["historical_nlp_adapters"]["students"].items():
        student["conditions"] = {
            label: accuracy(student[key], student["cases"])
            for label, key in [
                ("positive_examples", "positive_examples_correct"),
                ("routed_recorded", "recorded_routed_correct"),
                (
                    "routed_identical_prompt_fallback_corrected",
                    "identical_prompt_corrected_routed",
                ),
            ]
        }
    raw = {
        (str(row["task"]), json.dumps(row["input"], sort_keys=True)): row
        for row in load_rows(
            resolve_path(root, root / SUBMITTED_RECORDS / "main_predictions.jsonl")
        )
        if row["condition"] == "raw"
    }
    direct = 0
    for task in read(
        resolve_path(root, root / INSTRUCTION_RECORDS / "prepared_inputs.json")
    ):
        if task["cohort"] != "llama":
            continue
        for case in task["cases"]:
            label = (
                case["final_program_label"]
                if task["approved"]
                else raw[
                    str(task["number"]), json.dumps(case["record"], sort_keys=True)
                ]["predicted"]
            )
            direct += label == case["gold"]
    result["main_llama"]["direct_execution_historical_description_fallback"] = accuracy(
        direct, 648
    )
    need(direct == 610, "Historical direct-execution fallback differs")
    return result
