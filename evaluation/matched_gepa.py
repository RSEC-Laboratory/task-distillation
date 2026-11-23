"""Current native GEPA comparison with matched public examples and call ceilings."""

from __future__ import annotations

from collections import Counter
import random
import statistics

from .common import GEPA_RECORDS, accuracy, read, require, rows
from .layout import resolve_path


def current_gepa(root):
    base = root / GEPA_RECORDS
    protocol = read(resolve_path(root, base / "protocol.json"))
    output = {}
    for model in protocol["solver_models"]:
        prefix = model.replace(":", "_")
        totals = Counter()
        per_task = []
        costs = {arm: Counter() for arm in ("full", "gepa")}
        for number in protocol["task_ids"]:
            tid = f"policy-{number:03d}"
            panels = {
                arm: read(
                    resolve_path(root, base / "final" / f"{prefix}-{tid}-{arm}.json")
                )
                for arm in ("raw", "full", "gepa", "direct_dsl")
            }
            indexed = {
                arm: {row["case_id"]: row for row in panel["rows"]}
                for arm, panel in panels.items()
            }
            require(
                all((len(v) == 27 for v in indexed.values())),
                "Incomplete native GEPA final cell",
            )
            ids = set(indexed["raw"])
            require(
                all((set(v) == ids for v in indexed.values())),
                "Unpaired native GEPA cases",
            )
            counts = Counter()
            for cid in ids:
                gold = indexed["raw"][cid]["expected_label"]
                for arm in panels:
                    row = indexed[arm][cid]
                    require(
                        row["expected_label"] == gold
                        and row["correct"] == (row["predicted_label"] == gold),
                        "Native GEPA rescoring differs",
                    )
                    counts[arm] += row["predicted_label"] == gold
                direct_label = (
                    indexed["direct_dsl"][cid]["predicted_label"]
                    if panels["direct_dsl"]["program_available"]
                    else indexed["raw"][cid]["predicted_label"]
                )
                counts["direct_matched_fallback"] += direct_label == gold
            totals.update(counts)
            per_task.append(
                {
                    "task": tid,
                    "correct": dict(counts),
                    "program_available": panels["direct_dsl"]["program_available"],
                }
            )
            for arm in ("full", "gepa"):
                events = rows(
                    resolve_path(root, base / "calls" / f"{prefix}-{tid}-{arm}.jsonl")
                )
                requests = [r for r in events if r["event"] == "model_request"]
                roles = Counter((r["role"] for r in requests))
                require(
                    roles["solver_optimization"] <= 216
                    and roles["teacher_optimization"] <= 4,
                    "Native GEPA call ceiling exceeded",
                )
                costs[arm].update(roles)
                costs[arm]["failed_attempts"] += sum(
                    (r["event"] == "model_failure" for r in events)
                )
        differences = [
            (t["correct"]["full"] - t["correct"]["gepa"]) / 27 for t in per_task
        ]
        generator = random.Random(protocol["seed"])
        draws = sorted(
            (
                statistics.mean(generator.choices(differences, k=len(differences)))
                for _ in range(10000)
            )
        )
        contrast = {
            "difference": statistics.mean(differences),
            "descriptive_95_interval": [draws[250], draws[9750]],
            "resamples": 10000,
            "seed": protocol["seed"],
            "unit_count": len(differences),
            "implementation": "Original native-report Python random.choices and indexed quantiles",
        }
        recorded = next(
            (
                r
                for r in read(resolve_path(root, base / "summary.json"))["models"]
                if r["solver_model"] == model
            )
        )
        require(
            all(
                (
                    abs(a - b) < 1e-12
                    for a, b in zip(
                        contrast["descriptive_95_interval"],
                        recorded["descriptive_task_bootstrap_95_interval"],
                    )
                )
            ),
            "Native GEPA interval differs from retained report",
        )
        output[model] = {
            "conditions": {arm: accuracy(count, 216) for arm, count in totals.items()},
            "per_task": per_task,
            "full_minus_gepa": contrast,
            "preparation_attempts_by_role": {k: dict(v) for k, v in costs.items()},
        }
    return {
        "students": output,
        "scope": "Current native GEPA comparison: same 54 public labeled cases and call ceilings. All eight tasks retained. Execution-only and symmetric-fallback policies are reported separately.",
        "boundary": "Saved output rescoring and call-budget checks. Does not rerun MiniLM routing or reproduce every original chronological/source-package audit.",
    }
