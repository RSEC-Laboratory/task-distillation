"""Adapter measurements, reserved-task failures, and complete attempt accounting."""

from __future__ import annotations

from collections import Counter, defaultdict

from .bootstrap import bootstrap
from .common import (
    RESERVED_PROTOCOL,
    accuracy,
    attempt_accounting,
    call_ledger,
    digest,
    read,
    require,
    rows,
)
from .layout import resolve_path, run_directory

QWEN_RUNS = {
    "counting_adapter": "2026-10-07_000448_bbh_bbh_counting_qwen25_3b_adapter_20261006",
    "counting_textgrad": "2026-10-06_230851_bbh_bbh_counting_qwen25_3b_textgrad_tgd_20261006",
    "counting_transfer": "2026-10-07_001100_bbh_bbh_counting_qwen25_3b_textgrad_prompt_20261006",
    "superni": "2026-10-07_001402_superni_superni_qwen25_3b_fixed_llama70_development_20261006",
}


def nlp_rows(path, score, expected_tasks, per_task=20):
    predicted = rows(path)
    seen = set()
    counts = defaultdict(lambda: [0, 0])
    refs = {}
    for row in predicted:
        key = (row["task_id"], row["item_id"], row["condition"])
        require(key not in seen, "Duplicate NLP prediction")
        seen.add(key)
        refkey = key[:2]
        references = tuple(row["references"])
        require(
            refkey not in refs or refs[refkey] == references,
            "NLP reference changes across conditions",
        )
        refs[refkey] = references
        correct = score["response_matches"](row["response"], references)
        require(correct == row["correct"], "NLP response rescoring differs")
        counts[row["condition"]][0] += correct
        counts[row["condition"]][1] += 1
    require(
        len({t for t, _ in refs}) == expected_tasks
        and len(refs) == expected_tasks * per_task,
        "NLP task/item inventory differs",
    )
    require(
        len(counts) == 5
        and all(v[1] == expected_tasks * per_task for v in counts.values()),
        "Incomplete NLP condition grid",
    )
    return predicted, refs, {k: accuracy(*v) for k, v in counts.items()}


def qwen_adapters(root, score):
    counting = {}
    ids = None
    for method, name in QWEN_RUNS.items():
        if method == "superni":
            continue
        data = rows(
            resolve_path(
                root, run_directory(root, name) / "artifacts/predictions.jsonl"
            )
        )
        require(len(data) == 100, "Incomplete Qwen counting panel")
        mapping = {row["item_id"]: row for row in data}
        require(
            len(mapping) == 100 and (ids is None or set(mapping) == ids),
            "Unpaired Qwen counting cases",
        )
        ids = set(mapping)
        correct = initial = corrections = 0
        for row in data:
            value = score["parse_answer"]("object_counting", row["raw_output"])
            require(
                value == row["parsed_output"]
                and row["correct"] == (value == row["gold"]),
                "Qwen counting rescoring differs",
            )
            correct += value == row["gold"]
            initial += (
                score["parse_answer"](
                    "object_counting", row["initial_output"] or row["raw_output"]
                )
                == row["gold"]
            )
            corrections += bool(row["local_correction_used"])
        counting[method] = {
            **accuracy(correct, 100),
            "initial_correct": initial,
            "local_corrections": corrections,
            "all_preparation_and_final_attempts": attempt_accounting(
                resolve_path(
                    root, run_directory(root, name) / "artifacts/model_calls.jsonl"
                )
            ),
        }
    _, _, nlp = nlp_rows(
        resolve_path(
            root,
            run_directory(root, QWEN_RUNS["superni"]) / "artifacts/predictions.jsonl",
        ),
        score,
        4,
    )
    interrupted = (
        run_directory(
            root, "2026-10-06_230554_bbh_bbh_counting_qwen25_3b_textgrad_tgd_20261006"
        )
        / "artifacts"
    )
    require(
        (interrupted / "termination.json").exists(),
        "Interrupted predecessor lacks its termination record",
    )
    return {
        "counting": counting,
        "nlp_conditions": nlp,
        "nlp_attempts": attempt_accounting(
            resolve_path(
                root,
                run_directory(root, QWEN_RUNS["superni"])
                / "artifacts/model_calls.jsonl",
            )
        ),
        "interrupted_predecessor_attempts": attempt_accounting(
            resolve_path(root, interrupted / "model_calls.jsonl"), True
        ),
        "scope": "Submitted exploratory Qwen adapter panels; counting has unequal runtime assistance. Positive-example fallback is retained in the four-task NLP panel. Completed algorithm runs and interrupted project overhead are shown separately.",
    }


def reserved(root, score):
    name = "2026-11-20_115943_superni_reviewer1_reserved20_llama_20261120"
    run = run_directory(root, name)
    predicted, refs, conditions = nlp_rows(
        resolve_path(root, run / "artifacts/predictions.jsonl"), score, 20
    )
    artifacts = [
        read(resolve_path(root, path))
        for path in (run / "artifacts").glob("task*.json")
    ]
    require(
        len(artifacts) == 20
        and all((a["final_representation"] is None for a in artifacts)),
        "Reserved task fallback inventory differs",
    )
    starts, ends = call_ledger(resolve_path(root, run / "artifacts/model_calls.jsonl"))
    roles = Counter((r["role"] for r in starts.values()))
    require(
        roles["teacher"] == 20 and roles["student"] == 800,
        "Reserved call accounting differs",
    )
    protocol = read(resolve_path(root, root / RESERVED_PROTOCOL))
    means = {
        t["task_id"]: {
            condition: sum(
                (
                    r["correct"]
                    for r in predicted
                    if r["task_id"] == t["task_id"] and r["condition"] == condition
                )
            )
            / 20
            for condition in conditions
        }
        for t in protocol["tasks"]
    }
    contrasts = {}
    for comparator in [
        "raw_definition",
        "positive_examples",
        "parent_one_shot",
        "validated_full",
    ]:
        differences = [v["validated_routed"] - v[comparator] for v in means.values()]
        contrasts["validated_routed minus " + comparator] = bootstrap(
            differences, 100000, 20260727
        )
    source_checked = 0
    for task in protocol["tasks"]:
        source = root / "data/superni/tasks" / (task["task_id"] + ".json")
        if not source.exists():
            continue
        require(
            digest(source) == task["source_sha256"],
            "Downloaded upstream NLP task hash differs",
        )
        official = read(resolve_path(root, source))
        for item in official["Instances"][:20]:
            require(
                tuple(item["output"]) == refs[task["task_id"], item["id"]],
                "NLP references differ from upstream instances",
            )
        source_checked += 1
    schema = []
    schema_requests = []
    for name in [
        "2026-11-20_123413_superni_reviewer1_reserved20_schema70b_llama_20261120",
        "2026-11-20_123628_superni_reviewer1_reserved20_schema70b_transport1_llama_20261120",
    ]:
        path = run_directory(root, name)
        first, last = call_ledger(
            resolve_path(root, path / "artifacts/model_calls.jsonl")
        )
        require(
            len(first) == len(last) == 1
            and all(
                (r["event"] == "failed" and "429" in r["error"] for r in last.values())
            ),
            "Schema endpoint failure differs",
        )
        require(
            not (path / "artifacts/predictions.jsonl").exists(),
            "Unavailable endpoint has fabricated final observations",
        )
        schema_requests.append(next(iter(first.values()))["request"])
        schema.append(
            {
                "failed_teacher_attempts": 1,
                "successful_responses": 0,
                "final_observations": 0,
            }
        )
    require(
        schema_requests[0] == schema_requests[1],
        "Transport restart changed schema-teacher request content",
    )
    return {
        "conditions": conditions,
        "fallback_tasks": 20,
        "physical_attempts_by_role": dict(roles),
        "contrasts": contrasts,
        "source_instance_validation": {
            "tasks_checked": source_checked,
            "expected_tasks": 20,
            "status": "passed"
            if source_checked == 20
            else "not_complete; saved-row references rescored, upstream files not redistributed",
        },
        "schema_enabled_configuration": {
            "status": "unavailable",
            "attempts": schema,
            "accuracy": None,
        },
        "scope": "All twenty compilation-format failures and the two HTTP-429 attempts retained. No positive independent-transfer claim.",
    }
