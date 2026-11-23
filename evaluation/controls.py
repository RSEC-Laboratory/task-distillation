"""Instruction-format, delivery, checker, and lookup control experiments."""

from __future__ import annotations

from collections import Counter, defaultdict

from .bootstrap import bootstrap
from .common import (
    COUNTING_RECORDS,
    INSTRUCTION_RECORDS,
    LOOKUP_RECORDS,
    accuracy,
    call_ledger,
    read,
    require,
    rows,
)
from .layout import resolve_path


def original_ablation(root, score):
    base = root / INSTRUCTION_RECORDS
    prepared = read(resolve_path(root, base / "prepared_inputs.json"))
    tasks = {(t["cohort"], t["task_id"]): t for t in prepared}
    protocol = read(resolve_path(root, base / "protocol.json"))
    predictions = rows(resolve_path(root, base / "predictions.jsonl"))
    starts, ends = call_ledger(resolve_path(root, base / "calls.jsonl"))
    primary = [r for r in predictions if not r["sentinel"]]
    require(
        len(primary) == 9072 and len(predictions) == 9408, "Incomplete ablation grid"
    )
    seen = set()
    cells = defaultdict(list)
    for row in predictions:
        key = (
            row["cohort"],
            row["task_id"],
            row["case_index"],
            row["arm"],
            row["sentinel"],
        )
        require(key not in seen, "Duplicate ablation row")
        seen.add(key)
        task = tasks[key[:2]]
        case = task["cases"][row["case_index"]]
        require(row["gold"] == case["gold"], "Ablation reference mismatch")
        start, end = (starts[row["call_id"]], ends[row["call_id"]])
        require(
            row["request_hash"] == start["request_hash"] == end["request_hash"],
            "Ablation call linkage differs",
        )
        raw = end.get("response", {}).get("text", "")
        label = (
            "__INVALID__:transport_failure"
            if end["event"] == "failed"
            else score["normalise_student_label"](raw, set(task["labels"]))
        )
        require(
            row["raw_output"] == raw and row["predicted"] == label,
            "Ablation response normalization differs",
        )
        require(row["correct"] == (label == case["gold"]), "Ablation score differs")
        if row["arm"] != "raw_original":
            require(
                start["system"].endswith(task["raw_labels"].rsplit("\n\n", 1)[1]),
                "Allowed-label contract differs",
            )
        if not row["sentinel"]:
            cells[row["cohort"], row["task_id"], row["arm"]].append(row)
    output = {}
    for cohort, expected_tasks in [("llama", 24), ("gemma", 18)]:
        cohort_tasks = [t for t in prepared if t["cohort"] == cohort]
        require(len(cohort_tasks) == expected_tasks, "Cohort task count differs")
        conditions = {}
        for arm in protocol["arms"]:
            group = []
            for task in cohort_tasks:
                cell = cells[cohort, task["task_id"], arm]
                require(len(cell) == 27, "Incomplete task/arm cell")
                group.extend(cell)
            conditions[arm] = {
                **accuracy(sum((r["correct"] for r in group)), len(group)),
                "invalid_labels": sum((r["invalid"] for r in group)),
            }
        contrasts = {}
        for first, second in protocol["contrasts"]:
            differences = [
                (
                    sum((r["correct"] for r in cells[cohort, t["task_id"], first]))
                    - sum((r["correct"] for r in cells[cohort, t["task_id"], second]))
                )
                / 27
                for t in cohort_tasks
            ]
            contrasts[first + " minus " + second] = bootstrap(differences)
        direct = 0
        for task in cohort_tasks:
            for case in task["cases"]:
                label = (
                    case["final_program_label"]
                    if task["approved"]
                    else next(
                        (
                            r["predicted"]
                            for r in cells[cohort, task["task_id"], "raw_labels"]
                            if r["case_index"] == case["index"]
                        )
                    )
                )
                direct += label == case["gold"]
        output[cohort] = {
            "tasks": expected_tasks,
            "conditions": conditions,
            "contrasts": contrasts,
            "direct_execution_with_same_fresh_description_fallback": accuracy(
                direct, 27 * expected_tasks
            ),
            "approved_tasks": sum((t["approved"] for t in cohort_tasks)),
        }
    return {
        "cohorts": output,
        "primary_rows": len(primary),
        "diagnostic_sentinels": len(predictions) - len(primary),
        "physical_attempts": len(starts),
        "transport_failures": sum((r["event"] == "failed" for r in ends.values())),
        "scope": "Fresh inference on retained historical families. Intervals resample whole tasks; this is not independent-family confirmation.",
    }


def counting_control(root, score):
    base = root / COUNTING_RECORDS
    prepared = read(resolve_path(root, base / "prepared_inputs.json"))
    references = {c["item_id"]: c["gold"] for c in prepared["cases"]}
    cases = {c["item_id"]: c for c in prepared["cases"]}
    predicted = rows(resolve_path(root, base / "predictions.jsonl"))
    require(len(predicted) == 200, "Incomplete common-checker grid")
    starts, ends = call_ledger(resolve_path(root, base / "calls.jsonl"))
    by_phase = {
        (r["method"], r["item_id"], r["phase"]): ends[cid] for cid, r in starts.items()
    }
    requests = {(r["method"], r["item_id"], r["phase"]): r for r in starts.values()}
    methods = defaultdict(list)
    for row in predicted:
        key = (row["method"], row["item_id"])
        initial_text = by_phase[*key, "initial"].get("response", {}).get("text", "")
        require(
            requests[*key, "initial"]["system"] == prepared["systems"][row["method"]]
            and requests[*key, "initial"]["prompt"] == cases[row["item_id"]]["prompt"],
            "Counting prompts differ from the frozen input/contract",
        )
        parsed = score["parse_answer"]("object_counting", initial_text)
        ledger = score["_verified_ledger_sum"](initial_text)
        require(
            parsed == row["initial"] and ledger == row["ledger"],
            "Counting initial parse/ledger differs",
        )
        correction = by_phase.get((*key, "correction"))
        require(
            bool(correction) == row["correction_used"],
            "Counting correction flag differs",
        )
        require(
            bool(correction) == (ledger is not None and ledger != parsed),
            "Counting common trigger rule differs",
        )
        if correction:
            request = requests[*key, "correction"]
            expected_prompt = f"A local verifier checked your structured trace and obtained {ledger}. Return exactly `Answer: {ledger}`."
            require(
                request["prompt"] == expected_prompt
                and request["system"]
                == "Return the locally verified answer exactly as requested. Do not add reasoning.",
                "Counting correction instruction differs",
            )
        checked = (
            score["parse_answer"](
                "object_counting", correction.get("response", {}).get("text", "")
            )
            if correction
            else parsed
        )
        require(
            checked == row["checked"] and row["gold"] == references[row["item_id"]],
            "Counting final/reference differs",
        )
        require(
            row["initial_correct"] == (parsed == row["gold"])
            and row["checked_correct"] == (checked == row["gold"]),
            "Counting rescoring differs",
        )
        methods[row["method"]].append(row)
    result = {
        name: {
            "off": accuracy(sum((r["initial_correct"] for r in group)), 100),
            "on": accuracy(sum((r["checked_correct"] for r in group)), 100),
            "corrections": sum((r["correction_used"] for r in group)),
            "parseable_ledgers": sum((r["ledger"] is not None for r in group)),
        }
        for name, group in methods.items()
    }
    contrasts = {}
    indexed = {(r["method"], r["item_id"]): r for r in predicted}
    for phase in ("initial", "checked"):
        differences = [
            int(indexed["counting_guidance", key][phase + "_correct"])
            - int(indexed["textgrad_guidance", key][phase + "_correct"])
            for key in references
        ]
        contrasts[phase] = bootstrap(differences)
    return {
        "methods": result,
        "contrasts": contrasts,
        "physical_attempts": len(starts),
        "transport_failures": sum((r["event"] == "failed" for r in ends.values())),
        "scope": "Frozen prompts on 100 historical Llama cases; same checker/at-most-one-correction policy, different realized call counts. Combined adapter result, not compilation-only gain.",
    }


def lookup_control(root):
    tasks = {
        (t["cohort"], t["task_id"]): t
        for t in read(
            resolve_path(root, root / INSTRUCTION_RECORDS / "prepared_inputs.json")
        )
    }
    counts = defaultdict(lambda: [0, 0])
    seen = set()
    for row in rows(resolve_path(root, root / LOOKUP_RECORDS / "predictions.jsonl")):
        key = (row["cohort"], row["task_id"], row["case_index"], row["method"])
        require(key not in seen, "Duplicate lookup observation")
        seen.add(key)
        task = tasks[key[:2]]
        public = task["public_build"] + task["public_audit"]
        if row["method"] == "hamming_nearest_vote":

            def flatten(record, prefix=""):
                result = {}
                for k, v in record.items():
                    if isinstance(v, dict):
                        result.update(flatten(v, prefix + k + "."))
                    else:
                        result[prefix + k] = str(v)
                return result

            source = flatten(task["cases"][row["case_index"]]["record"])
            distances = [
                sum((source[k] != flatten(c["input_record"])[k] for k in source))
                for c in public
            ]
            neighbors = [i for i, d in enumerate(distances) if d == min(distances)]
            require(
                neighbors == row["public_neighbor_indices"],
                "Hamming neighbor selection differs",
            )
            votes = Counter((public[i]["expected_label"] for i in neighbors))
            label = min(
                votes, key=lambda x: (-votes[x], "uncertain" if x is None else x)
            )
        else:
            require(
                row["method"] == "minilm_1nn"
                and len(row["public_neighbor_indices"]) == 1,
                "Invalid lookup method",
            )
            label = public[row["public_neighbor_indices"][0]]["expected_label"]
        gold = task["cases"][row["case_index"]]["gold"]
        require(
            row["predicted"] == label
            and row["gold"] == gold
            and (row["correct"] == (label == gold)),
            "Lookup rescoring differs",
        )
        counts[row["cohort"], row["method"]][0] += label == gold
        counts[row["cohort"], row["method"]][1] += 1
    require(len(seen) == 2268, "Incomplete lookup grid")
    return {
        "conditions": {c + "/" + m: accuracy(*v) for (c, m), v in counts.items()},
        "scope": "Both use 54 public labels. Supplementary protocol specified after observing the completed Llama cohort. Hamming selection recomputed; embedding labels checked against recorded neighbor IDs, without regenerating encoder embeddings.",
    }
