"""Execute retained DSL programs, accepted repairs, and rendering checks."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
import sys

from .common import GEPA_RECORDS, INSTRUCTION_RECORDS, read, require, rows
from .layout import resolve_path


def teacher_divergence(root):
    sys.path.insert(0, str(root / "src"))
    from taskcompiler.dsl.parser import parse_program
    from taskcompiler.dsl.interpreter import run

    base = root / GEPA_RECORDS
    public = {
        t["task_id"]: t for t in read(resolve_path(root, base / "public_tasks.json"))
    }
    final = read(resolve_path(root, base / "sealed/final_cases.json"))
    output = []
    for number in range(8):
        tid = f"policy-{number:03d}"
        request = []
        response = []
        programs = []
        for model in ["qwen2.5_3b", "llama3.2_3b"]:
            events = [
                r
                for r in rows(
                    resolve_path(root, base / "calls" / f"{model}-{tid}-full.jsonl")
                )
                if r.get("role") == "teacher_optimization"
            ]
            request.append(
                next((r["body"] for r in events if r["event"] == "transport_request"))
            )
            response.append(
                next(
                    (
                        r["payload"]["message"]["content"]
                        for r in events
                        if r["event"] == "transport_response"
                    )
                )
            )
            prepared = read(
                resolve_path(root, base / "prepared" / f"{model}-{tid}-full.json")
            )
            source = prepared["method"]["trace"].get("dsl", {}).get("initial_source")
            programs.append(parse_program(source) if source else None)
        domain = [
            c["input_record"]
            for c in public[tid]["build_cases"]
            + public[tid]["audit_cases"]
            + final[tid]
        ]
        require(
            len(domain) == len({json.dumps(r, sort_keys=True) for r in domain}) == 81,
            "Divergence replay domain differs",
        )
        differences = (
            sum(
                (
                    run(programs[0], record).output_label
                    != run(programs[1], record).output_label
                    for record in domain
                )
            )
            if all((p is not None for p in programs))
            else None
        )
        output.append(
            {
                "task": tid,
                "identical_initial_request": request[0] == request[1],
                "identical_initial_response": response[0] == response[1],
                "initial_program_available": [p is not None for p in programs],
                "domain_inputs": 81,
                "program_output_disagreements": differences,
            }
        )
    require(
        all((r["identical_initial_request"] for r in output)),
        "Initial teacher request pairing differs",
    )
    require(
        [r["task"] for r in output if not r["identical_initial_response"]]
        == ["policy-001", "policy-004", "policy-007"],
        "Teacher response-difference inventory differs",
    )
    require(
        output[1]["program_output_disagreements"] == 12,
        "Policy-001 program disagreement differs",
    )
    return {
        "tasks": output,
        "scope": "Separate generated responses differed before student selection. This replay does not identify a hardware/kernel root cause.",
    }


def program_repairs(root):
    sys.path.insert(0, str(root / "src"))
    from taskcompiler.dsl.parser import parse_program
    from taskcompiler.dsl.interpreter import run

    tasks = read(
        resolve_path(root, root / INSTRUCTION_RECORDS / "prepared_inputs.json")
    )
    protocol = read(resolve_path(root, root / INSTRUCTION_RECORDS / "protocol.json"))
    retained = list(protocol["retained_input_sha256"])
    accepted = 0
    counts = Counter()
    program_scores = defaultdict(lambda: Counter())
    conversion = Counter()
    for task in tasks:
        if task["syntax_failure"]:
            continue
        group = program_scores[task["cohort"]]
        programs = {}
        for phase in ["initial", "final"]:
            suffix = f"{task['task_id']}.teacher.{phase}.dsl"
            candidates = [
                p
                for p in retained
                if p.endswith(suffix) and ("gemma" in p) == (task["cohort"] == "gemma")
            ]
            require(len(candidates) == 1, "Ambiguous retained program source")
            programs[phase] = parse_program(
                resolve_path(root, candidates[0]).read_text(encoding="utf-8")
            )
        for case in task["cases"]:
            initial = run(programs["initial"], case["record"]).output_label
            final = run(programs["final"], case["record"]).output_label
            require(
                initial == case["initial_program_label"]
                and final == case["final_program_label"],
                "Program execution differs from retained labels",
            )
            group["initial_correct"] += initial == case["gold"]
            group["final_correct"] += final == case["gold"]
            group["total"] += 1
        public = task["public_build"] + task["public_audit"]
        if task["approved"]:
            domain = [c["input_record"] for c in public] + [
                c["record"] for c in task["cases"]
            ]
            require(
                len(domain)
                == len({json.dumps(c, sort_keys=True) for c in domain})
                == 81,
                "Finite task domain is not 81 disjoint records",
            )
            decision_lines = task["representations"]["final_list"][1:]
            for record in domain:
                flat = {
                    k: str(v) for fields in record.values() for k, v in fields.items()
                }
                labels = set()
                for line in decision_lines:
                    match = re.fullmatch("If (.*), return ([A-Za-z_]+)\\.", line)
                    require(bool(match), "Unrecognized retained decision entry")
                    conditions, label = match.groups()
                    if conditions == "any input" or all(
                        (
                            flat.get(k.split(".")[-1]) == v
                            for k, v in (
                                part.split("=", 1) for part in conditions.split(", ")
                            )
                        )
                    ):
                        labels.add(label)
                expected = run(programs["final"], record).output_label or "uncertain"
                require(
                    labels == {expected},
                    "Rendered decision list changes program output",
                )
            conversion[task["cohort"]] += 1
        for repair in task["repair_records"]:
            if not repair["accepted"]:
                continue
            before = parse_program(repair["original_source"])
            after = parse_program(repair["proposed_source"])
            old = {
                i
                for i, c in enumerate(public)
                if run(before, c["input_record"]).output_label == c["expected_label"]
            }
            new = {
                i
                for i, c in enumerate(public)
                if run(after, c["input_record"]).output_label == c["expected_label"]
            }
            require(old < new, "Accepted repair violates strict public non-regression")
            accepted += 1
            name = repair["rule_name"]
            kind = (
                "teacher"
                if name == "<complete-program>"
                else "condition_narrowing"
                if name.startswith("local-exclusion:")
                else "insertion"
            )
            counts[task["cohort"], kind] += 1
    require(accepted == 72, "Accepted repair count differs")
    return {
        "accepted_changes_replayed": accepted,
        "repair_counts": {c + "/" + k: v for (c, k), v in counts.items()},
        "program_only_scores_reexecuted": {
            k: dict(v) for k, v in program_scores.items()
        },
        "approved_decision_lists_checked_over_81_inputs": dict(conversion),
        "scope": "All accepted changes executed on 54 public cases. Program-only totals reexecuted from saved DSL; include unapproved parsed candidates and exclude two missing Gemma programs. Conversion agrees with generated programs, not necessarily with reference labels everywhere.",
    }
