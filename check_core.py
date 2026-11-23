"""Replay retained rule programs and public repair gates without model packages.

Run with the Python standard library, including with site packages disabled:
    python -S check_core.py --root task-compiler

This executes the supplied parser, interpreter, and sandbox. It does not
regenerate teacher/student responses or treat public-case checks as a proof
of correctness on arbitrary inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import sys


class CheckError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "An evidence path escapes the artifact root")
    require(path.is_file(), f"Missing input: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(record, prefix=""):
    fields = {}
    for key, value in record.items():
        name = prefix + key
        if isinstance(value, dict):
            fields.update(flatten(value, name + "."))
        else:
            fields[name] = str(value)
    return fields


def record_from_fields(paths, values):
    record = {}
    for path, value in zip(paths, values):
        root, field = path.split(".", 1)
        record.setdefault(root, {})[field] = value
    return record


def label(value):
    return value if value is not None else "uncertain"


def import_core(root):
    source = root / "src"
    if source.is_dir():
        sys.path.insert(0, str(source))
    names = ("taskcompiler.dsl.parser", "taskcompiler.dsl.interpreter",
             "taskcompiler.sandbox", "taskcompiler.schema")
    modules = [importlib.import_module(name) for name in names]
    if source.is_dir():
        require(all(Path(module.__file__).resolve().is_relative_to(source.resolve())
                    for module in modules), "The imported core is outside the supplied source tree")
    return modules


def make_schema(task, public, schema):
    domains = {}
    for case in public:
        for path, value in flatten(case["input_record"]).items():
            domains.setdefault(path, set()).add(value)
    records = {}
    enums = []
    for index, (path, values) in enumerate(sorted(domains.items())):
        name, field = path.split(".", 1)
        enum_name = f"ObservedValue{index}"
        enums.append(schema.EnumType(enum_name, sorted(values)))
        records.setdefault(name, []).append(schema.FieldDecl(field, enum_name))
    spec = schema.TaskSpec(task_name=task["task_id"], description="Retained policy replay",
        labels=task["labels"], enums=enums,
        records=[schema.RecordType(name, fields) for name, fields in records.items()])
    return spec, domains


def decision_entries(lines, paths):
    entries = []
    for line in lines:
        if not line.startswith("If "):
            continue
        conditions, separator, output = line[3:].partition(", return ")
        require(bool(separator) and output.endswith("."), "Unrecognized rendered decision entry")
        output = output[:-1]
        terms = []
        if conditions != "any input":
            for term in conditions.split(", "):
                field, separator, value = term.partition("=")
                require(bool(separator), "Malformed rendered field condition")
                candidates = [p for p in paths if p == field or p.endswith("." + field)]
                require(len(candidates) == 1, "A rendered field name is missing or ambiguous")
                terms.append((candidates[0], value))
        entries.append((terms, output))
    require(bool(entries), "No machine-readable decision entries in an approved artifact")
    return entries


def verify(root):
    # Audit hooks reject any attempted network access, including loopback.
    def no_network(event, _args):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
            raise CheckError("Network access is not permitted during deterministic replay")
    sys.addaudithook(no_network)
    parser, interpreter, sandbox, schema = import_core(root)
    prepared = read(root, "reports/reviewer1_20261120/original_ablation/prepared_inputs.json")
    require(len(prepared) == 42, "Expected all 42 retained task/model pairs")
    metadata = {}
    for cohort, filename in (("llama", "main_runs.json"), ("gemma", "gemma_runs.json")):
        rows = read(root, "reviewer_artifact/data/" + filename)
        for row in rows:
            if row.get("condition") == "full":
                key = (cohort, row["task"])
                require(key not in metadata, "Duplicate retained full-run metadata")
                metadata[key] = row
    cohorts = {}
    missing = []
    accepted = 0
    rendering_artifacts = 0
    rendered_domain_cases = 0
    executions = 0

    def execute(program, record):
        nonlocal executions
        result = sandbox.run_sandboxed(program, record)
        require(not result.error and not result.timed_out and result.trace is not None,
                "A retained program failed sandbox execution")
        executions += 1
        return result.trace.output_label

    for task in prepared:
        if task["syntax_failure"]:
            missing.append({"cohort": task["cohort"], "task": task["task_id"],
                            "reason": "retained initial syntax failure; no parsed program pair"})
            continue
        cohort = task["cohort"]
        row = metadata[(cohort, task["number"])]
        run_id = row["run_id"]
        require(Path(run_id).name == run_id, "A run identifier is not a single directory name")
        public = task["public_build"] + task["public_audit"]
        require(len(public) == 54 and len(task["cases"]) == 27, "Unexpected case partition size")
        spec, domains = make_schema(task, public, schema)
        programs = {}
        for stage in ("initial", "final"):
            relative = f"runs/{run_id}/artifacts/policy-{task['number']:03d}.teacher.{stage}.dsl"
            path = (root / relative).resolve()
            require(path.is_relative_to(root) and path.is_file(), f"Missing retained program: {relative}")
            programs[stage] = parser.parse_program(path.read_text(encoding="utf-8"), task_spec=spec)
            canonical = parser.render_program_source(programs[stage]).encode("utf-8")
            require(hashlib.sha256(canonical).hexdigest() == row["policy"][stage + "_program_sha256"],
                    "A retained program differs from its recorded canonical-source hash")
        score = cohorts.setdefault(cohort, {"parsed_tasks": 0, "initial_correct": 0,
                                           "final_correct": 0, "total_final_cases": 0})
        score["parsed_tasks"] += 1
        for case in task["cases"]:
            for stage in ("initial", "final"):
                output = label(execute(programs[stage], case["record"]))
                require(output == label(case[stage + "_program_label"]),
                        "Replayed program output differs from the retained final-case output")
                score[stage + "_correct"] += output == label(case["gold"])
            score["total_final_cases"] += 1
        if task["approved"]:
            require(all(execute(programs["final"], case["input_record"]) == case["expected_label"]
                        for case in public), "An approved program fails a required public case")
        for repair in task["repair_records"]:
            if not repair["accepted"]:
                continue
            before = parser.parse_program(repair["original_source"], task_spec=spec)
            after = parser.parse_program(repair["proposed_source"], task_spec=spec)
            old = {i for i, case in enumerate(public)
                   if execute(before, case["input_record"]) == case["expected_label"]}
            new = {i for i, case in enumerate(public)
                   if execute(after, case["input_record"]) == case["expected_label"]}
            require(old < new, "An accepted repair does not satisfy strict public non-regression")
            accepted += 1
        if task["approved"]:
            paths = sorted(domains)
            values = [sorted(domains[path]) for path in paths]
            require(len(paths) == 4 and all(len(domain) == 3 for domain in values),
                    "Expected the retained four-field, 81-combination domain")
            entries = decision_entries(task["representations"]["selected"], paths)
            for combination in itertools.product(*values):
                record = record_from_fields(paths, combination)
                fields = flatten(record)
                matches = {output for terms, output in entries
                           if all(fields[field] == value for field, value in terms)}
                output = label(execute(programs["final"], record))
                require(matches == {output}, "Rendered decisions lack coverage, purity, or program equivalence")
                rendered_domain_cases += 1
            rendering_artifacts += 1
    require(cohorts == {
        "llama": {"parsed_tasks": 24, "initial_correct": 604, "final_correct": 627, "total_final_cases": 648},
        "gemma": {"parsed_tasks": 16, "initial_correct": 385, "final_correct": 424, "total_final_cases": 432}},
        "Program-only counts differ from the retained study")
    require(len(missing) == 2 and accepted == 72, "Unexpected missing-program or accepted-repair count")
    require(rendering_artifacts == 38 and rendered_domain_cases == 3078,
            "Expected every approved artifact's complete finite-domain conversion check")
    modules = [parser, interpreter, sandbox, schema]
    return {"status": "passed", "mode": "deterministic_retained_program_replay",
        "model_generation_calls": 0, "network_access": "blocked by Python audit hook",
        "program_only_final_scores": cohorts, "missing_program_tasks": missing,
        "accepted_public_repairs_replayed": accepted,
        "approved_rendered_artifacts_checked": rendering_artifacts,
        "finite_domain_records_checked": rendered_domain_cases,
        "sandbox_executions": executions,
        "core_module_sha256": {module.__name__: sha(Path(module.__file__)) for module in modules},
        "scope": "Program-only counts include unapproved parsed candidates and exclude two missing Gemma programs. Public repair non-regression and rendering equivalence do not imply unseen-input reference correctness. Student outputs are not regenerated."}


def main(argv=None):
    arguments = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parent
    if (default / "task-compiler").is_dir():
        default = default / "task-compiler"
    arguments.add_argument("--root", type=Path, default=default,
                           help="Project root containing src, reports, runs, and reviewer_artifact")
    arguments.add_argument("--output", type=Path, help="Optional JSON report destination")
    options = arguments.parse_args(argv)
    try:
        result = verify(options.root.resolve())
    except (CheckError, ImportError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2), file=sys.stderr)
        return 1
    payload = json.dumps(result, indent=2) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
