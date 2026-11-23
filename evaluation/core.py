"""Replay retained rule programs with the supplied parser and sandbox.

Run from the repository root with ``python -S -m evaluation.core``. No model
packages or network access are needed. The checks cover retained cases,
accepted public repairs, and complete finite-domain rendering equivalence;
they do not regenerate model responses or prove unseen-input correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import sys

from evaluation.layout import PROJECT_ROOT, prepare_records, run_directory


class CheckError(RuntimeError):
    """A retained input or replayed result violates the recorded contract."""


def require(condition, message):
    if not condition:
        raise CheckError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "An input path escapes the project root")
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
        parent, field = path.split(".", 1)
        record.setdefault(parent, {})[field] = value
    return record


def label(value):
    return value if value is not None else "uncertain"


def import_core(root):
    """Load actual implementation modules, rather than replay substitutes."""
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


def make_schema(task, public_cases, schema):
    """Recover typed finite field domains from the retained public cases."""
    domains = {}
    for case in public_cases:
        for path, value in flatten(case["input_record"]).items():
            domains.setdefault(path, set()).add(value)
    records, enums = {}, []
    for index, (path, values) in enumerate(sorted(domains.items())):
        name, field = path.split(".", 1)
        enum_name = f"ObservedValue{index}"
        enums.append(schema.EnumType(enum_name, sorted(values)))
        records.setdefault(name, []).append(schema.FieldDecl(field, enum_name))
    spec = schema.TaskSpec(
        task_name=task["task_id"], description="Retained policy replay",
        labels=task["labels"], enums=enums,
        records=[schema.RecordType(name, fields) for name, fields in records.items()])
    return spec, domains


def decision_entries(lines, paths):
    """Parse the generated decision list independently of the DSL evaluator."""
    entries = []
    for line in lines:
        if not line.startswith("If "):
            continue
        conditions, separator, output = line[3:].partition(", return ")
        require(bool(separator) and output.endswith("."), "Unrecognized rendered decision entry")
        terms = []
        if conditions != "any input":
            for term in conditions.split(", "):
                field, separator, value = term.partition("=")
                require(bool(separator), "Malformed rendered field condition")
                candidates = [path for path in paths if path == field or path.endswith("." + field)]
                require(len(candidates) == 1, "A rendered field name is missing or ambiguous")
                terms.append((candidates[0], value))
        entries.append((terms, output[:-1]))
    require(bool(entries), "No machine-readable decision entries in an approved artifact")
    return entries


def load_inputs(root):
    prepared = read_json(root, "evaluation/records/instruction_ablation/prepared_inputs.json")
    require(len(prepared) == 42, "Expected all 42 retained task/model pairs")
    metadata = {}
    for cohort, filename in (("llama", "main_runs.json"), ("gemma", "gemma_runs.json")):
        for row in read_json(root, "evaluation/records/submitted/" + filename):
            if row.get("condition") == "full":
                key = (cohort, row["task"])
                require(key not in metadata, "Duplicate retained full-run metadata")
                metadata[key] = row
    return prepared, metadata


def load_programs(root, task, metadata, spec, parser):
    folder = run_directory(root, metadata["run_id"])
    programs = {}
    for stage in ("initial", "final"):
        path = folder / "artifacts" / f"policy-{task['number']:03d}.teacher.{stage}.dsl"
        require(path.resolve().is_relative_to(root) and path.is_file(),
                f"Missing retained {stage} program for {task['task_id']}")
        program = parser.parse_program(path.read_text(encoding="utf-8"), task_spec=spec)
        canonical = parser.render_program_source(program).encode("utf-8")
        require(hashlib.sha256(canonical).hexdigest() == metadata["policy"][stage + "_program_sha256"],
                "A program differs from its recorded canonical-source hash")
        programs[stage] = program
    return programs


def check_repairs(task, public_cases, spec, parser, execute):
    accepted = 0
    for repair in task["repair_records"]:
        if not repair["accepted"]:
            continue
        before = parser.parse_program(repair["original_source"], task_spec=spec)
        after = parser.parse_program(repair["proposed_source"], task_spec=spec)
        old_correct = {index for index, case in enumerate(public_cases)
                       if execute(before, case["input_record"]) == case["expected_label"]}
        new_correct = {index for index, case in enumerate(public_cases)
                       if execute(after, case["input_record"]) == case["expected_label"]}
        require(old_correct < new_correct,
                "An accepted repair does not satisfy strict public non-regression")
        accepted += 1
    return accepted


def check_rendering(task, domains, program, execute):
    paths = sorted(domains)
    values = [sorted(domains[path]) for path in paths]
    require(len(paths) == 4 and all(len(domain) == 3 for domain in values),
            "Expected the retained four-field, 81-combination domain")
    entries = decision_entries(task["representations"]["selected"], paths)
    checked = 0
    for combination in itertools.product(*values):
        record = record_from_fields(paths, combination)
        fields = flatten(record)
        matches = {output for terms, output in entries
                   if all(fields[field] == value for field, value in terms)}
        require(matches == {label(execute(program, record))},
                "Rendered decisions lack coverage, purity, or program equivalence")
        checked += 1
    return checked


def verify(root):
    def no_network(event, _args):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
            raise CheckError("Network access is not permitted during deterministic replay")
    sys.addaudithook(no_network)
    parser, interpreter, sandbox, schema = import_core(root)
    prepared, metadata = load_inputs(root)
    cohorts, missing = {}, []
    accepted = rendering_artifacts = rendered_domain_cases = executions = 0

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
        public_cases = task["public_build"] + task["public_audit"]
        require(len(public_cases) == 54 and len(task["cases"]) == 27, "Unexpected case partition size")
        spec, domains = make_schema(task, public_cases, schema)
        row = metadata[(task["cohort"], task["number"])]
        programs = load_programs(root, task, row, spec, parser)
        score = cohorts.setdefault(task["cohort"], {"parsed_tasks": 0, "initial_correct": 0,
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
                        for case in public_cases), "An approved program fails a required public case")
        accepted += check_repairs(task, public_cases, spec, parser, execute)
        if task["approved"]:
            rendered_domain_cases += check_rendering(task, domains, programs["final"], execute)
            rendering_artifacts += 1

    require(cohorts == {
        "llama": {"parsed_tasks": 24, "initial_correct": 604, "final_correct": 627, "total_final_cases": 648},
        "gemma": {"parsed_tasks": 16, "initial_correct": 385, "final_correct": 424, "total_final_cases": 432}},
        "Program-only counts differ from the retained study")
    require(len(missing) == 2 and accepted == 72, "Unexpected missing-program or accepted-repair count")
    require(rendering_artifacts == 38 and rendered_domain_cases == 3078,
            "Expected every approved artifact's complete finite-domain conversion check")
    return {"status": "passed", "mode": "deterministic_retained_program_replay",
        "model_generation_calls": 0, "network_access": "blocked by Python audit hook",
        "program_only_final_scores": cohorts, "missing_program_tasks": missing,
        "accepted_public_repairs_replayed": accepted,
        "approved_rendered_artifacts_checked": rendering_artifacts,
        "finite_domain_records_checked": rendered_domain_cases,
        "sandbox_executions": executions,
        "core_module_sha256": {module.__name__: sha256(Path(module.__file__))
                               for module in (parser, interpreter, sandbox, schema)},
        "scope": "Program-only counts include unapproved parsed candidates and exclude two missing Gemma programs. Public repair non-regression and rendering equivalence do not imply unseen-input reference correctness. Student outputs are not regenerated."}


def main(argv=None):
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root; defaults to this clone")
    arguments.add_argument("--output", type=Path, help="JSON report; default evaluation/results/reproduced/core.json")
    options = arguments.parse_args(argv)
    root = options.root.resolve()
    try:
        prepare_records(root)
        result = verify(root)
        destination = options.output or root / "evaluation/results/reproduced/core.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result, indent=2) + "\n"
        destination.write_text(payload, encoding="utf-8")
    except (CheckError, ImportError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2), file=sys.stderr)
        return 1
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
