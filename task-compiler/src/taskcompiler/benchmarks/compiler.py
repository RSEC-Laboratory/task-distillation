"""Native-text version of the frozen short-correction search schedule.

Only synthetic practice and its scorer enter compilation. Independent benchmark
cases are accepted exclusively by the later evaluation stage.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import itertools

from taskcompiler.procedural.experiment import CallLedger, TaskAdapter, ensure_local, normalized_input_hash
from taskcompiler.procedural.schema import content_hash
from .teaching import all_cases, split_cases, validate_pack

METHODS = ["raw", "raw_or_full", "corrections", "examples", "corrections_examples", "gepa_local", "mipro_local"]


def task_adapter(pack, evaluator):
    return TaskAdapter(contract=pack["description"], scorer_id=evaluator.scorer_id(pack["task_id"]),
                       score=evaluator, reference_text=lambda case: case["reference"], output_schema={})


def compile_instructions(pack, model, method, config, evaluator):
    validate_pack(pack, evaluator)
    config.validate()
    ensure_local(model)
    if method not in METHODS or config.candidates != 4 or config.shortlist != 2:
        raise ValueError("Expected a locked seven-arm/four-candidate configuration")
    if pack["provenance"]["kind"] == "fixture" and model.identity["mode"] != "fixture":
        raise ValueError("Fixture teaching packs cannot be used as live evidence")
    adapter = task_adapter(pack, evaluator)
    splits = split_cases(pack, config.seed)
    if method in {"gepa_local", "mipro_local"}:
        from taskcompiler.procedural.gepa import optimize_gepa
        from taskcompiler.procedural.mipro import optimize_mipro
        generated = all_cases(pack)
        practice_ids = {c["id"] for rows in splits.values() for c in rows}
        splits["optimization"] += [c for c in generated if c["id"] not in practice_ids]
        converted = {**pack, "cases": generated, "teacher_instruction": pack["full_instructions"][0]["text"]}
        if method == "gepa_local":
            result = optimize_gepa(converted, model, config, prepared_splits=splits, task_adapter=adapter)
        else:
            result = optimize_mipro(converted, model, config, prepared_splits=splits, task_adapter=adapter,
                                    max_labeled_demos=3, max_bootstrapped_demos=1)
        result["bundle"]["task_pack_hash"] = content_hash(pack)
        result["bundle"]["general_benchmark_protocol"] = "native_text_pilot_v1"
        result["bundle"]["bundle_hash"] = content_hash({k: v for k, v in result["bundle"].items() if k != "bundle_hash"})
        return result

    corrections = {r["id"]: r for r in pack["corrections"]}
    pairs = {r["id"]: r for r in pack["demo_pairs"]}
    linked = {r["correction_id"]: r["id"] for r in pairs.values()}

    def candidate(identifier, clauses=(), demos=(), full=None):
        clauses = [r["id"] for r in pack["corrections"] if r["id"] in clauses]
        demos = [r["id"] for r in pack["demo_pairs"] if r["id"] in demos]
        parts = [adapter.contract]
        if full:
            parts.append("Additional task instructions:\n" + full["text"])
        if clauses:
            parts.append("Task guidance:\n" + "\n".join(corrections[c]["text"] for c in clauses))
        examples = [c for pair in demos for c in pairs[pair]["cases"]]
        if examples:
            parts.append("Generated worked examples:\n" + "\n\n".join(
                "Input:\n" + c["input"] + "\nOutput:\n" + c["reference"] for c in examples))
        instruction = "\n\n".join(parts)
        if len(instruction) > config.max_instruction_chars:
            raise ValueError("Rendered candidate exceeds the instruction cap")
        return {"id": identifier, "instruction": instruction,
                "instruction_hash": hashlib.sha256(instruction.encode()).hexdigest(),
                "correction_ids": clauses, "example_pair_ids": demos,
                "demonstration_ids": [c["id"] for c in examples],
                "unit_count": len(clauses) + len(examples) + bool(full)}

    def combine(rows):
        return candidate("combined", {c for r in rows for c in r["correction_ids"]},
                         {p for r in rows for p in r["example_pair_ids"]})

    def tie(row):
        return row["unit_count"], len(row["instruction"]), row["id"]

    raw = candidate("raw")
    if method == "raw_or_full":
        singles = [candidate("full_" + r["id"], full=r) for r in pack["full_instructions"]]
    elif method == "examples":
        singles = [candidate("examples_" + r["id"], demos=[r["id"]]) for r in pairs.values()]
    elif method in {"corrections", "corrections_examples"}:
        singles = [candidate("correction_" + r["id"], [r["id"]], [linked[r["id"]]] if method.endswith("_examples") else ()) for r in corrections.values()]
    else:
        singles = []
    use_union = method in {"corrections", "examples", "corrections_examples"}
    if use_union:
        for pair in itertools.combinations(singles, 2):
            combine(pair)  # Check every possible union before observing model scores.
    required = ((1 + len(singles) + use_union) * len(splits["optimization"]) + 3 * len(splits["selection"])) if singles else 0
    if required > config.max_calls:
        raise ValueError("Search exceeds the frozen call cap")
    ledger = CallLedger(model, config.max_calls, task_adapter=adapter)
    candidates, scores = [raw, *singles], {}

    def evaluate(row, phase):
        rows = [ledger.ask(row["instruction"], c, {}, phase, row["id"]) for c in splits[phase]]
        scores.setdefault(row["id"], {})[phase] = sum(r["score"] for r in rows)

    selected = raw
    if singles:
        for row in candidates:
            evaluate(row, "optimization")
        ranked = sorted(singles, key=lambda r: (-scores[r["id"]]["optimization"], *tie(r)))
        if use_union:
            union = combine(ranked[:2])
            candidates.append(union)
            evaluate(union, "optimization")
        shortlist = sorted(candidates[1:], key=lambda r: (-scores[r["id"]]["optimization"], *tie(r)))[:2]
        for row in [raw, *shortlist]:
            evaluate(row, "selection")
        best = min(shortlist, key=lambda r: (-scores[r["id"]]["selection"], *tie(r)))
        if scores[best["id"]]["selection"] > scores["raw"]["selection"]:
            selected = best
    bundle = {"version": 1, "protocol": "native_text_corrections_v1", "task_id": pack["task_id"],
              "method": method, "mode": model.identity["mode"], "task_pack_hash": content_hash(pack),
              "output_schema": {}, "task_adapter": adapter.metadata(), "config": asdict(config),
              "student_identity": dict(model.identity), "selected": selected,
              "candidates": candidates, "candidate_scores": scores,
              "split_ids": {p: [c["id"] for c in rows] for p, rows in splits.items()},
              "generated_input_hashes": [normalized_input_hash(c["input"]) for c in all_cases(pack)],
              "synthetic_holdout_hashes": {c["id"]: content_hash(c) for c in splits["synthetic_holdout"]},
              "usage": ledger.summary(), "required_compilation_calls": required,
              "compile_status": "partial" if any(r["error"] for r in ledger.rows) else "completed",
              "search_reference_kind": "generated_constraints_or_generated_answers",
              "teacher_access": "public_description_and_fixed_protocol_only"}
    bundle["bundle_hash"] = content_hash(bundle)
    return {"bundle": bundle, "calls": ledger.rows}
