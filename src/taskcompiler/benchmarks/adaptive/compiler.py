"""Comparison selectors for the next, separately versioned experiment.

The pooled selector is an engineering control, not a claimed novel optimizer.
Only generated preparation material enters this module.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import itertools
import json

from taskcompiler.benchmarks.compiler import task_adapter
from taskcompiler.procedural.experiment import CallLedger, ensure_local, normalized_input_hash
from taskcompiler.procedural.schema import content_hash
from . import teaching

METHODS = ["raw", "format_only", "raw_or_full", "corrections", "examples",
           "corrections_examples", "pooled", "gepa_local", "mipro_local"]

FORMAT_REMINDERS = {
    "ifeval": "Before returning the response, check every requested output constraint. Return the requested response itself, without an extra preamble.",
    "logical_deduction_three_objects": "Your complete response must be exactly (A), (B), or (C). Do not add any other text.",
    "boolean_expressions": "Your complete response must be exactly True or False. Do not add any other text.",
    "object_counting": "Your complete response must be one nonnegative integer. Do not add any other text.",
}


class ProposalBankModel:
    """Expose the same frozen teacher bank to native local proposal calls.

    The pinned native bridges explicitly mark these calls with the prefix below.
    Solver calls retain their own candidate instruction. All extra tokens are
    included in the underlying backend's recorded usage.
    """
    def __init__(self, model, pack):
        self.model = model
        self.identity = dict(model.identity)
        self.bank = json.dumps({key: pack[key] for key in
                               ("full_instructions", "corrections", "demo_pairs")},
                              ensure_ascii=False, sort_keys=True)
        self.bank_hash = hashlib.sha256(self.bank.encode()).hexdigest()

    def generate(self, system, prompt, schema):
        if system.startswith("Improve reusable instructions locally."):
            system += ("\n\nThe following frozen public teaching bank is also available. "
                       "You may use, combine, or revise its guidance. It contains only "
                       "generated preparation examples, never evaluation inputs.\n"
                       "<public_teaching_bank>\n" + self.bank + "\n</public_teaching_bank>")
        return self.model.generate(system, prompt, schema)


def compile_instructions(pack, model, method, config, evaluator, *, selection_repeats=2):
    teaching.validate_pack(pack, evaluator)
    config.validate()
    ensure_local(model)
    if method not in METHODS or config.candidates != 4 or config.shortlist != 2:
        raise ValueError("Expected a declared comparison method and four teacher candidates")
    if type(selection_repeats) is not int or selection_repeats < 1:
        raise ValueError("selection_repeats must be a positive integer")
    if pack["provenance"]["kind"] == "fixture" and model.identity["mode"] != "fixture":
        raise ValueError("Fixture teaching packs cannot produce live evidence")
    adapter = task_adapter(pack, evaluator)
    splits = teaching.split_cases(pack, config.seed)

    if method in {"gepa_local", "mipro_local"}:
        from taskcompiler.procedural.gepa import optimize_gepa
        from taskcompiler.procedural.mipro import optimize_mipro
        generated = teaching.all_cases(pack)
        native_splits = {key: list(rows) for key, rows in splits.items()}
        practice_ids = {case["id"] for rows in splits.values() for case in rows}
        native_splits["optimization"] += [c for c in generated if c["id"] not in practice_ids]
        converted = {**pack, "cases": generated,
                     "teacher_instruction": pack["full_instructions"][0]["text"]}
        bank_model = ProposalBankModel(model, pack)
        # Eight native candidates produce twelve MIPRO trials instead of the
        # old six. Actual completed trials and budget stops remain reported.
        native_config = replace(config, candidates=8)
        optimizer = optimize_gepa if method == "gepa_local" else optimize_mipro
        extra = {} if method == "gepa_local" else {"max_labeled_demos": 3, "max_bootstrapped_demos": 1}
        result = optimizer(converted, bank_model, native_config,
                           prepared_splits=native_splits, task_adapter=adapter, **extra)
        bundle = result["bundle"]
        bundle.update(protocol="adaptive_components_v2", task_pack_hash=content_hash(pack),
                      config=asdict(config), native_config=asdict(native_config),
                      selection_repeats=selection_repeats,
                      actual_native_selection_repeats="native_optimizer_schedule",
                      teacher_bank_hash=bank_model.bank_hash,
                      teacher_bank_access="all_full_instructions_corrections_and_demonstrations")
        bundle["optimizer_protocol"]["adaptations"].append("complete frozen teacher bank available to local proposals")
        bundle["bundle_hash"] = content_hash({k: v for k, v in bundle.items() if k != "bundle_hash"})
        return result

    corrections = {row["id"]: row for row in pack["corrections"]}
    pairs = {row["id"]: row for row in pack["demo_pairs"]}
    linked = {row["correction_id"]: row["id"] for row in pairs.values()}

    def candidate(identifier, kind, clauses=(), demos=(), full=None, reminder=False):
        clauses = [r["id"] for r in pack["corrections"] if r["id"] in clauses]
        demos = [r["id"] for r in pack["demo_pairs"] if r["id"] in demos]
        parts = [adapter.contract]
        if reminder:
            parts.append(FORMAT_REMINDERS[pack["task_id"]])
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
        return {"id": identifier, "kind": kind, "instruction": instruction,
                "instruction_hash": hashlib.sha256(instruction.encode()).hexdigest(),
                "correction_ids": clauses, "example_pair_ids": demos,
                "demonstration_ids": [c["id"] for c in examples],
                "unit_count": len(clauses) + len(examples) + bool(full) + reminder}

    def union(rows, kind):
        return candidate("union_" + kind, kind,
                         {c for row in rows for c in row["correction_ids"]},
                         {p for row in rows for p in row["example_pair_ids"]})

    def tie(row):
        return row["unit_count"], len(row["instruction"]), row["id"]

    raw = candidate("raw", "raw")
    categories = {
        "full": [candidate("full_" + r["id"], "full", full=r) for r in pack["full_instructions"]],
        "corrections": [candidate("correction_" + r["id"], "corrections", clauses=[r["id"]]) for r in pack["corrections"]],
        "examples": [candidate("examples_" + r["id"], "examples", demos=[r["id"]]) for r in pack["demo_pairs"]],
        "joint": [candidate("joint_" + r["id"], "joint", clauses=[r["id"]], demos=[linked[r["id"]]]) for r in pack["corrections"]],
    }
    kinds = {"raw": [], "format_only": [], "raw_or_full": ["full"],
             "corrections": ["corrections"], "examples": ["examples"],
             "corrections_examples": ["joint"], "pooled": list(categories)}[method]
    singles = [row for kind in kinds for row in categories[kind]]
    union_kinds = [kind for kind in kinds if kind != "full"]
    for kind in union_kinds:
        for rows in itertools.combinations(categories[kind], 2):
            union(rows, kind)  # All possible instruction caps checked before scoring.
    finalists_count = 1 + (len(kinds) if method == "pooled" else min(2, len(singles) + len(union_kinds)))
    required = ((1 + len(singles) + len(union_kinds)) * len(splits["optimization"])
                + finalists_count * len(splits["selection"]) * selection_repeats) if singles else 0
    if required > config.max_calls:
        raise ValueError("Search exceeds the declared call cap")
    ledger = CallLedger(model, config.max_calls, task_adapter=adapter)
    candidates, scores = [raw, *singles], {}

    def evaluate(row, phase, repeats=1):
        batches = []
        for repeat in range(repeats):
            rows = [ledger.ask(row["instruction"], c, {}, phase, row["id"]) for c in splits[phase]]
            for call in rows:
                call["repeat_index"] = repeat
            batches.append(sum(r["score"] for r in rows))
        scores.setdefault(row["id"], {})[phase] = sum(batches) / repeats
        scores[row["id"]][phase + "_repeats"] = batches

    selected = candidate("format_only", "format", reminder=True) if method == "format_only" else raw
    if singles:
        for row in candidates:
            evaluate(row, "optimization")
        for kind in union_kinds:
            ranked = sorted(categories[kind], key=lambda r: (-scores[r["id"]]["optimization"], *tie(r)))
            combined = union(ranked[:2], kind)
            candidates.append(combined)
            evaluate(combined, "optimization")
        if method == "pooled":
            shortlist = [min((r for r in candidates if r["kind"] == kind),
                             key=lambda r: (-scores[r["id"]]["optimization"], *tie(r))) for kind in kinds]
        else:
            shortlist = sorted(candidates[1:], key=lambda r: (-scores[r["id"]]["optimization"], *tie(r)))[:2]
        for row in [raw, *shortlist]:
            evaluate(row, "selection", selection_repeats)
        best = min(shortlist, key=lambda r: (-scores[r["id"]]["selection"], *tie(r)))
        if scores[best["id"]]["selection"] > scores["raw"]["selection"]:
            selected = best
    if method == "format_only":
        candidates.append(selected)
    bundle = {"version": 2, "protocol": "adaptive_components_v2", "task_id": pack["task_id"],
              "method": method, "mode": model.identity["mode"], "task_pack_hash": content_hash(pack),
              "output_schema": {}, "task_adapter": adapter.metadata(), "config": asdict(config),
              "selection_repeats": selection_repeats, "student_identity": dict(model.identity),
              "selected": selected, "candidates": candidates, "candidate_scores": scores,
              "split_ids": {phase: [c["id"] for c in rows] for phase, rows in splits.items()},
              "generated_input_hashes": [normalized_input_hash(c["input"]) for c in teaching.all_cases(pack)],
              "usage": ledger.summary(), "required_compilation_calls": required,
              "compile_status": "partial" if any(r["error"] for r in ledger.rows) else "completed",
              "search_reference_kind": "generated_constraints_or_generated_answers",
              "teacher_access": "public_description_and_fixed_protocol_only",
              "novelty_status": "comparison_control_no_new_algorithm_claim"}
    bundle["bundle_hash"] = content_hash(bundle)
    return {"bundle": bundle, "calls": ledger.rows}
