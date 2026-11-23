"""Frozen-student search over short corrections and generated demonstrations.

Only a validated description-generated pack enters compilation. Raw is a measured
incumbent, not a promised lower bound on independent accuracy. Generated references
remain unverified; accepted changes improve only the recorded selection score.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json

from taskcompiler.procedural.experiment import (
    CallLedger, LocalModel, SearchConfig, ensure_local, normalized_input_hash,
)
from taskcompiler.procedural.schema import content_hash
from .schema import common_contract, split_cases, validate_pack

METHODS = ("raw", "raw_or_full", "corrections", "examples", "corrections_examples")


def _candidate(pack, identifier, *, corrections=(), pairs=(), full=None):
    correction_by_id = {row["id"]: row for row in pack["corrections"]}
    pair_by_id = {row["id"]: row for row in pack["demo_pairs"]}
    # Preserve the teacher's declared order when combining independently useful units.
    correction_ids = [row["id"] for row in pack["corrections"] if row["id"] in corrections]
    pair_ids = [row["id"] for row in pack["demo_pairs"] if row["id"] in pairs]
    fragments = [common_contract(pack)]
    if full is not None:
        fragments.append("Additional task instructions:\n" + full["text"])
    if correction_ids:
        fragments.append("Apply these task-specific corrections:\n" + "\n".join(
            f"{index}. {correction_by_id[key]['text']}" for index, key in enumerate(correction_ids, 1)))
    demonstrations = [case for key in pair_ids for case in pair_by_id[key]["cases"]]
    if demonstrations:
        examples = []
        for index, case in enumerate(demonstrations, 1):
            examples.append(f"Example {index}\nInput:\n{case['input']}\nProposed output:\n" +
                            json.dumps(case["reference"], ensure_ascii=False, sort_keys=True))
        fragments.append("Generated worked examples:\n" + "\n\n".join(examples))
    instruction = "\n\n".join(fragments)
    return {"id": identifier, "instruction": instruction,
            "instruction_hash": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            "correction_ids": correction_ids, "example_pair_ids": pair_ids,
            "demonstration_ids": [case["id"] for case in demonstrations],
            "full_instruction_id": full["id"] if full else None,
            "unit_count": len(correction_ids) + len(demonstrations) + int(full is not None)}


def _single_variants(pack, method):
    if method == "raw_or_full":
        return [_candidate(pack, "full_" + row["id"], full=row) for row in pack["full_instructions"]]
    if method == "corrections":
        return [_candidate(pack, "correction_" + row["id"], corrections=[row["id"]])
                for row in pack["corrections"]]
    if method == "examples":
        return [_candidate(pack, "examples_" + row["id"], pairs=[row["id"]]) for row in pack["demo_pairs"]]
    if method == "corrections_examples":
        linked = {row["correction_id"]: row["id"] for row in pack["demo_pairs"]}
        return [_candidate(pack, "correction_examples_" + row["id"], corrections=[row["id"]],
                           pairs=[linked[row["id"]]]) for row in pack["corrections"]]
    return []


def _union(pack, candidates):
    return _candidate(pack, "combined",
                      corrections={key for row in candidates for key in row["correction_ids"]},
                      pairs={key for row in candidates for key in row["example_pair_ids"]})


def _cost_tie_key(candidate):
    return candidate["unit_count"], len(candidate["instruction"]), candidate["id"]


def _comparison(rows, raw_rows):
    by_id = {row["case_id"]: row for row in rows}
    raw_by_id = {row["case_id"]: row for row in raw_rows}
    if len(by_id) != len(rows) or len(raw_by_id) != len(raw_rows) or by_id.keys() != raw_by_id.keys():
        raise RuntimeError("Paired comparison requires complete, distinct, matching case IDs")
    fixes = [key for key in by_id if by_id[key]["correct"] and not raw_by_id[key]["correct"]]
    breaks = [key for key in by_id if raw_by_id[key]["correct"] and not by_id[key]["correct"]]
    return {"n": len(rows), "fixes": len(fixes), "breaks": len(breaks),
            "fix_case_ids": fixes, "break_case_ids": breaks,
            "net_correct_gain": len(fixes) - len(breaks),
            "reference_kind": "teacher_generated_unverified"}


def compile_instructions(pack: dict, model: LocalModel, method: str, config: SearchConfig) -> dict:
    """Compile without any external evaluation or private-input parameter.

    The locked schedule scores raw plus four singleton variants on optimization.
    Correction/example searches additionally score one union of their best two
    singleton variants, then evaluate raw and the best two nonraw configurations
    on the separate selection set. There is no selection-set-driven revision.
    """
    validate_pack(pack)
    config.validate()
    ensure_local(model)
    if method not in METHODS:
        raise ValueError(f"Unknown corrections method: {method}")
    if config.candidates != 4 or config.shortlist != 2:
        raise ValueError("The corrections protocol requires four candidates and a shortlist of two")
    if pack["provenance"]["kind"] == "fixture" and model.identity["mode"] != "fixture":
        raise ValueError("Fixture task packs cannot produce live empirical evidence")
    pack = deepcopy(pack)
    splits = split_cases(pack, config.seed)
    if {name: len(rows) for name, rows in splits.items()} != {
        "optimization": 16, "selection": 8, "synthetic_holdout": 8,
    }:
        raise ValueError("The locked corrections split requires 16 optimization, 8 selection, and 8 holdout cases")
    raw = _candidate(pack, "raw")
    singles = _single_variants(pack, method)
    combine = method in {"corrections", "examples", "corrections_examples"}
    possible = [raw, *singles]
    if combine:
        possible += [_union(pack, pair) for pair in itertools.combinations(singles, 2)]
    if any(len(row["instruction"]) > config.max_instruction_chars for row in possible):
        raise ValueError("A possible instruction exceeds the common instruction length cap")
    if len({row["instruction_hash"] for row in [raw, *singles]}) != 1 + len(singles):
        raise ValueError("Raw and singleton variants must have distinct rendered instructions")
    required = (len(singles) + 1 + int(combine)) * len(splits["optimization"]) + 3 * len(splits["selection"]) if singles else 0
    if required > config.max_calls:
        raise ValueError(f"Locked corrections search needs {required} calls, exceeding budget {config.max_calls}")
    ledger = CallLedger(model, config.max_calls)
    candidates = [raw, *singles]
    scores, phase_rows = {}, {}
    optimization_order, selection_order, combined_from = [], [], []

    def evaluate(candidate, phase):
        rows = [ledger.ask(candidate["instruction"], case, pack["output_schema"], phase, candidate["id"])
                for case in splits[phase]]
        if len(rows) != len(splits[phase]) or {row["case_id"] for row in rows} != {case["id"] for case in splits[phase]}:
            raise RuntimeError("Incomplete candidate evaluation cannot enter selection")
        phase_rows[(candidate["id"], phase)] = rows
        score = scores.setdefault(candidate["id"], {"candidate_id": candidate["id"],
                                                   "optimization_correct": None, "selection_correct": None})
        score[phase + "_correct"] = sum(row["correct"] for row in rows)
        score[phase + "_n"] = len(rows)
        score[phase + "_complete"] = True
        score[phase + "_comparison"] = _comparison(rows, phase_rows[("raw", phase)])
        (optimization_order if phase == "optimization" else selection_order).append(candidate["id"])

    selected, best_nonraw = raw, None
    if singles:
        for candidate in candidates:
            evaluate(candidate, "optimization")
        ranked = sorted(singles, key=lambda row: (-scores[row["id"]]["optimization_correct"], *_cost_tie_key(row)))
        if combine:
            combined_from = [row["id"] for row in ranked[:2]]
            combined = _union(pack, ranked[:2])
            candidates.append(combined)
            evaluate(combined, "optimization")
        shortlist = sorted(candidates[1:], key=lambda row: (-scores[row["id"]]["optimization_correct"], *_cost_tie_key(row)))[:2]
        for candidate in [raw, *shortlist]:
            evaluate(candidate, "selection")
        best_nonraw = min(shortlist, key=lambda row: (-scores[row["id"]]["selection_correct"], *_cost_tie_key(row)))
        if scores[best_nonraw["id"]]["selection_correct"] > scores["raw"]["selection_correct"]:
            selected = best_nonraw
    failed = any(row["error"] is not None for row in ledger.rows)
    all_generated = [case for group in pack["groups"] for case in group["cases"]]
    all_generated += [case for pair in pack["demo_pairs"] for case in pair["cases"]]
    bundle = {"version": 1, "protocol": "description_only_corrections_v1",
              "task_id": pack["task_id"], "method": method, "mode": model.identity["mode"],
              "task_pack_hash": content_hash(pack),
              "description_hash": hashlib.sha256(pack["description"].encode("utf-8")).hexdigest(),
              "output_schema": deepcopy(pack["output_schema"]), "config": asdict(config),
              "student_identity": deepcopy(dict(model.identity)), "selected": selected,
              "candidates": candidates, "candidate_scores": [scores[row["id"]] for row in candidates if row["id"] in scores],
              "split_ids": {name: [case["id"] for case in cases] for name, cases in splits.items()},
              "generated_input_hashes": [normalized_input_hash(case["input"]) for case in all_generated],
              "synthetic_holdout_hashes": {case["id"]: content_hash(case) for case in splits["synthetic_holdout"]},
              "demonstration_ids": [case["id"] for pair in pack["demo_pairs"] for case in pair["cases"]],
              "optimization_order": optimization_order, "selection_order": selection_order,
              "combined_from_candidate_ids": combined_from, "required_compilation_calls": required,
              "acceptance": {"policy": "Strict improvement over raw on generated selection cases; ties retain raw.",
                             "nonraw_selected": selected["id"] != "raw",
                             "best_nonraw_id": best_nonraw["id"] if best_nonraw else None,
                             "best_nonraw_selection_gain": scores[best_nonraw["id"]]["selection_correct"] - scores["raw"]["selection_correct"] if best_nonraw else None,
                             "nonraw_tie_break": "fewer correction/demo/full-instruction units, shorter rendered text, candidate ID; optimization score is not a selection tie-break"},
              "usage": ledger.summary(), "compile_status": "partial" if failed else "completed",
              "search_reference_kind": "teacher_generated_unverified", "novelty_status": "not_established",
              "semantic_equivalence": "not_proven", "teacher_access": "public_description_and_fixed_protocol_only"}
    bundle["bundle_hash"] = content_hash(bundle)
    return {"bundle": bundle, "calls": ledger.rows}
