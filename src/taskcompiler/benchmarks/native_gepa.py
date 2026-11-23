"""Pinned GEPA search for the public two-clause dialogue development task.

GEPA owns parent selection, mutation acceptance, Pareto bookkeeping and merges.
The custom proposal hook enforces the study's local, schema-constrained editor.
This is an adapted native optimizer baseline, not the paper's default proposer.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time

from taskcompiler.procedural.experiment import ensure_local
from taskcompiler.procedural.schema import content_hash
from .interventions.compiler import summarize_usage
from .interventions.program import (
    RULES, execute_events, output_schema, prompt_for, render, validate_clauses,
    validate_group, validate_program, variants,
)
from .interventions.response import equal, parse_output, select_candidate
from .interventions.teaching import editor_request, editing_schema, parse_edit, split_groups


GEPA_VERSION = "0.1.4"


class NativeGEPAError(RuntimeError):
    """A failed compilation, with all locally retained accounting attached."""

    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial
        self.calls = partial["calls"]


class _BudgetStop(RuntimeError):
    pass


def _load_gepa():
    try:
        version = importlib.metadata.version("gepa")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("Native dialogue baseline requires gepa==0.1.4; no fallback") from exc
    if version != GEPA_VERSION:
        raise ImportError(f"Native dialogue baseline requires gepa=={GEPA_VERSION}, found {version}")
    import gepa
    from gepa.core.adapter import EvaluationBatch
    return gepa, EvaluationBatch


class _Log:
    def __init__(self):
        self.rows = []

    def log(self, message):
        self.rows.append(str(message))


class _Adapter:
    def __init__(self, pack, solver, editor, *, seed, solver_slot_cap,
                 reflection_cap, on_record, evaluation_batch):
        self.pack, self.program = pack, pack["program"]
        self.solver, self.editor = solver, editor
        self.partitions = split_groups(pack, seed)
        self.phase_by_id = {g["id"]: phase for phase, groups in self.partitions.items() for g in groups}
        self.allowed_groups = {g["id"]: g for phase in ("optimization", "selection")
                               for g in self.partitions[phase]}
        self.references = {(g["id"], state): execute_events(self.program, g, rules)
                           for g in self.allowed_groups.values() for state, rules in variants().items()}
        self.candidates = {"format": {r: "{rule}" for r in RULES}, "teacher": deepcopy(pack["clauses"])}
        self.calls, self.aliases, self.evaluations, self.proposals = [], [], [], []
        self.predictions, self.cache = {}, {}
        self.logical_slots = self.native_metric_calls = self.reflections = 0
        self.solver_slot_cap, self.reflection_cap = solver_slot_cap, reflection_cap
        self.on_record, self.evaluation_batch = on_record, evaluation_batch
        self.stop_reason = None

    def candidate_id(self, candidate):
        validate_clauses(candidate)
        if candidate == self.candidates["teacher"]:
            return "teacher"
        key = "gepa_" + content_hash(candidate)[:16]
        if key in self.candidates and candidate != self.candidates[key]:
            raise ValueError("Candidate identifier collision")
        self.candidates[key] = deepcopy(candidate)
        return key

    def _ask(self, model, system, prompt, schema, *, role, phase, candidate, group=None, state=None):
        row = {"call_id": len(self.calls), "role": role, "phase": phase,
               "candidate": candidate, "group": group, "state": state,
               "system": system, "prompt": prompt, "schema": deepcopy(schema),
               "request_schema_sha256": content_hash(schema),
               "structured_decoding": bool(schema), "output": None, "error": None,
               "input_tokens": None, "output_tokens": None}
        started = time.perf_counter()
        try:
            response = model.generate(system, prompt, schema)
            row.update(asdict(response))
            row["output"] = row.pop("text")
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}",
                       input_tokens=getattr(exc, "input_tokens", None),
                       output_tokens=getattr(exc, "output_tokens", None))
            # Configuration/context errors must not be relabeled as wrong answers.
            if isinstance(exc, ValueError):
                raise
        finally:
            row["elapsed_seconds"] = time.perf_counter() - started
            self.calls.append(row)
            if self.on_record:
                self.on_record(deepcopy(row))
        return row

    def _measure(self, batch, candidate, *, native, key=None, capture_traces=False):
        validate_clauses(candidate)
        if not batch:
            raise ValueError("GEPA evaluation batch must contain public groups")
        # Native merge sampling pads its five-ID subsample with replacement when
        # only four validation groups exist. Preserve repeated metric requests;
        # each consumes four slots even when its identical response is aliased.
        for group in batch:
            if group["id"] not in self.allowed_groups or group != self.allowed_groups[group["id"]]:
                raise ValueError("GEPA attempted to access a held-out or altered public group")
        # Reserve the complete batch before any call. Aliases consume logical slots.
        if self.logical_slots + 4 * len(batch) > self.solver_slot_cap:
            raise _BudgetStop("Insufficient logical slots for a complete GEPA evaluation batch")
        key = key or self.candidate_id(candidate)
        outputs, scores, trajectories = [], [], []
        for group in batch:
            phase, values = self.phase_by_id[group["id"]], {}
            observation_rows, call_refs = [], []
            for state, rules in variants().items():
                system, prompt, schema = render(self.program, candidate, rules), prompt_for(group), output_schema(self.program)
                cache_key = (phase, system, prompt, json.dumps(schema, sort_keys=True))
                self.logical_slots += 1
                alias = {"logical_slot": self.logical_slots, "native": native,
                         "phase": phase, "candidate": key, "group": group["id"], "state": state,
                         "physical_call_index": self.cache.get(cache_key), "reused": cache_key in self.cache}
                self.aliases.append(alias)
                if cache_key not in self.cache:
                    # Set the intended call index before invoking, including failures.
                    alias["physical_call_index"] = len(self.calls)
                    row = self._ask(self.solver, system, prompt, schema, role="solver",
                                    phase=phase, candidate=key, group=group["id"], state=state)
                    self.cache[cache_key] = row["call_id"]
                row = self.calls[alias["physical_call_index"]]
                value = parse_output(row["output"], self.program)
                values[state] = value
                self.predictions.setdefault(key, {})[(group["id"], state)] = value
                call_refs.append(alias["physical_call_index"])
                observation_rows.append({"task_rules": rules, "reference": self.references[(group["id"], state)],
                                         "student_outputs": {"parent": value}})
            score = sum(equal(values[state], self.references[(group["id"], state)]) for state in variants()) / 4
            outputs.append(values)
            scores.append(score)
            trajectories.append({"group": group["id"], "family": group["family"],
                                 "input": prompt_for(group), "variant_observations": observation_rows})
            self.evaluations.append({"native": native, "candidate": key, "group": group["id"],
                                     "phase": phase, "score": score, "physical_call_indices": call_refs})
            if native:
                self.native_metric_calls += 1
        return self.evaluation_batch(outputs=outputs, scores=scores,
                                     trajectories=trajectories if capture_traces else None,
                                     num_metric_calls=len(batch))

    def evaluate(self, batch, candidate, capture_traces=False):
        return self._measure(batch, candidate, native=True, capture_traces=capture_traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if not eval_batch.trajectories:
            raise ValueError("GEPA reflection requires actual public optimization traces")
        for trace in eval_batch.trajectories:
            if self.phase_by_id[trace["group"]] != "optimization":
                raise ValueError("Selection/audit feedback may not reach the local proposer")
        return {component: deepcopy(eval_batch.trajectories) for component in components_to_update}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        if self.reflections >= self.reflection_cap:
            raise _BudgetStop("Local reflection cap exhausted")
        self.reflections += 1
        evidence = reflective_dataset[components_to_update[0]]
        if any(reflective_dataset[c] != evidence for c in components_to_update):
            raise ValueError("Inconsistent GEPA component evidence")
        signal = {"kind": "native_gepa_reflection", "objective": "mean exact accuracy across all four task-rule variants",
                  "instruction": "Use the observed public errors to improve the selected instruction component."}
        system, prompt = editor_request(self.program, candidate, components_to_update, evidence, signal)
        key = self.candidate_id(candidate)
        row = self._ask(self.editor, system, prompt, editing_schema(candidate, components_to_update),
                        role="editor", phase="proposal", candidate=key)
        error = None
        try:
            edited = parse_edit(row["output"], candidate, components_to_update)
        except (ValueError, TypeError) as exc:
            edited, error = deepcopy(candidate), f"{type(exc).__name__}: {exc}"
        self.proposals.append({"parent": key, "components": list(components_to_update),
                               "physical_call_index": row["call_id"], "error": error,
                               "result": self.candidate_id(edited), "retained_parent": edited == candidate})
        # The documented GEPA hook returns ONLY requested component texts.
        return {component: edited[component] for component in components_to_update}

    def stop(self, _state):
        if self.reflections >= self.reflection_cap:
            self.stop_reason = "reflection_cap"
        elif self.solver_slot_cap - self.logical_slots < 48:
            self.stop_reason = "insufficient_slots_for_complete_iteration"
        return self.stop_reason is not None

    def payload(self):
        return {"candidates": deepcopy(self.candidates), "calls": deepcopy(self.calls),
                "request_aliases": deepcopy(self.aliases), "group_evaluations": deepcopy(self.evaluations),
                "proposals": deepcopy(self.proposals),
                "predictions": [{"candidate": key, "group": group, "state": state, "output": value}
                                for key, values in self.predictions.items() for (group, state), value in values.items()],
                "split_ids": {phase: [g["id"] for g in groups] for phase, groups in self.partitions.items()},
                "usage": {**summarize_usage(self.calls), "logical_solver_slots": self.logical_slots,
                          "physical_solver_calls": sum(r["role"] == "solver" for r in self.calls),
                          "reflection_calls": self.reflections, "native_metric_calls": self.native_metric_calls,
                          "request_aliases": sum(r["reused"] for r in self.aliases)}}


def run_native_gepa(pack, solver, editor, *, seed, solver_slot_cap=192,
                    reflection_cap=3, on_record=None, run_dir=None):
    """Optimize only public optimization/selection groups; never evaluate audit cases.

    A native metric is one four-variant dialogue group. The 32 format-control
    slots count toward the global logical cap. At most three 900-token local
    rewrites are allowed. Synthetic audit/inference is a separate runner stage.
    A fresh run_dir may retain official engine checkpoints, but never resumes.
    """
    ensure_local(solver)
    ensure_local(editor)
    if editor.identity.get("max_output_tokens") != 900:
        raise ValueError("The local native GEPA editor must declare max_output_tokens=900")
    if ({k: v for k, v in solver.identity.items() if k != "max_output_tokens"}
            != {k: v for k, v in editor.identity.items() if k != "max_output_tokens"}):
        raise ValueError("Solver and editor must use identical frozen model/settings except output cap")
    if solver.identity["mode"] == "live" and solver.identity.get("max_output_tokens") != 512:
        raise ValueError("The live native GEPA solver must declare max_output_tokens=512")
    if type(seed) is not int:
        raise ValueError("GEPA seed must be an integer")
    if type(solver_slot_cap) is not int or not 48 <= solver_slot_cap <= 192 or solver_slot_cap % 4:
        raise ValueError("Logical solver cap must be a multiple of four in [48,192]")
    if type(reflection_cap) is not int or not 0 <= reflection_cap <= 3:
        raise ValueError("Reflection cap must be an integer in [0,3]")
    validate_program(pack["program"])
    validate_clauses(pack["clauses"])
    if len(pack["groups"]) != 12 or len({g["id"] for g in pack["groups"]}) != 12:
        raise ValueError("Native baseline requires twelve distinct public groups")
    for group in pack["groups"]:
        validate_group(group, pack["program"])
    partitions = split_groups(pack, seed)
    if any(len(groups) != 4 for groups in partitions.values()):
        raise ValueError("Native baseline requires four public groups in each fixed partition")
    if run_dir is not None and Path(run_dir).exists():
        raise ValueError("Native GEPA run_dir must be fresh; resume/reuse is not permitted")
    package, evaluation_batch = _load_gepa()
    adapter = _Adapter(pack, solver, editor, seed=seed, solver_slot_cap=solver_slot_cap,
                       reflection_cap=reflection_cap, on_record=on_record, evaluation_batch=evaluation_batch)
    logger = _Log()
    options = {"candidate_selection_strategy": "pareto", "frontier_type": "instance",
               "module_selector": "round_robin", "reflection_minibatch_size": 4,
               "skip_perfect_score": False, "use_merge": True, "max_merge_invocations": 3,
               "merge_val_overlap_floor": 4, "val_evaluation_policy": "full_eval",
               "cache_evaluation": False, "max_metric_calls": (solver_slot_cap - 32) // 4,
               "seed": seed}
    try:
        for phase in ("optimization", "selection"):
            adapter._measure(partitions[phase], adapter.candidates["format"], native=False, key="format")
        result = package.optimize(seed_candidate={rule: pack["clauses"][rule] for rule in RULES},
                                  trainset=partitions["optimization"], valset=partitions["selection"],
                                  adapter=adapter, stop_callbacks=adapter.stop, logger=logger,
                                  run_dir=str(run_dir) if run_dir is not None else None,
                                  track_best_outputs=False, display_progress_bar=False,
                                  use_wandb=False, use_mlflow=False, raise_on_exception=True, **options)
        best = adapter.candidate_id(result.best_candidate)
        if result.total_metric_calls != adapter.native_metric_calls:
            raise ValueError("Official GEPA metric counter disagrees with retained grouped evaluations")
        selection_ids = [g["id"] for g in partitions["selection"]]
        decision = select_candidate({key: adapter.predictions[key] for key in {"format", best}},
                                    adapter.references, selection_ids,
                                    objective="augmented_accuracy", baseline="format")
        return {"version": 1, "method": "native_gepa_local", "status": "completed",
                **adapter.payload(), "selected": {"native_gepa": decision["selected"]},
                "guard": decision, "native_best": best, "native_state": result.to_dict(),
                "native_log": logger.rows, "native_options": options,
                "stop_reason": adapter.stop_reason or "native_metric_cap",
                "synthetic_audit": None,
                "provenance": {"gepa_version": GEPA_VERSION, "solver": deepcopy(solver.identity),
                               "editor": deepcopy(editor.identity), "pack_hash": content_hash(pack),
                               "solver_slot_cap": solver_slot_cap, "reflection_cap": reflection_cap,
                               "component_order": list(RULES),
                               "adaptations": ["local constrained proposal hook", "four-variant grouped metric",
                                               "phase-scoped exact-request aliases charged as logical slots",
                                               "format retention guard after native best selection",
                                               "whole-batch and conservative iteration budget guards"]}}
    except Exception as exc:
        partial = {"status": "failed", **adapter.payload(), "native_log": logger.rows,
                   "native_options": options, "error": f"{type(exc).__name__}: {exc}"}
        raise NativeGEPAError(partial["error"], partial) from exc
