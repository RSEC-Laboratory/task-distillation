"""Official DSPy MIPROv2 with a frozen local proposal model and shared call cap.

Pinned API: https://github.com/stanfordnlp/dspy/tree/3.4.0
This access-constrained baseline is not a reproduction of published MIPRO results.
Native bootstrapping, grounded proposals, and Optuna search remain enabled. The
legacy default freezes no demonstrations. Explicit demo limits permit native
instruction-and-example optimization; all internal calls share the same cap.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import time
from dataclasses import asdict

from .experiment import (
    CallLedger, LocalModel, SearchConfig, common_contract, ensure_local,
    normalized_input_hash, score_output, TaskAdapter, TaskScoringError, _ScoringAbort,
    optimizer_splits, validate_optimizer_pack,
)
from .schema import content_hash

DSPY_VERSION = "3.4.0"


class MIPROCompilationError(RuntimeError):
    def __init__(self, message: str, calls: list[dict]):
        super().__init__(message)
        self.calls = [dict(row) for row in calls]


class _BudgetStop(BaseException):
    # DSPy's proposal/bootstrap helpers catch Exception and may continue. This
    # private control signal must escape those handlers; it is caught below.
    pass


def _load_dspy():
    try:
        installed = importlib.metadata.version("dspy")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("mipro_local requires official dspy[optuna]==3.4.0; no fallback optimizer") from exc
    if installed != DSPY_VERSION:
        raise ImportError(f"mipro_local requires dspy=={DSPY_VERSION}, found {installed}")
    try:
        importlib.metadata.version("optuna")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("mipro_local requires the official DSPy optuna extra") from exc
    # DSPy 3.4 installs a lazy NumPy proxy; eagerly loading Optuna first avoids
    # a circular NumPy import on the supported 1.26 runtime. No algorithm patch.
    importlib.import_module("optuna")
    return importlib.import_module("dspy")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def optimize_mipro(
    pack: dict, model: LocalModel, config: SearchConfig, *,
    prepared_splits: dict | None = None, max_labeled_demos: int = 0,
    max_bootstrapped_demos: int = 0,
    task_adapter: TaskAdapter | None = None,
) -> dict:
    """Compile with the official optimizer; expose no independent evaluation API.

    Calls include internal bootstrap solving, dataset/program descriptions,
    instruction proposals, adapter retries, and validation solving. On a budget
    stop only a fully scored selection candidate may be frozen. Other failures
    retain the attempted-call ledger in MIPROCompilationError.
    """
    validate_optimizer_pack(pack, task_adapter)
    config.validate()
    ensure_local(model)
    if pack["provenance"]["kind"] == "fixture" and model.identity["mode"] != "fixture":
        raise ValueError("Fixture task packs cannot be used for live empirical runs")
    for name, value in (("max_labeled_demos", max_labeled_demos),
                        ("max_bootstrapped_demos", max_bootstrapped_demos)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    splits = optimizer_splits(pack, config, prepared_splits, task_adapter)
    dspy = _load_dspy()
    train, val = splits["optimization"], splits["selection"]
    if config.max_calls < len(val):
        raise ValueError("MIPRO budget must cover a complete selection evaluation")
    base = task_adapter.contract if task_adapter else common_contract(pack)
    output_schema = task_adapter.output_schema if task_adapter else pack["output_schema"]
    seed_text = pack["teacher_instruction"]
    ledger = CallLedger(model, config.max_calls, task_adapter=task_adapter)
    by_input = {case["input"]: case for case in train + val}
    reference_texts = ({case["id"]: task_adapter.answer(case) for case in train + val}
                       if task_adapter else {})
    latest_scores: dict[str, tuple[str, float]] = {}
    val_ids = {case["id"] for case in val}
    completed: list[dict] = []
    pending: dict[str, dict] = {}
    rejections: list[dict] = []
    budget_stopped = False

    demos_enabled = max_labeled_demos + max_bootstrapped_demos > 0
    train_by_input = {case["input"]: case for case in train}

    def rendered_demos(demos):
        rendered = []
        for demo in demos:
            if demo.get("input") not in train_by_input:
                raise ValueError("MIPRO demonstration is outside generated training inputs")
            if not isinstance(demo.get("answer"), str):
                raise ValueError("MIPRO demonstration answer must be text")
            rendered.append({"input": demo["input"], "answer": demo["answer"]})
        return rendered

    def instruction(fragment, demonstrations=None, *, internal=False):
        if not isinstance(fragment, str) or not fragment.strip():
            raise ValueError("MIPRO returned an empty/non-text instruction")
        full = base + "\n\n" + fragment
        if demonstrations:
            heading = "Internal bootstrap demonstrations:" if internal else "Demonstrations:"
            full += "\n\n" + heading + "\n" + json.dumps(demonstrations, ensure_ascii=False)
        return full

    def candidate_id(fragment, demonstrations):
        # Retain legacy instruction-only IDs. Example-bearing candidates must
        # differ when their demonstrations or their ordering differ.
        identity = (content_hash({"fragment": fragment, "demonstrations": demonstrations})
                    if demonstrations else _text_hash(fragment))
        return "mipro_" + identity[:16]

    if len(instruction(seed_text)) > config.max_instruction_chars:
        raise ValueError("MIPRO seed exceeds the common instruction length cap")

    def reserve():
        if len(ledger.rows) >= config.max_calls:
            raise _BudgetStop("Shared solver/proposal attempted-call budget exhausted")

    class LocalBridge(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="local-protocol-student", temperature=0, cache=False,
                             max_tokens=model.identity.get("max_output_tokens"), num_retries=0)

        def __call__(self, prompt=None, *, messages=None, **kwargs):
            requested = {**self.kwargs, **kwargs}
            if requested.get("n", 1) != 1 or requested.get("num_generations", 1) != 1:
                raise ValueError("MIPRO bridge requires one completion per accounted call")
            if prompt is not None and messages is not None:
                raise ValueError("Ambiguous MIPRO LM request")
            if messages is None:
                if not isinstance(prompt, str):
                    raise TypeError("MIPRO bridge accepts text-only prompts")
                system, user = "", prompt
            else:
                if any(not isinstance(item.get("content"), str) for item in messages):
                    raise TypeError("MIPRO bridge accepts text-only messages")
                system = "\n\n".join(item["content"] for item in messages if item.get("role") == "system")
                others = [item for item in messages if item.get("role") != "system"]
                user = "\n\n".join(
                    item["content"] if len(others) == 1 else f"[{item.get('role', 'user')}]\n{item['content']}"
                    for item in others
                )
            reserve()
            row = {"call_index": len(ledger.rows), "phase": "mipro_proposal",
                   "role": "optimizer", "case_id": None, "candidate_id": None,
                   "input_hash": _text_hash(system + "\n\n" + user),
                   "input_tokens": None, "output_tokens": None, "latency_seconds": 0.0,
                   "error": None, "format_valid": None, "correct": None, "output": None,
                   "requested_decoding": {k: requested[k] for k in ("temperature", "max_tokens", "rollout_id") if k in requested},
                   "actual_decoding": {"temperature": model.identity.get("temperature"),
                                       "max_output_tokens": model.identity.get("max_output_tokens"),
                                       "seed": model.identity.get("seed")},
                   "optimizer_decoding_overrides_honored": False}
            ledger.rows.append(row)
            started = time.perf_counter()
            try:
                response = model.generate(
                    "Improve reusable instructions locally. The following contract governs the task solver, "
                    "not the format of this proposal reply. Create reusable instructions; do not copy task examples "
                    "into candidate instructions:\n<fixed_task_contract>\n" + base
                    + "\n</fixed_task_contract>\n" + system, user, {},
                )
                row.update(input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                           output=response.text)
                if not isinstance(response.text, str) or not response.text.strip():
                    raise ValueError("Local MIPRO proposal returned empty text")
                return [response.text]
            except Exception as exc:
                row.update(error=f"{type(exc).__name__}: {exc}",
                           input_tokens=getattr(exc, "input_tokens", row["input_tokens"]),
                           output_tokens=getattr(exc, "output_tokens", row["output_tokens"]))
                raise
            finally:
                row["latency_seconds"] = time.perf_counter() - started

    bridge = LocalBridge()

    class CommonContractAdapter(dspy.ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            is_task = set(signature.input_fields) == {"input"} and set(signature.output_fields) == {"answer"}
            if not is_task:
                return super().__call__(lm, lm_kwargs, signature, demos, inputs)
            if set(inputs) != {"input"} or inputs["input"] not in by_input:
                raise ValueError("MIPRO attempted to access a non-optimization input")
            case = by_input[inputs["input"]]
            fragment = signature.instructions
            rendered = rendered_demos(demos)
            if rendered and case["id"] in val_ids and not demos_enabled:
                raise ValueError("MIPRO attempted selection with final demonstrations enabled")
            if (case["id"] in val_ids
                    and len(rendered) > max_labeled_demos + max_bootstrapped_demos):
                raise ValueError("MIPRO exceeded the declared final demonstration limit")
            full = instruction(fragment, rendered, internal=not demos_enabled)
            identifier = candidate_id(fragment, rendered)
            if len(full) > config.max_instruction_chars:
                rejections.append({"candidate_id": identifier, "case_id": case["id"],
                                   "reason": "common instruction length cap exceeded"})
                if task_adapter is not None:
                    latest_scores[case["id"]] = ("", 0.0)
                return [{"answer": ""}]
            reserve()
            phase = "mipro_selection" if case["id"] in val_ids else "mipro_bootstrap"
            try:
                row = ledger.ask(full, case, output_schema, phase, identifier)
            except TaskScoringError as exc:
                ledger.rows[-1]["role"] = "solver"
                raise _ScoringAbort(exc) from exc
            row["role"] = "solver"
            answer = row["output"] if isinstance(row["output"], str) else ""
            if task_adapter is not None:
                latest_scores[case["id"]] = (answer, row["score"])
            if phase == "mipro_selection":
                scores = pending.setdefault(identifier, {})
                if case["id"] in scores:
                    raise RuntimeError("MIPRO repeated a selection case before completing its batch")
                scores[case["id"]] = row["score"] if task_adapter else int(row["correct"])
                if scores.keys() == val_ids:
                    completed.append({"fragment": fragment, "demonstrations": rendered,
                                      "candidate_id": identifier,
                                      "score": sum(scores.values()) / len(val_ids),
                                      "case_ids": list(scores), "scores": list(scores.values())})
                    pending[identifier] = {}
            return [{"answer": answer}]

    signature = dspy.Signature("input: str -> answer: str", instructions=seed_text)
    program = dspy.Predict(signature)

    def metric(example, prediction, trace=None):
        case = by_input.get(example.input)
        if case is None:
            raise ValueError("MIPRO metric received a non-optimization input")
        if task_adapter is not None:
            cached = latest_scores.get(case["id"])
            if cached is None or cached[0] != prediction.answer:
                raise _ScoringAbort(TaskScoringError("MIPRO metric prediction lacks its accounted task score"))
            return cached[1]
        return float(score_output(prediction.answer, case["reference"], output_schema)["correct"])

    def examples(rows):
        return [dspy.Example(input=case["input"],
                             answer=(reference_texts[case["id"]] if task_adapter else
                                     json.dumps(case["reference"], ensure_ascii=False))).with_inputs("input")
                for case in rows]

    # Fixed finite trial schedule; demo-enabled runs use the same common cap.
    trials = int(max(2 * math.log2(config.candidates), 1.5 * config.candidates))
    result = None
    try:
        with dspy.context(lm=bridge, adapter=CommonContractAdapter(), max_errors=1):
            optimizer = dspy.MIPROv2(
                metric=metric, prompt_model=bridge, task_model=bridge,
                max_bootstrapped_demos=max_bootstrapped_demos, max_labeled_demos=max_labeled_demos,
                auto=None, num_candidates=config.candidates, num_threads=1,
                max_errors=1, seed=config.seed, init_temperature=0.0,
                verbose=False, track_stats=True,
            )
            result = optimizer.compile(
                program, trainset=examples(train), valset=examples(val),
                num_trials=trials, seed=config.seed, minibatch=False,
                program_aware_proposer=True, data_aware_proposer=True,
                tip_aware_proposer=True, fewshot_aware_proposer=True,
            )
    except _ScoringAbort as exc:
        exc.error.calls = [dict(row) for row in ledger.rows]
        raise exc.error from exc
    except TaskScoringError as exc:
        exc.calls = [dict(row) for row in ledger.rows]
        raise
    except _BudgetStop:
        budget_stopped = True
    except Exception as exc:
        raise MIPROCompilationError(
            f"Official MIPROv2 failed; no fallback optimizer was used: {type(exc).__name__}: {exc}",
            ledger.rows,
        ) from exc
    if not completed:
        raise MIPROCompilationError("MIPRO produced no fully scored selection candidate; no bundle frozen", ledger.rows)
    if budget_stopped:
        chosen = max(completed, key=lambda record: record["score"])
        selection_source = "best_completed_selection_before_budget_stop"
    else:
        predictors = result.predictors() if result is not None else []
        if len(predictors) != 1 or (predictors[0].demos and not demos_enabled):
            raise MIPROCompilationError("MIPRO returned a different program shape or final demonstrations", ledger.rows)
        fragment = predictors[0].signature.instructions
        try:
            selected_demos = rendered_demos(predictors[0].demos)
        except ValueError as exc:
            raise MIPROCompilationError(str(exc), ledger.rows) from exc
        if len(selected_demos) > max_labeled_demos + max_bootstrapped_demos:
            raise MIPROCompilationError("MIPRO returned too many final demonstrations", ledger.rows)
        selected_id = candidate_id(fragment, selected_demos)
        matches = [record for record in completed if record["candidate_id"] == selected_id]
        if not matches:
            raise MIPROCompilationError("MIPRO best instruction lacks a complete selection evaluation", ledger.rows)
        chosen = matches[-1]
        selection_source = "official_miprov2_result"
    candidates = {}
    for record in completed:
        text = instruction(record["fragment"], record["demonstrations"])
        candidates.setdefault(record["candidate_id"], {"id": record["candidate_id"],
                              "instruction": text, "instruction_hash": _text_hash(text),
                              "demonstration_count": len(record["demonstrations"]),
                              "demonstration_case_ids": [
                                  train_by_input[demo["input"]]["id"]
                                  for demo in record["demonstrations"]]})
    failed = any(row["error"] is not None for row in ledger.rows)
    protocol = {
        "implementation": "official_dspy_miprov2", "dspy_version": DSPY_VERSION,
        "optuna_version": importlib.metadata.version("optuna"),
        "baseline_label": "MIPROv2 with synthetic data and local frozen-student proposal model",
        "published_result_reproduction": False,
        "adaptations": ["local frozen student is proposal model", "immutable common task/output contract",
                        "shared attempted-call cap", "fixed decoding from local model identity",
                        "teacher-generated unverified optimization references"],
        "remote_teacher_calls": 0, "benchmark_test_access": False, "synthetic_holdout_access": False,
        "shared_attempted_call_budget": config.max_calls,
        "unused_call_budget": config.max_calls - len(ledger.rows),
        "solver_calls": sum(row["role"] == "solver" for row in ledger.rows),
        "proposal_calls": sum(row["role"] == "optimizer" for row in ledger.rows),
        "native_bootstrapping_enabled": True,
        "final_max_bootstrapped_demos": max_bootstrapped_demos,
        "final_max_labeled_demos": max_labeled_demos, "num_candidates": config.candidates,
        "num_trials": trials, "minibatch": False, "num_threads": 1,
        "proposal_init_temperature": 0.0, "optimizer_decoding_overrides_honored": False,
        "decoding_policy": "All solver and proposer calls use the frozen local model identity; native temperature, rollout and max-token requests are recorded but not applied.",
        "program_aware_proposer": True, "data_aware_proposer": True,
        "fewshot_aware_proposer": True, "tip_aware_proposer": True,
        "selection_source": selection_source, "optimizer_returned": result is not None,
        "budget_stopped": budget_stopped, "candidate_rejections": rejections,
    }
    bundle = {
        "version": 1, "task_id": pack["task_id"], "method": "mipro_local",
        "mode": model.identity["mode"], "task_pack_hash": content_hash(pack),
        "description_hash": _text_hash(pack["description"]), "output_schema": output_schema,
        "config": asdict(config), "student_identity": dict(model.identity),
        "selected": candidates[chosen["candidate_id"]], "candidates": list(candidates.values()),
        "candidate_scores": [{"candidate_id": record["candidate_id"],
                               "selection_score" if task_adapter else "selection_exact_match": record["score"]}
                             for record in completed],
        "split_ids": {name: [case["id"] for case in rows] for name, rows in splits.items()},
        "generated_input_hashes": [normalized_input_hash(case["input"]) for case in pack["cases"]],
        "synthetic_holdout_hashes": {case["id"]: content_hash(case) for case in splits["synthetic_holdout"]},
        "usage": ledger.summary(),
        "compile_status": "budget_stopped" if budget_stopped else ("partial" if failed else "completed"),
        "novelty_status": "not_established", "semantic_equivalence": "not_proven",
        "search_reference_kind": "teacher_generated_unverified", "optimizer_protocol": protocol,
    }
    if task_adapter is not None:
        bundle["task_adapter"] = task_adapter.metadata()
        protocol["task_scorer_id"] = task_adapter.scorer_id
        protocol["adaptations"][-1] = "generated practice cases scored by the declared local task verifier"
    bundle["bundle_hash"] = content_hash(bundle)
    return {"bundle": bundle, "calls": ledger.rows}
