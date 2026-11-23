"""Prospectively locked inference ablations of retained original-framework artifacts.

This evaluation harness imports the original compiler, renderers, router, student,
and scoring code without editing them. Historical cases are reused deliberately;
this is an attribution study, not independent task-family confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "reports" / "reviewer1_20261120" / "original_ablation"
ARMS = ("raw_original", "raw_labels", "initial_prose", "initial_list",
        "final_prose", "final_list", "selected_full", "selected_routed")
MODELS = {"llama": "llama3.2:3b", "gemma": "gemma2:9b"}
DIGESTS = {
    "llama": "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
    "gemma": "ff02c3702f322b9e075e9568332d96c0a7028002f1a5a056e0a6784320a4db0b",
}
CONTRASTS = (
    ("raw_labels", "raw_original"),
    ("initial_list", "initial_prose"),
    ("final_list", "initial_list"),
    ("final_prose", "initial_prose"),
    ("final_list", "final_prose"),
    ("selected_routed", "selected_full"),
    ("selected_full", "final_list"),
    ("selected_routed", "raw_labels"),
    ("selected_routed", "initial_prose"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()


def read_rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def dependencies():
    from taskcompiler.rules_track.synthetic_policies import make_policy_task, _prompt_prefix
    from taskcompiler.dsl.parser import parse_program, render_program_source
    from taskcompiler.dsl.interpreter import run
    from taskcompiler.translate import (
        ProbeCase, render_program, compile_verified_decision_list,
        certify_verified_decision_list, normalise_student_label,
    )
    return locals()


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("A frozen plan already exists; do not overwrite it")
    d = dependencies()
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
               for p in sorted((ROOT / "src" / "taskcompiler").rglob("*.py"))}
    sources[str(Path(__file__).relative_to(ROOT)).replace("\\", "/")] = sha(__file__)
    bindings, tasks = {}, []
    for cohort, metadata in (("llama", "main_runs.json"), ("gemma", "gemma_runs.json")):
        metadata_path = ROOT / "reviewer_artifact" / "data" / metadata
        bindings[str(metadata_path.relative_to(ROOT)).replace("\\", "/")] = sha(metadata_path)
        rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected = [r for r in rows if r.get("condition") == "full" or r.get("status") == "syntax_failure"]
        for row in sorted(selected, key=lambda r: r["task"]):
            number = row["task"]
            task = d["make_policy_task"](number)
            artifact = None
            if row.get("run_id"):
                artifact_path = ROOT / "runs" / row["run_id"] / "artifacts" / f"policy-{number:03d}.json"
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                bindings[str(artifact_path.relative_to(ROOT)).replace("\\", "/")] = sha(artifact_path)
            approved = bool(artifact and artifact["teacher_compilation"]["validation_gate"]["passed"])
            initial, final = None, None
            representations = {}
            if artifact:
                for stage in ("initial", "final"):
                    program_path = artifact_path.with_name(f"policy-{number:03d}.teacher.{stage}.dsl")
                    bindings[str(program_path.relative_to(ROOT)).replace("\\", "/")] = sha(program_path)
                    program = d["parse_program"](program_path.read_text(encoding="utf-8"), task_spec=task.task_spec)
                    if stage == "initial":
                        initial = program
                    else:
                        final = program
                    probes = [d["ProbeCase"](f"build-{i:03d}", c.input_record, c.expected_label, ())
                              for i, c in enumerate(task.build_cases)]
                    lines = d["compile_verified_decision_list"](program, probes)
                    certificate = d["certify_verified_decision_list"](program, probes, lines)
                    if not certificate.get("semantically_equivalent") or certificate["finite_domain_size"] != 81:
                        raise RuntimeError(f"Rendering certificate failed: {cohort}/{number}/{stage}")
                    representations[stage + "_list"] = lines
                    representations[stage + "_prose"] = d["render_program"](program)
                if approved:
                    representations["selected"] = artifact["rendered_bundle"]
                    cert = d["certify_verified_decision_list"](final, probes, representations["selected"])
                    if not cert.get("semantically_equivalent"):
                        raise RuntimeError("Retained selected bundle failed equivalence replay")
            labels = ", ".join([*task.task_spec.labels, "uncertain"])
            contract = f"Output contract: return exactly one of {labels}. Do not include reasoning or formatting."
            raw_original = "\n\n".join([
                f"Task: {task.task_spec.description}",
                "Output contract: return exactly one declared label, or uncertain if the policy does not decide. Do not include reasoning or formatting."
            ])
            raw_labels = "\n\n".join([f"Task: {task.task_spec.description}", contract])
            description = (
                artifact["deployment_description"] if artifact else
                f"Classify one structured event using the supplied compiled instructions. The allowed outputs are {labels}. "
                "These names are arbitrary symbols, so do not infer their meaning. Copy the output stated by the matching compiled instruction."
            )
            systems = {"raw_original": raw_original, "raw_labels": raw_labels}
            for arm in ARMS[2:]:
                if not artifact or (arm.startswith(("final_", "selected_")) and not approved):
                    systems[arm] = raw_labels
                elif arm in representations:
                    systems[arm] = d["_prompt_prefix"](task, representations[arm], description)
                elif arm == "selected_full":
                    systems[arm] = d["_prompt_prefix"](task, representations["selected"], description)
                else:
                    systems[arm] = None
            for arm, system in systems.items():
                if arm != "raw_original" and system is not None and not system.endswith(contract):
                    raise RuntimeError("The allowed-label contract must be byte-identical")
            repairs = artifact["teacher_compilation"]["repairs"] if artifact else []
            cases = []
            for index, case in enumerate(task.final_cases):
                cases.append({"index": index, "record": case.input_record, "gold": case.expected_label})
            for subset, program in (("initial", initial), ("final", final)):
                if program is not None:
                    for case in cases:
                        case[subset + "_program_label"] = d["run"](program, case["record"]).output_label
            tasks.append({
                "cohort": cohort, "model": MODELS[cohort], "task_id": task.task_id, "number": number,
                "approved": approved, "syntax_failure": artifact is None,
                "labels": task.task_spec.labels, "description": description,
                "raw_labels": raw_labels, "systems": systems, "representations": representations,
                "cases": cases, "public_build": [asdict(c) for c in task.build_cases],
                "public_audit": [asdict(c) for c in task.audit_cases], "repair_records": repairs,
            })
    if Counter(t["cohort"] for t in tasks) != {"llama": 24, "gemma": 18}:
        raise RuntimeError("The full historical cohort must be retained")
    write(OUT / "prepared_inputs.json", tasks)
    protocol = {
        "id": "reviewer1-original-ablation-20261120-v1", "frozen_at_utc": now(),
        "question": "Separate output-label wording, program refinement, representation, and routing with fixed retained programs and unchanged framework code.",
        "scope": "New paired inference on historical development-supported task families; no independent-family confirmation.",
        "arms": ARMS, "cohorts": {"llama": 24, "gemma": 18}, "cases_per_task": 27,
        "expected_primary_logical_rows": 42 * 27 * len(ARMS),
        "contrasts": CONTRASTS, "bootstrap_unit": "whole task paired across arms",
        "bootstrap_resamples": 10000, "bootstrap_seed": 20261120,
        "arm_order": "per-case deterministic shuffle; tasks and models retain fixed order",
        "sentinels": "repeat every arm once on case zero of every task; repeats are diagnostics excluded from primary accuracy",
        "fallback": "Syntax failures use fresh label-matched raw responses in every instruction arm. Rejected final artifacts use label-matched raw fallback in final/selected arms; initial unverified controls retain the initial parsed program.",
        "rendering": "Original canonical prose and finite decision-list routines; initial lists are equivalent to unapproved P0, not independently verified correct policies.",
        "routing": "Original CertifiedInstructionRouter with top_k=6, no label encoding; MiniLM embeddings on CPU to leave GPU memory for the student.",
        "student": {"temperature": 0, "seed": 0, "context_tokens": 8192, "max_answer_tokens": 16, "thinking": False},
        "endpoint": "http://127.0.0.1:11434", "required_model_digests": DIGESTS,
        "cache": "Exact model/request reuse inside this new study; shared predictions remain logical paired observations, not independent calls.",
        "failures": "Charge started requests, no model retry, invalid/missing output is wrong; stop on five consecutive transport failures and report incomplete coverage.",
        "limits": {"max_physical_student_attempts": 10000, "max_wall_seconds": 7200},
        "stopping": "Complete all prespecified rows and sentinels or stop at the operational limit; never change arms or select tasks by outcome.",
        "source_sha256": sources, "retained_input_sha256": bindings,
        "prepared_inputs_sha256": sha(OUT / "prepared_inputs.json"),
        "interpretation_limits": [
            "One historical preparation per task and one new primary response per unique request.",
            "Repair includes accepted teacher revisions and existing automatic corrections; this is not a pure teacher-only causal effect.",
            "Fixed-bundle routing controls isolate delivery conditional on the retained chosen bundle.",
            "Intervals are descriptive and unadjusted for multiple exploratory contrasts.",
            "Direct execution remains an explicit control; no SLM necessity claim follows for canonical structured inputs.",
            "Runtime/version changes are disclosed; new rates do not replace historical rates."
        ],
    }
    write(OUT / "protocol.json", protocol)
    print(f"FROZEN {protocol['frozen_at_utc']}: {len(tasks)} tasks, {protocol['expected_primary_logical_rows']} primary rows", flush=True)


def verify_plan():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for group in ("source_sha256", "retained_input_sha256"):
        for relative, expected in protocol[group].items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError("Frozen bytes changed: " + relative)
    if sha(OUT / "prepared_inputs.json") != protocol["prepared_inputs_sha256"]:
        raise RuntimeError("Prepared input lock changed")
    return protocol


def run_study():
    from requests.adapters import HTTPAdapter
    import requests
    from taskcompiler.student import StudentModel
    from taskcompiler.route import CertifiedInstructionRouter
    from taskcompiler.translate import normalise_student_label

    protocol = verify_plan()
    tasks = json.loads((OUT / "prepared_inputs.json").read_text(encoding="utf-8"))
    if (OUT / "completion.json").exists():
        raise RuntimeError("Completed studies cannot be rerun")
    calls_path, predictions_path = OUT / "calls.jsonl", OUT / "predictions.jsonl"
    prior_calls = read_rows(calls_path)
    completed_ids = {r["call_id"] for r in prior_calls if r["event"] in ("completed", "failed")}
    if any(r["event"] == "started" and r["call_id"] not in completed_ids for r in prior_calls):
        raise RuntimeError("An unfinished request must be audited before resumption; no silent redispatch")
    cache = {r["request_hash"]: r for r in prior_calls if r["event"] in ("completed", "failed") and not r.get("sentinel")}
    finished = {(r["cohort"], r["task_id"], r["case_index"], r["arm"], r["sentinel"])
                for r in read_rows(predictions_path)}
    attempt = sum(r["event"] == "started" for r in prior_calls)
    started = time.monotonic()
    router = CertifiedInstructionRouter("sentence-transformers/all-MiniLM-L6-v2")
    router.retriever.to("cpu")
    write(OUT / "runtime.json", {"started_at_utc": now(), "python": sys.version,
          "ollama": requests.get(protocol["endpoint"] + "/api/version", timeout=10).json(),
          "router": router.identity, "router_device": "cpu", "pid": os.getpid()})
    consecutive_failures, logical = 0, len(finished)
    try:
        for cohort in MODELS:
            student = StudentModel(MODELS[cohort], backend="ollama", ollama_base_url=protocol["endpoint"],
                seed=0, context_tokens=8192, max_new_tokens=16, enable_thinking=False, ollama_timeout_seconds=120)
            for scheme in ("http://", "https://"):
                student._session.mount(scheme, HTTPAdapter(max_retries=0))
            if student.identity["digest"] != DIGESTS[cohort]:
                raise RuntimeError(f"Model identity mismatch for {cohort}: {student.identity['digest']}")
            write(OUT / f"identity_{cohort}.json", student.identity)
            for task in [t for t in tasks if t["cohort"] == cohort]:
                for case in task["cases"]:
                    arms = list(ARMS)
                    random.Random(f"20261120:{cohort}:{task['task_id']}:{case['index']}").shuffle(arms)
                    # Sentinels run after all primary cases in a task.
                    for arm in arms:
                        key = (cohort, task["task_id"], case["index"], arm, False)
                        if key in finished:
                            continue
                        system, routing = system_for(task, case, arm, router)
                        record = invoke(student, protocol, task, case, arm, system, routing, False,
                                        cache, calls_path, attempt, started, normalise_student_label)
                        attempt = record.pop("_attempt")
                        consecutive_failures = consecutive_failures + 1 if record["transport_failure"] else 0
                        append(predictions_path, record)
                        logical += 1
                        finished.add(key)
                        if consecutive_failures >= 5:
                            raise RuntimeError("Five consecutive transport failures")
                for arm in ARMS:
                    key = (cohort, task["task_id"], 0, arm, True)
                    if key in finished:
                        continue
                    case = task["cases"][0]
                    system, routing = system_for(task, case, arm, router)
                    record = invoke(student, protocol, task, case, arm, system, routing, True,
                                    cache, calls_path, attempt, started, normalise_student_label)
                    attempt = record.pop("_attempt")
                    append(predictions_path, record)
                    finished.add(key)
                progress = {"at_utc": now(), "cohort": cohort, "task_id": task["task_id"],
                            "primary_rows": logical, "physical_attempts": attempt,
                            "elapsed_seconds": time.monotonic() - started}
                write(OUT / "progress.json", progress)
                print(canonical(progress), flush=True)
        write(OUT / "completion.json", {"status": "complete", "at_utc": now(),
              "primary_rows": logical, "physical_attempts": attempt,
              "elapsed_seconds": time.monotonic() - started})
    except BaseException as exc:
        write(OUT / "interruption.json", {"status": "incomplete", "at_utc": now(),
              "error_type": type(exc).__name__, "error": str(exc), "physical_attempts": attempt,
              "primary_rows": logical, "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        report()


def system_for(task, case, arm, router):
    if task["systems"][arm] is not None:
        return task["systems"][arm], None
    from taskcompiler.rules_track.synthetic_policies import make_policy_task, _prompt_prefix
    routed = router.route(case["record"], task["representations"]["selected"], top_k=6, label_encoder=None)
    policy = make_policy_task(task["number"])
    return _prompt_prefix(policy, routed.instructions, task["description"]), routed.certificate


def invoke(student, protocol, task, case, arm, system, routing, sentinel,
           cache, calls_path, attempt, started, normalise):
    prompt = f"Input: {case['record']}"
    payload = {"model_identity": student.identity, "prompt": prompt, "system": system,
               "max_answer_tokens": 16}
    request_hash = hashlib.sha256(canonical(payload).encode()).hexdigest()
    cached = request_hash in cache and not sentinel
    if cached:
        result = cache[request_hash]
    else:
        if attempt >= protocol["limits"]["max_physical_student_attempts"] or time.monotonic()-started >= protocol["limits"]["max_wall_seconds"]:
            raise RuntimeError("Frozen resource limit reached")
        attempt += 1
        append(calls_path, {"event": "started", "call_id": attempt, "at_utc": now(),
               "request_hash": request_hash, "sentinel": sentinel, **payload})
        try:
            response = student.ask(prompt, system=system)
            result = {"event": "completed", "call_id": attempt, "at_utc": now(),
                      "request_hash": request_hash, "sentinel": sentinel, "response": asdict(response)}
        except Exception as exc:
            result = {"event": "failed", "call_id": attempt, "at_utc": now(),
                      "request_hash": request_hash, "sentinel": sentinel,
                      "error_type": type(exc).__name__, "error": str(exc)}
        append(calls_path, result)
        if not sentinel:
            cache[request_hash] = result
    failed = result["event"] == "failed"
    response = result.get("response", {})
    raw = response.get("text", "")
    predicted = "__INVALID__:transport_failure" if failed else normalise(raw, set(task["labels"]))
    return {
        "cohort": task["cohort"], "model": task["model"], "task_id": task["task_id"],
        "case_index": case["index"], "arm": arm, "sentinel": sentinel,
        "gold": case["gold"], "predicted": predicted, "raw_output": raw,
        "correct": predicted == case["gold"], "invalid": isinstance(predicted, str) and predicted.startswith("__INVALID__"),
        "call_id": result["call_id"], "request_hash": request_hash, "cached": cached,
        "transport_failure": failed, "model_latency_seconds": response.get("latency_seconds"),
        "input_tokens": response.get("input_tokens"), "output_tokens": response.get("output_tokens"),
        "approved": task["approved"], "syntax_failure": task["syntax_failure"],
        "routing_certificate": routing, "_attempt": attempt,
    }


def report():
    import numpy as np
    protocol = verify_plan()
    rows = read_rows(OUT / "predictions.jsonl")
    primary = [r for r in rows if not r["sentinel"]]
    if not primary:
        return
    result = {"generated_at_utc": now(), "status": "complete" if (OUT/"completion.json").exists() else "incomplete",
              "scope": protocol["scope"], "cohorts": {}, "new_model_calls_by_report": 0}
    for cohort in MODELS:
        group = [r for r in primary if r["cohort"] == cohort]
        if not group:
            continue
        conditions = {}
        for arm in ARMS:
            cells = [r for r in group if r["arm"] == arm]
            conditions[arm] = {"correct": sum(r["correct"] for r in cells), "total": len(cells),
                "accuracy": sum(r["correct"] for r in cells)/len(cells) if cells else None,
                "invalid": sum(r["invalid"] for r in cells),
                "cached_rows": sum(r["cached"] for r in cells)}
        task_means = {}
        for r in group:
            task_means.setdefault(r["task_id"], {}).setdefault(r["arm"], []).append(r["correct"])
        contrasts = {}
        for a,b in CONTRASTS:
            paired = [(tid, float(np.mean(v[a]))-float(np.mean(v[b]))) for tid,v in task_means.items()
                      if len(v.get(a,[]))==27 and len(v.get(b,[]))==27]
            if paired:
                diff=np.array([v for _,v in paired])
                rng=np.random.default_rng(20261120)
                bootstrap=rng.choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
                contrasts[a+" minus "+b]={"tasks":len(diff),"mean_difference":float(diff.mean()),
                    "descriptive_task_bootstrap_95":np.quantile(bootstrap,[.025,.975]).tolist(),
                    "task_differences":dict(paired)}
        repeat=[r for r in rows if r["cohort"]==cohort and r["sentinel"]]
        originals={(r["task_id"],r["arm"]):r for r in group if r["case_index"]==0}
        result["cohorts"][cohort]={"conditions":conditions,"contrasts":contrasts,
            "sentinel_repeats":len(repeat),
            "sentinel_label_changes":sum(r["predicted"]!=originals[(r["task_id"],r["arm"])]["predicted"] for r in repeat),
            "per_task":{tid:{arm:float(np.mean(vs)) for arm,vs in values.items()} for tid,values in task_means.items()}}
    calls=read_rows(OUT/"calls.jsonl")
    result["physical_attempts"]=sum(r["event"]=="started" for r in calls)
    result["physical_failures"]=sum(r["event"]=="failed" for r in calls)
    write(OUT/"summary.json",result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","run","report"))
    action=parser.parse_args().action
    {"prepare":prepare,"run":run_study,"report":report}[action]()
