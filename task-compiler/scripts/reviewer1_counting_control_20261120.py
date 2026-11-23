"""Frozen two-guidance by checker-control study; no optimizer or core changes."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "reports/reviewer1_20261120/counting_control"
EXPECTED_DIGEST = "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72"


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append(p, v):
    with Path(p).open("a", encoding="utf-8") as f:
        f.write(canonical(v) + "\n")
        f.flush()


def prepare():
    from taskcompiler.guidance_track.bbh import (
        _render_generic_scratchpad, _render_scratchpad_lineage_guidance,
        _verified_scratchpad_prompt, TASK_DESCRIPTIONS,
    )
    if (OUT / "protocol.json").exists():
        raise RuntimeError("Do not overwrite a frozen counting protocol")
    parent = ROOT / "runs/2026-07-27_005006_bbh_bbh_counting_llama32_3b_verified_parent70_test100/artifacts"
    tg = ROOT / "runs/2026-07-27_012913_bbh_bbh_counting_llama32_3b_textgrad_tgd_test100/artifacts"
    guidance_path = parent / "offline_guidance.txt"
    history_path = tg / "textgrad_history.json"
    cases_path = parent / "predictions.jsonl"
    guidance = json.loads(guidance_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    accepted = [r for r in history["history"] if r["accepted"]]
    prompt = accepted[-1]["prompt"]
    assert accepted[-1]["validation_accuracy"] == history["protocol"]["best_validation_accuracy"]
    description = TASK_DESCRIPTIONS["object_counting"]
    shared = _render_generic_scratchpad("object_counting", description)
    teacher_system = _render_scratchpad_lineage_guidance("object_counting", description, guidance)
    systems = {method: "Method-specific task guidance:\n" + specific +
               "\n\nShared execution and output contract:\n" + shared
               for method, specific in (("counting_guidance", teacher_system), ("textgrad_guidance", prompt))}
    cases = [json.loads(s) for s in cases_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    tg_cases = [json.loads(s) for s in (tg/"predictions.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    assert len(cases) == 100
    assert [(c["item_id"], c["gold"]) for c in cases] == [(c["item_id"], c["gold"]) for c in tg_cases]
    prepared = {"systems": systems, "shared_contract": shared,
        "textgrad_selected_step": accepted[-1]["step"],
        "cases": [{"item_id": c["item_id"], "gold": c["gold"],
            "prompt": _verified_scratchpad_prompt("object_counting", c["question"], parent_transferred=False)}
            for c in cases]}
    write(OUT/"prepared_inputs.json", prepared)
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in [guidance_path, history_path, cases_path, tg/"predictions.jsonl"]}
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT/"src/taskcompiler").rglob("*.py")}
    sources[Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
    protocol = {"id": "reviewer1-counting-checker-control-20261120-v1", "frozen_at_utc": now(),
        "question": "Compare frozen counting guidance with frozen locally optimized TextGrad guidance under a shared ledger format and identical checker/correction allowance.",
        "scope": "One historically examined task and 100 reused cases; descriptive inference control, not fresh confirmation or a rerun of TextGrad optimization.",
        "methods": list(systems), "checker_conditions": ["off", "on"], "cases": 100,
        "initial_response_reuse": "Each method/case initial response supplies both the off outcome and the on condition before any correction.",
        "checker": "Unchanged _verified_scratchpad_answer; correction only when a parsed ledger sum exists and differs from the initial parsed answer.",
        "correction": "The same original one-call correction message and system instruction for both methods; no reference answer enters this branch.",
        "format_control": "Both frozen method-specific instructions receive the identical existing generic ledger execution contract as the final instruction block.",
        "ordering": "Fixed cases; method order shuffled per case using seed 20261120.",
        "student": {"model": "llama3.2:3b", "digest": EXPECTED_DIGEST, "temperature": 0,
                    "seed": 0, "context_tokens": 8192, "max_answer_tokens": 256, "thinking": False},
        "limits": {"max_physical_calls": 400, "max_wall_seconds": 1800,
                   "per_method_case_calls": 2, "fresh_teacher_calls": 0},
        "analysis": "Paired correct counts, checker coverage, correction regressions, actual calls/tokens, and descriptive paired-example bootstrap intervals (10000, seed 20261120). No confirmatory p-value.",
        "stopping": "Complete all cases or stop at an operational/resource limit; retain failures and never retune or replace prompts based on measured outcomes.",
        "source_sha256": sources, "input_sha256": bindings,
        "prepared_inputs_sha256": sha(OUT/"prepared_inputs.json")}
    write(OUT/"protocol.json", protocol)
    print("FROZEN", protocol["id"], protocol["frozen_at_utc"], flush=True)


def verify():
    p = json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    for field in ("source_sha256", "input_sha256"):
        for relative, expected in p[field].items():
            assert sha(ROOT/relative) == expected, relative
    assert sha(OUT/"prepared_inputs.json") == p["prepared_inputs_sha256"]
    return p


def run():
    from requests.adapters import HTTPAdapter
    from taskcompiler.student import StudentModel
    from taskcompiler.guidance_track.bbh import parse_answer, _verified_scratchpad_answer
    p = verify()
    if (OUT/"calls.jsonl").exists():
        raise RuntimeError("No automatic rerun or redispatch of an existing counting study")
    data = json.loads((OUT/"prepared_inputs.json").read_text(encoding="utf-8"))
    model = StudentModel("llama3.2:3b", backend="ollama", ollama_base_url="http://127.0.0.1:11434",
        context_tokens=8192, max_new_tokens=256, seed=0, enable_thinking=False, ollama_timeout_seconds=120)
    assert model.identity["digest"] == EXPECTED_DIGEST
    for scheme in ("http://", "https://"):
        model._session.mount(scheme, HTTPAdapter(max_retries=0))
    write(OUT/"identity.json", model.identity)
    start, attempts = time.monotonic(), 0

    def ask(prompt, system, phase, method, item):
        nonlocal attempts
        if attempts >= 400 or time.monotonic()-start > 1800:
            raise RuntimeError("Frozen counting resource limit")
        attempts += 1
        append(OUT/"calls.jsonl", {"event": "started", "call_id": attempts,
            "at_utc": now(), "phase": phase, "method": method, "item_id": item,
            "prompt": prompt, "system": system, "model_identity": model.identity})
        try:
            response = model.ask(prompt, system=system)
        except Exception as exc:
            append(OUT/"calls.jsonl", {"event": "failed", "call_id": attempts,
                   "error_type": type(exc).__name__, "error": str(exc), "at_utc": now()})
            return None
        append(OUT/"calls.jsonl", {"event": "completed", "call_id": attempts,
               "response": asdict(response), "at_utc": now()})
        return response
    try:
        for index, case in enumerate(data["cases"]):
            methods = list(data["systems"])
            random.Random(f"20261120:{index}").shuffle(methods)
            for method in methods:
                response = ask(case["prompt"], data["systems"][method], "initial", method, case["item_id"])
                initial = parse_answer("object_counting", response.text) if response else None
                ledger = _verified_scratchpad_answer("object_counting", response.text) if response else None
                final = initial
                corrected = ledger is not None and ledger != initial
                if corrected:
                    correction = ask("A local verifier checked your structured trace " +
                        f"and obtained {ledger}. Return exactly " + f"\x60Answer: {ledger}\x60.",
                        "Return the locally verified answer exactly as requested. Do not add reasoning.",
                        "correction", method, case["item_id"])
                    final = parse_answer("object_counting", correction.text) if correction else None
                append(OUT/"predictions.jsonl", {"item_id": case["item_id"], "method": method,
                    "gold": case["gold"], "initial": initial, "checked": final,
                    "initial_correct": initial == case["gold"], "checked_correct": final == case["gold"],
                    "ledger": ledger, "correction_used": corrected})
            if (index+1)%10 == 0:
                progress={"at_utc":now(),"cases_completed":index+1,"physical_calls":attempts,
                          "elapsed_seconds":time.monotonic()-start}
                write(OUT/"progress.json",progress);print(canonical(progress),flush=True)
        write(OUT/"completion.json",{"status":"complete","at_utc":now(),"physical_calls":attempts,
                                    "elapsed_seconds":time.monotonic()-start})
    except BaseException as exc:
        write(OUT/"interruption.json",{"status":"incomplete","error_type":type(exc).__name__,
            "error":str(exc),"at_utc":now(),"physical_calls":attempts})
        raise


def queued():
    path = ROOT/"reports/reviewer1_20261120/queue/status.json"
    start = time.monotonic()
    while True:
        state=json.loads(path.read_text()) if path.exists() else {}
        if state.get("state") in {"finished","supervisor_failed","reserved_operational_stop"}:
            break
        if time.monotonic()-start > 5*3600:
            raise RuntimeError("Sequential experiment wait exceeded five hours")
        print(canonical({"at_utc":now(),"state":"waiting_for_prior_studies","prior":state.get("state")}),flush=True)
        time.sleep(45)
    run()


if __name__ == "__main__":
    action = sys.argv[1]
    {"prepare": prepare, "run": run, "queued": queued}[action]()
