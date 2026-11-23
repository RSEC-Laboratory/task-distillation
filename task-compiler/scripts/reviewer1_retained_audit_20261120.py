"""Read-only replay of retained compilation calls, repair gates, and fallbacks."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taskcompiler.dsl.parser import parse_program, render_program_source
from taskcompiler.dsl.interpreter import run
from taskcompiler.rules_track.synthetic_policies import make_policy_task

OUT = ROOT / "reports/reviewer1_20261120/retained_audit"
OUT.mkdir(parents=True, exist_ok=True)


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def readrows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]


def main():
    matched = ROOT / "reports/matched_original/development_20261006_v1"
    comparisons = []
    for number in range(8):
        pair = {}
        for model in ("qwen2.5_3b", "llama3.2_3b"):
            rows = readrows(matched / f"calls/{model}-policy-{number:03d}-full.jsonl")
            calls = [r for r in rows if r.get("role") == "teacher_optimization"]
            request = next(r["body"] for r in calls if r["event"] == "transport_request")
            response = next(r["payload"]["message"]["content"] for r in calls if r["event"] == "transport_response")
            pair[model] = {"initial_request_sha256": digest(request),
                           "initial_response_sha256": digest(response)}
        comparisons.append({"task": f"policy-{number:03d}", "models": pair,
            "identical_initial_request": len({r["initial_request_sha256"] for r in pair.values()}) == 1,
            "identical_initial_response": len({r["initial_response_sha256"] for r in pair.values()}) == 1})
    summary = json.loads((matched / "summary.json").read_text())
    fallback = {}
    for model in summary["models"]:
        rows = [r for r in summary["tasks"] if r["solver_model"] == model["solver_model"]]
        unavailable = [r for r in rows if r["deployment_status"]["full"] == "raw_fallback"]
        correct = model["correct"]["direct_dsl"] + sum(r["correct"]["raw"] for r in unavailable)
        fallback[model["solver_model"]] = {"correct": correct, "total": 216,
            "accuracy": correct / 216, "fallback_tasks": len(unavailable)}
    prepared = json.loads((ROOT / "reports/reviewer1_20261120/original_ablation/prepared_inputs.json").read_text())
    repair_rows, counts, program_scores = [], Counter(), {}
    for t in prepared:
        if t["syntax_failure"]:
            continue
        task = make_policy_task(t["number"])
        cases = task.build_cases + task.audit_cases
        cohort = t["cohort"]
        scores = program_scores.setdefault(cohort, {"initial_correct": 0, "final_correct": 0, "total": 0})
        for c in t["cases"]:
            scores["initial_correct"] += c["initial_program_label"] == c["gold"]
            scores["final_correct"] += c["final_program_label"] == c["gold"]
            scores["total"] += 1
        for index, repair in enumerate(t["repair_records"]):
            if not repair["accepted"]:
                counts[(cohort, "teacher_rejected")] += 1
                continue
            before = parse_program(repair["original_source"], task_spec=task.task_spec)
            after = parse_program(repair["proposed_source"], task_spec=task.task_spec)
            old = {i for i, c in enumerate(cases) if run(before, c.input_record).output_label == c.expected_label}
            new = {i for i, c in enumerate(cases) if run(after, c.input_record).output_label == c.expected_label}
            if not old < new:
                raise RuntimeError(f"Non-regression replay failed for {cohort}/{task.task_id}/{index}")
            name = repair["rule_name"]
            kind = ("teacher" if name == "<complete-program>" else
                    "local_exclusion" if name.startswith("local-exclusion:") else "exact_counterexample_patch")
            counts[(cohort, kind)] += 1
            old_rules = {r.name: r for r in before.rules}
            new_rules = {r.name: r for r in after.rules}
            changed = [n for n in old_rules.keys() & new_rules.keys()
                       if canonical(str(old_rules[n].condition)) != canonical(str(new_rules[n].condition))]
            repair_rows.append({"cohort": cohort, "task_id": task.task_id, "index": index,
                "kind": kind, "name": name, "correct_public_before": len(old), "correct_public_after": len(new),
                "strict_nonregression_verified": True,
                "existing_rule_conditions_changed": sorted(changed),
                "new_rule_names": sorted(new_rules.keys() - old_rules.keys())})
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_model_calls": 0, "scope": "Retrospective retained-record replay, not a new empirical sample.",
        "initial_teacher_request_pairs": comparisons,
        "identical_request_pairs": sum(r["identical_initial_request"] for r in comparisons),
        "different_response_pairs_under_identical_requests": sum(
            r["identical_initial_request"] and not r["identical_initial_response"] for r in comparisons),
        "fallback_matched_direct_execution": fallback,
        "accepted_repairs": repair_rows,
        "repair_counts": [{"cohort": c, "kind": k, "count": v} for (c, k), v in sorted(counts.items())],
        "program_only_scores_all_parsed_candidates": program_scores,
        "limits": ["Observed response variation is not an identified hardware/kernel cause.",
            "Program-only scores include unapproved candidates and exclude the two missing Gemma programs.",
            "Existing local-exclusion repairs are distinct from the terminal exact-record insertion operator.",
            "Public non-regression does not imply unseen-input correctness."]}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["identical_request_pairs",
        "different_response_pairs_under_identical_requests", "repair_counts",
        "fallback_matched_direct_execution", "program_only_scores_all_parsed_candidates"]}, indent=2))


if __name__ == "__main__":
    main()
