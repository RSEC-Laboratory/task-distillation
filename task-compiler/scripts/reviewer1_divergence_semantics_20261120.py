"""Replay initial-program disagreement from the retained matched preparations."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from taskcompiler.dsl.parser import parse_program
from taskcompiler.dsl.interpreter import run
from taskcompiler.rules_track.synthetic_policies import make_policy_task

BASE=ROOT/"reports/reviewer1_20261120/retained_audit"
results=[]
for number in range(8):
    task=make_policy_task(number)
    records=[c.input_record for c in task.build_cases+task.audit_cases+task.final_cases]
    assert len({json.dumps(r,sort_keys=True) for r in records})==81
    programs={};variants={}
    for model in ("qwen2.5_3b","llama3.2_3b"):
        path=ROOT/f"reports/matched_original/development_20261006_v1/prepared/{model}-policy-{number:03d}-full.json"
        data=json.loads(path.read_text())
        method=data["method"]
        source=method["trace"].get("dsl",{}).get("initial_source")
        program=parse_program(source,task_spec=task.task_spec) if source else None
        programs[model]=program
        variants[model]={"path":path.relative_to(ROOT).as_posix(),
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "initial_program_available":program is not None,"final_status":method["status"]}
    left,right=programs.values()
    disagreements=sum(run(left,r).output_label!=run(right,r).output_label for r in records) if left is not None and right is not None else None
    results.append({"task_id":task.task_id,"domain_size":81,
        "initial_program_output_disagreements":disagreements,"models":variants})
original=json.loads((BASE/"teacher_program_divergence.json").read_text())
assert {r["task_id"]:r["initial_program_output_disagreements"] for r in results}=={r["task_id"]:r["initial_program_output_disagreements"] for r in original["tasks"]}
for row in results:
    prior=next(r for r in original["tasks"] if r["task_id"]==row["task_id"])
    for model,v in row["models"].items():
        assert v["initial_program_available"]==prior["models"][model]["initial_program_available"]
report={"status":"passed","new_model_calls":0,"reference_labels_used":False,"tasks":results}
(BASE/"teacher_program_divergence_replay.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"status":"passed","tasks":len(results),"policy_001_disagreements":results[1]["initial_program_output_disagreements"]}))
