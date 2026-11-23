"""Rescore the completed reviewer studies from recorded responses, without calls."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
BASE=ROOT/"reports/reviewer1_20261120"


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(protocol):
    for field in ("source_sha256","input_sha256","retained_input_sha256","config_sha256"):
        for path,expected in protocol.get(field,{}).items():
            assert sha(ROOT/path)==expected,path


def original():
    import numpy as np
    from taskcompiler.translate import normalise_student_label
    p=BASE/"original_ablation"
    protocol=json.loads((p/"protocol.json").read_text(encoding='utf-8'))
    verify_sources(protocol)
    assert sha(p/"prepared_inputs.json")==protocol["prepared_inputs_sha256"]
    assert (p/"completion.json").exists(),"Original ablation is not complete"
    prepared=json.loads((p/"prepared_inputs.json").read_text(encoding='utf-8'))
    predicted=rows(p/"predictions.jsonl")
    calls=rows(p/"calls.jsonl")
    starts={r["call_id"]:r for r in calls if r["event"]=="started"}
    results={r["call_id"]:r for r in calls if r["event"] in ("completed","failed")}
    assert len(starts)==sum(r["event"]=="started" for r in calls)
    assert len(results)==sum(r["event"] in ("completed","failed") for r in calls)
    assert starts.keys()==results.keys(),"Unresolved or uncharged physical call"
    assert min(r["at_utc"] for r in starts.values())>protocol["frozen_at_utc"]
    task_map={(t["cohort"],t["task_id"]):t for t in prepared}
    seen=set()
    for r in predicted:
        key=(r["cohort"],r["task_id"],r["case_index"],r["arm"],r["sentinel"])
        assert key not in seen,key
        seen.add(key)
        task=task_map[(r["cohort"],r["task_id"])]
        case=task["cases"][r["case_index"]]
        assert case["gold"]==r["gold"]
        response=results[r["call_id"]]
        request=starts[r["call_id"]]
        assert r["request_hash"]==request["request_hash"]==response["request_hash"]
        assert request["prompt"]==f"Input: {case['record']}"
        assert request["model_identity"]["digest"]==protocol["required_model_digests"][r["cohort"]]
        if r["arm"]!="raw_original":
            assert request["system"].endswith(task["raw_labels"].rsplit("\n\n",1)[1])
        raw=response.get("response",{}).get("text","")
        label=("__INVALID__:transport_failure" if response["event"]=="failed" else
               normalise_student_label(raw,set(task["labels"])))
        assert raw==r["raw_output"] and label==r["predicted"]
        assert r["correct"]==(label==case["gold"])
    primary=[r for r in predicted if not r["sentinel"]]
    assert len(primary)==protocol["expected_primary_logical_rows"]
    assert len(predicted)-len(primary)==len(prepared)*len(protocol["arms"])
    for t in prepared:
        for arm in protocol["arms"]:
            assert sum(r["cohort"]==t["cohort"] and r["task_id"]==t["task_id"] and r["arm"]==arm for r in primary)==27
    controls={}
    for cohort in ("llama","gemma"):
        group=[t for t in prepared if t["cohort"]==cohort]
        raw={(r["task_id"],r["case_index"]):r for r in primary if r["cohort"]==cohort and r["arm"]=="raw_labels"}
        correct,total,per_task=0,0,{}
        for t in group:
            task_correct=0
            for case in t["cases"]:
                label=case.get("final_program_label") if t["approved"] else raw[(t["task_id"],case["index"])]["predicted"]
                task_correct+=label==case["gold"]
            per_task[t["task_id"]]=task_correct/27
            correct+=task_correct
            total+=27
        controls[cohort]={"correct":correct,"total":total,"accuracy":correct/total,
            "per_task":per_task,"approved_tasks":sum(t["approved"] for t in group),
            "fallback_tasks":sum(not t["approved"] for t in group)}
    approved_diagnostics={}
    for cohort in ("llama","gemma"):
        subset=[r for r in primary if r["cohort"]==cohort and r["approved"]]
        task_ids=sorted({r["task_id"] for r in subset})
        conditions={arm:{"correct":sum(r["correct"] for r in subset if r["arm"]==arm),
                         "total":sum(r["arm"]==arm for r in subset)} for arm in protocol["arms"]}
        contrasts={}
        for a,b in protocol["contrasts"]:
            dif=np.array([(sum(r["correct"] for r in subset if r["task_id"]==tid and r["arm"]==a)-
                           sum(r["correct"] for r in subset if r["task_id"]==tid and r["arm"]==b))/27 for tid in task_ids])
            rng=np.random.default_rng(20261120)
            draws=rng.choice(dif,(10000,len(dif))).mean(axis=1)
            contrasts[a+" minus "+b]={"difference":float(dif.mean()),"descriptive_task_bootstrap_95":np.quantile(draws,[.025,.975]).tolist()}
        approved_diagnostics[cohort]={"tasks":len(task_ids),"conditions":conditions,"contrasts":contrasts,
            "scope":"Conditional diagnostic on historically approved artifacts. All-task contrasts remain primary and include every failure/fallback."}
    result={"status":"passed","primary_rows":len(primary),"sentinel_rows":len(predicted)-len(primary),
        "physical_attempts":len(starts),"physical_failures":sum(r["event"]=="failed" for r in results.values()),
        "direct_teacher_program_with_identical_fresh_raw_fallback":controls,
        "approved_only_diagnostics":approved_diagnostics,
        "known_input_tokens":sum(r.get("response",{}).get("input_tokens",0) for r in results.values()),
        "known_output_tokens":sum(r.get("response",{}).get("output_tokens",0) for r in results.values()),
        "sum_model_response_seconds":sum(r.get("response",{}).get("latency_seconds",0) for r in results.values())}
    write(p/"verification.json",result)
    return result


def reserved(schema=False):
    import numpy as np
    from taskcompiler.guidance_track.superni import load_task,response_matches,CONDITIONS
    transport=ROOT/"protocols/reviewer1_reserved_schema_superni_20261120_v2_transport.json"
    p=(transport if schema and transport.exists() else ROOT/("protocols/reviewer1_reserved_schema_superni_20261120_v1.json" if schema else "protocols/reviewer1_reserved_superni_20261120_v1.json"))
    protocol=json.loads(p.read_text(encoding='utf-8'))
    verify_sources(protocol)
    pattern=("*reviewer1_reserved20_schema70b_transport1_llama_20261120" if schema and transport.exists() else "*reviewer1_reserved20_schema70b_llama_20261120" if schema else "*reviewer1_reserved20_llama_20261120")
    runs=list((ROOT/"runs").glob(pattern))
    assert len(runs)==1,("Expected one charged run",runs)
    run=runs[0]
    if schema and transport.exists() and not (run/"artifacts/predictions.jsonl").exists():
        assert json.loads((BASE/"queue_schema_retry/status.json").read_text(encoding="utf-8"))["exit_code"]==1
        first=ROOT/protocol["transport_amendment"]["first_run"]
        records=[];requests=[]
        for current in (first,run):
            log=rows(current/"artifacts/model_calls.jsonl")
            starts=[r for r in log if r["event"]=="started"]
            ends=[r for r in log if r["event"]=="failed"]
            assert len(starts)==len(ends)==1 and starts[0]["role"]=="teacher"
            assert starts[0]["call_id"]==ends[0]["call_id"] and "429" in ends[0]["error"]
            assert not any(r["event"]=="completed" or r["role"]=="student" for r in log)
            assert not (current/"artifacts/predictions.jsonl").exists()
            requests.append(starts[0]["request"])
            records.append({"run":current.relative_to(ROOT).as_posix(),"failed_teacher_attempts":1,"successful_model_responses":0,"final_observations":0,"error":"HTTP 429","failed_usage":"unknown"})
        assert requests[0]==requests[1]
        assert sha(first/"artifacts/model_calls.jsonl")==protocol["transport_amendment"]["first_call_log_sha256"]
        result={"status":"unavailable","audit_passed":True,"reason":"HTTP 429 before the first teacher response on both the original attempt and the single documented cooldown restart.","attempts":records,
            "total_failed_teacher_attempts":2,"successful_model_responses":0,"final_observations":0,"accuracy":None,"identical_teacher_request_content":True,"scope":protocol["scope"]}
        write(BASE/"reserved_schema_verification.json",result)
        return result
    predictions=rows(run/"artifacts/predictions.jsonl")
    # A complete grid must exist; an interrupted run is never assigned a full accuracy.
    assert len(predictions)==2000,("Reserved grid incomplete",len(predictions))
    tasks={t["task_id"]:load_task(t["task_id"]) for t in protocol["tasks"]}
    refs={}
    for tid,task in tasks.items():
        for example in task.instances[:20]:
            refs[(tid,example.item_id)]=example.outputs
    # Keys are checked explicitly below; the frozen implementation determines
    # the field names recorded in the public prediction format.
    first=predictions[0]
    if "condition" not in first or "response" not in first:
        raise RuntimeError("Inspect actual frozen prediction schema before interpreting it: "+str(list(first)))
    seen=set();counts=defaultdict(lambda:[0,0])
    for r in predictions:
        key=(r["task_id"],r["item_id"],r["condition"])
        assert key not in seen
        seen.add(key)
        assert r["condition"] in CONDITIONS
        expected=refs[(r["task_id"],r["item_id"])]
        scored=response_matches(r["response"],expected)
        assert scored==r["correct"]
        counts[r["condition"]][0]+=scored
        counts[r["condition"]][1]+=1
    assert all(v[1]==400 for v in counts.values()) and len(counts)==5
    calls=rows(run/"artifacts/model_calls.jsonl")
    starts={r["call_id"]:r for r in calls if r["event"]=="started"}
    ends={r["call_id"]:r for r in calls if r["event"] in ("completed","failed")}
    assert len(starts)==sum(r["event"]=="started" for r in calls)
    assert len(ends)==sum(r["event"] in ("completed","failed") for r in calls)
    assert starts.keys()==ends.keys()
    assert min(r["at_utc"] for r in starts.values())>protocol["frozen_at_utc"]
    role_counts=Counter(r["role"] for r in starts.values())
    assert role_counts["teacher"]<=60 and role_counts["student"]<=2600
    final_physical=sum(not r["response_reused_for_identical_prompt"] for r in predictions)
    assert final_physical<=role_counts["student"]
    log=(run/"log.txt").read_text(encoding="utf-8")
    assert log.index("stage finished: compile_validate_and_freeze_all_tasks")<log.index("stage started: evaluate_frozen_official_instances")
    artifacts={}
    for tid in tasks:
        artifact=json.loads((run/"artifacts"/(tid+".json")).read_text(encoding="utf-8"))
        digest=artifact.pop("artifact_sha256")
        assert hashlib.sha256(json.dumps(artifact,sort_keys=True).encode("utf-8")).hexdigest()==digest
        assert artifact["parent_received_instances"] is False
        assert artifact["student_validation_used_instances"] is False
        artifacts[tid]={"has_final_representation":artifact["final_representation"] is not None,
                        "failure":artifact["failure"],"teacher_calls":artifact["teacher_calls"]}
    task_means={tid:{condition:sum(r["correct"] for r in predictions if r["task_id"]==tid and r["condition"]==condition)/20
                     for condition in CONDITIONS} for tid in tasks}
    contrasts={}
    for comparator in ("raw_definition","positive_examples","parent_one_shot","validated_full"):
        differences=np.array([v["validated_routed"]-v[comparator] for v in task_means.values()])
        rng=np.random.default_rng(20260727)
        draws=rng.choice(differences,(100000,len(differences))).mean(axis=1)
        contrasts["validated_routed minus "+comparator]={"mean_difference":float(differences.mean()),
            "descriptive_task_bootstrap_95":np.quantile(draws,[.025,.975]).tolist()}
    result={"status":"passed","run":run.relative_to(ROOT).as_posix(),"final_logical_rows":len(predictions),
        "conditions":{k:{"correct":v[0],"total":v[1],"accuracy":v[0]/v[1]} for k,v in counts.items()},
        "attempts_by_role":dict(role_counts),"physical_failures":sum(r["event"]=="failed" for r in ends.values()),
        "physical_final_student_calls":final_physical,"public_validation_student_calls":role_counts["student"]-final_physical,
        "all_artifacts_frozen_before_final_evaluation":True,"per_task":task_means,"artifacts":artifacts,"contrasts":contrasts,
        "scope":protocol["scope"]}
    write(BASE/("reserved_schema_verification.json" if schema else "reserved_verification.json"),result)
    return result


def counting():
    import numpy as np
    from taskcompiler.guidance_track.bbh import parse_answer,_verified_scratchpad_answer
    p=BASE/"counting_control"
    protocol=json.loads((p/"protocol.json").read_text(encoding='utf-8'))
    verify_sources(protocol)
    assert sha(p/"prepared_inputs.json")==protocol["prepared_inputs_sha256"]
    assert (p/"completion.json").exists(),"Counting control incomplete"
    prepared=json.loads((p/"prepared_inputs.json").read_text(encoding='utf-8'))
    cases={c["item_id"]:c for c in prepared["cases"]}
    predictions=rows(p/"predictions.jsonl")
    assert len(predictions)==200
    calls=rows(p/"calls.jsonl")
    starts={r["call_id"]:r for r in calls if r["event"]=="started"}
    ends={r["call_id"]:r for r in calls if r["event"] in ("completed","failed")}
    assert len(starts)==sum(r["event"]=="started" for r in calls)
    assert len(ends)==sum(r["event"] in ("completed","failed") for r in calls)
    assert starts.keys()==ends.keys()
    assert len(starts)<=400
    assert min(r["at_utc"] for r in starts.values())>protocol["frozen_at_utc"]
    indexed={(r["method"],r["item_id"],r["phase"]):ends[cid] for cid,r in starts.items()}
    requests={(r["method"],r["item_id"],r["phase"]):r for r in starts.values()}
    assert len(indexed)==len(starts)
    seen=set()
    for r in predictions:
        key=(r["method"],r["item_id"])
        assert key not in seen
        seen.add(key)
        initial=indexed[(*key,"initial")]
        initial_request=requests[(*key,"initial")]
        assert initial_request["prompt"]==cases[r["item_id"]]["prompt"]
        assert initial_request["system"]==prepared["systems"][r["method"]]
        text=initial.get("response",{}).get("text")
        parsed=parse_answer("object_counting",text) if text is not None else None
        ledger=_verified_scratchpad_answer("object_counting",text) if text is not None else None
        assert r["initial"]==parsed and r["ledger"]==ledger
        if ledger is not None and ledger!=parsed:
            correction=indexed[(*key,"correction")]
            correction_request=requests[(*key,"correction")]
            assert correction_request["prompt"]=="A local verifier checked your structured trace "+f"and obtained {ledger}. Return exactly "+f"`Answer: {ledger}`."
            assert correction_request["system"]=="Return the locally verified answer exactly as requested. Do not add reasoning."
            finaltext=correction.get("response",{}).get("text")
            final=parse_answer("object_counting",finaltext) if finaltext is not None else None
        else:
            assert (*key,"correction") not in indexed
            final=parsed
        assert r["checked"]==final
        assert r["gold"]==cases[r["item_id"]]["gold"]
        assert r["initial_correct"]==(parsed==r["gold"])
        assert r["checked_correct"]==(final==r["gold"])
    totals={}
    for method in protocol["methods"]:
        rs=[r for r in predictions if r["method"]==method]
        totals[method]={"total":len(rs),"initial_correct":sum(r["initial_correct"] for r in rs),
            "checked_correct":sum(r["checked_correct"] for r in rs),
            "ledger_parseable":sum(r["ledger"] is not None for r in rs),
            "correction_calls":sum(r["correction_used"] for r in rs),
            "correction_gains":sum(not r["initial_correct"] and r["checked_correct"] for r in rs),
            "correction_losses":sum(r["initial_correct"] and not r["checked_correct"] for r in rs)}
    indexed_rows={(r["method"],r["item_id"]):r for r in predictions}
    comparisons={}
    for phase in ("initial","checked"):
        dif=np.array([int(indexed_rows[("counting_guidance",item)][phase+"_correct"])-
                      int(indexed_rows[("textgrad_guidance",item)][phase+"_correct"]) for item in cases])
        rng=np.random.default_rng(20261120)
        draws=rng.choice(dif,(10000,len(dif))).mean(axis=1)
        comparisons[phase]={"difference":float(dif.mean()),"gains":int((dif>0).sum()),"losses":int((dif<0).sum()),
            "descriptive_paired_example_bootstrap_95":np.quantile(draws,[.025,.975]).tolist()}
    result={"status":"passed","scope":protocol["scope"],"methods":totals,"comparisons":comparisons,
        "physical_attempts":len(starts),"physical_failures":sum(r["event"]=="failed" for r in ends.values()),
        "fresh_teacher_calls":0}
    write(p/"summary.json",result)
    return result


def lookup():
    p=BASE/"lookup_control"
    protocol=json.loads((p/"protocol.json").read_text(encoding='utf-8'))
    input_path=BASE/"original_ablation/prepared_inputs.json"
    assert sha(input_path)==protocol["input_sha256"]
    assert sha(ROOT/"scripts/reviewer1_lookup_control_20261120.py")==protocol["runner_sha256"]
    tasks=json.loads(input_path.read_text(encoding='utf-8'))
    task_map={(t["cohort"],t["task_id"]):t for t in tasks}
    predictions=rows(p/"predictions.jsonl")
    assert len(predictions)==2268
    seen=set();counts=defaultdict(lambda:[0,0])
    def flatten(record,prefix=""):
        fields={}
        for k,v in record.items():
            if isinstance(v,dict):
                fields.update(flatten(v,prefix+k+"."))
            else:
                fields[prefix+k]=str(v)
        return fields
    for r in predictions:
        key=(r["cohort"],r["task_id"],r["case_index"],r["method"])
        assert key not in seen
        seen.add(key)
        task=task_map[key[:2]]
        public=task["public_build"]+task["public_audit"]
        if r["method"]=="hamming_nearest_vote":
            fs=flatten(task["cases"][r["case_index"]]["record"])
            distances=[sum(fs[k]!=flatten(c["input_record"])[k] for k in fs) for c in public]
            ids=[i for i,d in enumerate(distances) if d==min(distances)]
            assert ids==r["public_neighbor_indices"]
            votes=Counter(public[i]["expected_label"] for i in ids)
            label=min(votes,key=lambda x:(-votes[x],"uncertain" if x is None else x))
        else:
            assert r["method"]=="minilm_1nn" and len(r["public_neighbor_indices"])==1
            label=public[r["public_neighbor_indices"][0]]["expected_label"]
        gold=task["cases"][r["case_index"]]["gold"]
        assert r["predicted"]==label and r["gold"]==gold and r["correct"]==(label==gold)
        counts[(r["cohort"],r["method"])][0]+=r["correct"]
        counts[(r["cohort"],r["method"])][1]+=1
    result={"status":"passed","rows":len(predictions),"conditions":{c+"/"+m:{"correct":v[0],"total":v[1]} for (c,m),v in counts.items()},
        "checks":"Input/source locks, complete unique grid, Hamming distances and vote ties, copied public labels, final-label scoring. MiniLM numerical neighbor search is reproducible from the frozen encoder and runner; this audit validates its recorded label-copy operation.","new_model_generation_calls":0}
    write(p/"verification.json",result)
    return result


if __name__=="__main__":
    action=sys.argv[1]
    functions={"original":original,"reserved":reserved,"reserved_schema":lambda:reserved(True),"counting":counting,"lookup":lookup}
    result=functions[action]()
    display={k:v for k,v in result.items() if k not in {"per_task","artifacts","approved_only_diagnostics"}}
    if "direct_teacher_program_with_identical_fresh_raw_fallback" in display:
        display["direct_teacher_program_with_identical_fresh_raw_fallback"]={k:{name:value for name,value in v.items() if name!="per_task"} for k,v in display["direct_teacher_program_with_identical_fresh_raw_fallback"].items()}
    print(json.dumps(display,indent=2,ensure_ascii=False))
