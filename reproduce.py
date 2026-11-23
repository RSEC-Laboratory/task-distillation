"""Recompute paper and supplementary results from retained records, offline.

Only the Python standard library is required. No models are loaded and no
network requests are made. Sentence-encoder neighbor search and model response
generation are separate reproduction levels; saved-record checks do not claim
to regenerate either. Upstream NLP source-instance validation is optional.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accuracy(correct, total):
    return {"correct": int(correct), "total": int(total), "accuracy": correct / total if total else None}


def quantile(values, probability):
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[min(lower + 1, len(values) - 1)] * fraction


class FrozenPCG64:
    """PCG64 XSL-RR stream for the two frozen NumPy default_rng seeds.

    Initial state/increment pairs are recorded NumPy PCG64 seed expansions.
    Bounded 32-bit sampling uses multiply-high rejection, matching the NumPy
    Generator choice calls in these studies. This avoids requiring NumPy only
    to reproduce an existing Monte Carlo bootstrap sequence.

    Independent implementation of the published PCG XSL-RR transition/output
    and Lemire multiply-high bounded sampling, not copied NumPy source:
    https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf
    https://arxiv.org/abs/1805.10941
    """
    STATES = {
        20261120: (175293069496666043790200453473569188007, 303379588208611549728128745228403941347),
        20260727: (5170418299989641538360905967541131569, 209734487121680501811083848134438824815),
    }

    def __init__(self, seed):
        self.state, self.increment = self.STATES[seed]
        self.cached = None

    def uint32(self):
        if self.cached is not None:
            result, self.cached = self.cached, None
            return result
        self.state = (self.state * 47026247687942121848144207491837523525 + self.increment) & ((1 << 128) - 1)
        high, low = self.state >> 64, self.state & ((1 << 64) - 1)
        value, rotation = high ^ low, high >> 58
        raw = ((value >> rotation) | (value << ((-rotation) & 63))) & ((1 << 64) - 1)
        self.cached = raw >> 32
        return raw & ((1 << 32) - 1)

    def index(self, size):
        threshold = ((1 << 32) - size) % size
        while True:
            product = self.uint32() * size
            if (product & ((1 << 32) - 1)) >= threshold:
                return product >> 32


def bootstrap(differences, samples=10000, seed=20261120, numpy_stream=True):
    require(bool(differences), "Empty paired contrast")
    if max(differences)==min(differences):
        return {'difference':differences[0],'descriptive_95_interval':[differences[0],differences[0]],
            'resamples':samples,'seed':seed,'unit_count':len(differences),'implementation':'Exact degenerate bootstrap distribution'}
    if numpy_stream:
        generator = FrozenPCG64(seed)
        draws = sorted(sum(differences[generator.index(len(differences))] for _ in differences) / len(differences) for _ in range(samples))
        interval = [quantile(draws, .025), quantile(draws, .975)]
        implementation = "Frozen NumPy PCG64-compatible choice stream and linear quantiles"
    else:
        generator = random.Random(seed)
        draws = sorted(sum(generator.choice(differences) for _ in differences) / len(differences) for _ in range(samples))
        interval = [draws[int(.025 * (samples - 1))], draws[int(.975 * (samples - 1))]]
        implementation = "Original Python random bootstrap and indexed quantiles"
    return {"difference": sum(differences) / len(differences), "descriptive_95_interval": interval,
            "resamples": samples, "seed": seed, "unit_count": len(differences), "implementation": implementation}


def scoring_functions(root):
    """Load only unchanged, pure scoring functions, avoiding heavy module imports."""
    namespace = {"re": re}
    selections = {
        "src/taskcompiler/translate.py": {"normalise_student_label"},
        "src/taskcompiler/guidance_track/bbh.py": {"parse_answer", "_verified_ledger_sum"},
        "src/taskcompiler/guidance_track/superni.py": {"_normalise_label", "response_matches"},
    }
    hashes = {}
    for relative, names in selections.items():
        path = root / relative
        source = path.read_text(encoding="utf-8")
        functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
        require({node.name for node in functions} == names, "Required scoring functions missing: " + relative)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        hashes[relative] = digest(path)
    return namespace, hashes


def legacy_results(root):
    path = root / "reviewer_artifact/verify.py"
    spec = importlib.util.spec_from_file_location("retained_paper_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The historical GEPA function is deliberately excluded: it concerns a
    # predecessor comparison with different information access.
    result = {
        "main_llama": module.main_study(), "main_gemma": module.gemma_study(),
        "two_task_routing": module.routing_study(), "llama_counting": module.textgrad_study(),
        "historical_nlp_adapters": module.superni_study(), "additional_bbh": module.bbh_development_study(),
        "scope": "Submitted historical records; failed Gemma tasks use the recorded fallback estimates. NLP diagnostics retain the documented identical-prompt fallback correction.",
    }
    for name,denominator_key in [('main_llama','final_cases_per_condition'),('main_gemma','cases_per_condition_after_fallback')]:
        result[name]['conditions']={k:accuracy(v,result[name][denominator_key]) for k,v in result[name]['correct'].items()}
    result['llama_counting']['conditions']={name:accuracy(result['llama_counting'][key],100) for name,key in [('adapter','method_correct'),('textgrad','textgrad_correct'),('published_prompt','prompt_transfer_correct')]}
    for name,student in result['historical_nlp_adapters']['students'].items():
        student['conditions']={label:accuracy(student[key],student['cases']) for label,key in [('positive_examples','positive_examples_correct'),('routed_recorded','recorded_routed_correct'),('routed_identical_prompt_fallback_corrected','identical_prompt_corrected_routed')]}
    # Reapply the historical description fallback rather than mixing it with
    # the newly measured matched-label fallback.
    raw={(str(row['task']),json.dumps(row['input'],sort_keys=True)):row for row in module.load_jsonl('main_predictions.jsonl') if row['condition']=='raw'}
    direct=0
    for task in read(root/'reports/reviewer1_20261120/original_ablation/prepared_inputs.json'):
        if task['cohort']!='llama':continue
        for case in task['cases']:
            label=case['final_program_label'] if task['approved'] else raw[(str(task['number']),json.dumps(case['record'],sort_keys=True))]['predicted']
            direct+=label==case['gold']
    result['main_llama']['direct_execution_historical_description_fallback']=accuracy(direct,648)
    require(direct==610,'Historical direct-execution fallback differs')
    return result


def call_ledger(path):
    events = rows(path)
    starts = {r["call_id"]: r for r in events if r["event"] == "started"}
    ends = {r["call_id"]: r for r in events if r["event"] in {"completed", "failed"}}
    require(len(starts) == sum(r["event"] == "started" for r in events), "Duplicate call start")
    require(len(ends) == sum(r["event"] in {"completed", "failed"} for r in events), "Duplicate terminal call")
    require(starts.keys() == ends.keys(), "Unresolved physical calls")
    return starts, ends


def attempt_accounting(path,allow_unfinished=False):
    events=rows(path)
    starts={r['call_id']:r for r in events if r['event']=='started'}
    ends={r['call_id']:r for r in events if r['event'] in {'completed','failed'}}
    require(len(starts)==sum(r['event']=='started' for r in events),'Duplicate budget-ledger start')
    require(len(ends)==sum(r['event'] in {'completed','failed'} for r in events),'Duplicate budget-ledger terminal')
    require(ends.keys()<=starts.keys(),'Terminal event has no dispatch')
    unfinished=len(starts.keys()-ends.keys())
    require(allow_unfinished or unfinished==0,'Unresolved attempt in a completed panel')
    return {'attempted':len(starts),'completed':sum(r['event']=='completed' for r in ends.values()),
        'failed':sum(r['event']=='failed' for r in ends.values()),'unfinished_outcome_unknown':unfinished,
        'attempts_by_role':dict(Counter(r['role'] for r in starts.values())),
        'known_input_tokens':sum(r.get('response',{}).get('input_tokens',0) for r in ends.values() if r['event']=='completed'),
        'known_output_tokens':sum(r.get('response',{}).get('output_tokens',0) for r in ends.values() if r['event']=='completed'),
        'scope':'Known usage only. Interrupted outstanding attempts are retained, not assigned zero usage.'}


def original_ablation(root, score):
    base = root / "reports/reviewer1_20261120/original_ablation"
    prepared = read(base / "prepared_inputs.json")
    tasks = {(t["cohort"], t["task_id"]): t for t in prepared}
    protocol = read(base / "protocol.json")
    predictions = rows(base / "predictions.jsonl")
    starts, ends = call_ledger(base / "calls.jsonl")
    primary = [r for r in predictions if not r["sentinel"]]
    require(len(primary) == 9072 and len(predictions) == 9408, "Incomplete ablation grid")
    seen = set()
    cells = defaultdict(list)
    for row in predictions:
        key = (row["cohort"], row["task_id"], row["case_index"], row["arm"], row["sentinel"])
        require(key not in seen, "Duplicate ablation row")
        seen.add(key)
        task = tasks[key[:2]]
        case = task["cases"][row["case_index"]]
        require(row["gold"] == case["gold"], "Ablation reference mismatch")
        start, end = starts[row["call_id"]], ends[row["call_id"]]
        require(row["request_hash"] == start["request_hash"] == end["request_hash"], "Ablation call linkage differs")
        raw = end.get("response", {}).get("text", "")
        label = "__INVALID__:transport_failure" if end["event"] == "failed" else score["normalise_student_label"](raw, set(task["labels"]))
        require(row["raw_output"] == raw and row["predicted"] == label, "Ablation response normalization differs")
        require(row["correct"] == (label == case["gold"]), "Ablation score differs")
        if row["arm"] != "raw_original":
            require(start["system"].endswith(task["raw_labels"].rsplit("\n\n", 1)[1]), "Allowed-label contract differs")
        if not row["sentinel"]:
            cells[(row["cohort"], row["task_id"], row["arm"])].append(row)
    output = {}
    for cohort, expected_tasks in [("llama", 24), ("gemma", 18)]:
        cohort_tasks = [t for t in prepared if t["cohort"] == cohort]
        require(len(cohort_tasks) == expected_tasks, "Cohort task count differs")
        conditions = {}
        for arm in protocol["arms"]:
            group = []
            for task in cohort_tasks:
                cell = cells[(cohort, task["task_id"], arm)]
                require(len(cell) == 27, "Incomplete task/arm cell")
                group.extend(cell)
            conditions[arm] = {**accuracy(sum(r["correct"] for r in group), len(group)), "invalid_labels": sum(r["invalid"] for r in group)}
        contrasts = {}
        for first, second in protocol["contrasts"]:
            differences = [(sum(r["correct"] for r in cells[(cohort,t["task_id"],first)]) - sum(r["correct"] for r in cells[(cohort,t["task_id"],second)])) / 27 for t in cohort_tasks]
            contrasts[first + " minus " + second] = bootstrap(differences)
        direct = 0
        for task in cohort_tasks:
            for case in task["cases"]:
                label = case["final_program_label"] if task["approved"] else next(r["predicted"] for r in cells[(cohort,task["task_id"],"raw_labels")] if r["case_index"] == case["index"])
                direct += label == case["gold"]
        output[cohort] = {"tasks": expected_tasks, "conditions": conditions, "contrasts": contrasts,
            "direct_execution_with_same_fresh_description_fallback": accuracy(direct,27*expected_tasks),
            "approved_tasks": sum(t["approved"] for t in cohort_tasks)}
    return {"cohorts": output, "primary_rows": len(primary), "diagnostic_sentinels": len(predictions)-len(primary),
        "physical_attempts": len(starts), "transport_failures": sum(r["event"] == "failed" for r in ends.values()),
        "scope": "Fresh inference on retained historical families. Intervals resample whole tasks; this is not independent-family confirmation."}


def current_gepa(root):
    base = root / "reports/matched_original/development_20261006_v1"
    protocol = read(base / "protocol.json")
    output = {}
    for model in protocol["solver_models"]:
        prefix = model.replace(":", "_")
        totals = Counter()
        per_task = []
        costs = {arm: Counter() for arm in ("full", "gepa")}
        for number in protocol["task_ids"]:
            tid = f"policy-{number:03d}"
            panels = {arm: read(base / "final" / f"{prefix}-{tid}-{arm}.json") for arm in ("raw", "full", "gepa", "direct_dsl")}
            indexed = {arm: {row["case_id"]: row for row in panel["rows"]} for arm,panel in panels.items()}
            require(all(len(v) == 27 for v in indexed.values()), "Incomplete native GEPA final cell")
            ids = set(indexed["raw"])
            require(all(set(v) == ids for v in indexed.values()), "Unpaired native GEPA cases")
            counts = Counter()
            for cid in ids:
                gold = indexed["raw"][cid]["expected_label"]
                for arm in panels:
                    row = indexed[arm][cid]
                    require(row["expected_label"] == gold and row["correct"] == (row["predicted_label"] == gold), "Native GEPA rescoring differs")
                    counts[arm] += row["predicted_label"] == gold
                direct_label = indexed["direct_dsl"][cid]["predicted_label"] if panels["direct_dsl"]["program_available"] else indexed["raw"][cid]["predicted_label"]
                counts["direct_matched_fallback"] += direct_label == gold
            totals.update(counts)
            per_task.append({"task":tid, "correct":dict(counts), "program_available":panels["direct_dsl"]["program_available"]})
            for arm in ("full", "gepa"):
                events = rows(base / "calls" / f"{prefix}-{tid}-{arm}.jsonl")
                requests = [r for r in events if r["event"] == "model_request"]
                roles = Counter(r["role"] for r in requests)
                require(roles["solver_optimization"] <= 216 and roles["teacher_optimization"] <= 4, "Native GEPA call ceiling exceeded")
                costs[arm].update(roles)
                costs[arm]["failed_attempts"] += sum(r["event"] == "model_failure" for r in events)
        differences = [(t["correct"]["full"]-t["correct"]["gepa"])/27 for t in per_task]
        # The original native report uses Python's standard random generator.
        generator=random.Random(protocol['seed'])
        draws=sorted(statistics.mean(generator.choices(differences,k=len(differences))) for _ in range(10000))
        contrast={'difference':statistics.mean(differences),'descriptive_95_interval':[draws[250],draws[9750]],
            'resamples':10000,'seed':protocol['seed'],'unit_count':len(differences),'implementation':'Original native-report Python random.choices and indexed quantiles'}
        recorded=next(r for r in read(base/'summary.json')['models'] if r['solver_model']==model)
        require(all(abs(a-b)<1e-12 for a,b in zip(contrast['descriptive_95_interval'],recorded['descriptive_task_bootstrap_95_interval'])),'Native GEPA interval differs from retained report')
        output[model] = {"conditions":{arm:accuracy(count,216) for arm,count in totals.items()},
            "per_task":per_task,"full_minus_gepa":contrast,"preparation_attempts_by_role":{k:dict(v) for k,v in costs.items()}}
    return {"students":output,"scope":"Current native GEPA comparison: same 54 public labeled cases and call ceilings. All eight tasks retained. Execution-only and symmetric-fallback policies are reported separately.",
        "boundary":"Saved output rescoring and call-budget checks. Does not rerun MiniLM routing or reproduce every original chronological/source-package audit."}


def counting_control(root, score):
    base = root / "reports/reviewer1_20261120/counting_control"
    prepared = read(base / "prepared_inputs.json")
    references = {c["item_id"]:c["gold"] for c in prepared["cases"]}
    cases={c['item_id']:c for c in prepared['cases']}
    predicted = rows(base / "predictions.jsonl")
    require(len(predicted) == 200, "Incomplete common-checker grid")
    starts, ends = call_ledger(base / "calls.jsonl")
    by_phase = {(r["method"],r["item_id"],r["phase"]): ends[cid] for cid,r in starts.items()}
    requests={(r['method'],r['item_id'],r['phase']):r for r in starts.values()}
    methods = defaultdict(list)
    for row in predicted:
        key = row["method"],row["item_id"]
        initial_text = by_phase[(*key,"initial")].get("response",{}).get("text","")
        require(requests[(*key,'initial')]['system']==prepared['systems'][row['method']] and requests[(*key,'initial')]['prompt']==cases[row['item_id']]['prompt'],'Counting prompts differ from the frozen input/contract')
        parsed = score["parse_answer"]("object_counting",initial_text)
        ledger = score["_verified_ledger_sum"](initial_text)
        require(parsed == row["initial"] and ledger == row["ledger"], "Counting initial parse/ledger differs")
        correction = by_phase.get((*key,"correction"))
        require(bool(correction) == row["correction_used"], "Counting correction flag differs")
        require(bool(correction)==(ledger is not None and ledger!=parsed),'Counting common trigger rule differs')
        if correction:
            request=requests[(*key,'correction')]
            expected_prompt=f'A local verifier checked your structured trace and obtained {ledger}. Return exactly `Answer: {ledger}`.'
            require(request['prompt']==expected_prompt and request['system']=='Return the locally verified answer exactly as requested. Do not add reasoning.','Counting correction instruction differs')
        checked = score["parse_answer"]("object_counting",correction.get("response",{}).get("text", "")) if correction else parsed
        require(checked == row["checked"] and row["gold"] == references[row["item_id"]], "Counting final/reference differs")
        require(row["initial_correct"] == (parsed == row["gold"]) and row["checked_correct"] == (checked == row["gold"]), "Counting rescoring differs")
        methods[row["method"]].append(row)
    result = {name:{"off":accuracy(sum(r["initial_correct"] for r in group),100),"on":accuracy(sum(r["checked_correct"] for r in group),100),
        "corrections":sum(r["correction_used"] for r in group),"parseable_ledgers":sum(r["ledger"] is not None for r in group)} for name,group in methods.items()}
    contrasts = {}
    indexed = {(r["method"],r["item_id"]):r for r in predicted}
    for phase in ("initial","checked"):
        differences = [int(indexed[("counting_guidance",key)][phase+"_correct"])-int(indexed[("textgrad_guidance",key)][phase+"_correct"]) for key in references]
        contrasts[phase] = bootstrap(differences)
    return {"methods":result,"contrasts":contrasts,"physical_attempts":len(starts),"transport_failures":sum(r["event"]=="failed" for r in ends.values()),
        "scope":"Frozen prompts on 100 historical Llama cases; same checker/at-most-one-correction policy, different realized call counts. Combined adapter result, not compilation-only gain."}


def lookup_control(root):
    base=root/"reports/reviewer1_20261120"
    tasks={(t["cohort"],t["task_id"]):t for t in read(base/"original_ablation/prepared_inputs.json")}
    counts=defaultdict(lambda:[0,0]);seen=set()
    for row in rows(base/"lookup_control/predictions.jsonl"):
        key=(row["cohort"],row["task_id"],row["case_index"],row["method"])
        require(key not in seen,"Duplicate lookup observation");seen.add(key)
        task=tasks[key[:2]];public=task["public_build"]+task["public_audit"]
        if row["method"]=="hamming_nearest_vote":
            def flatten(record,prefix=""):
                result={}
                for k,v in record.items():
                    if isinstance(v,dict):result.update(flatten(v,prefix+k+"."))
                    else:result[prefix+k]=str(v)
                return result
            source=flatten(task["cases"][row["case_index"]]["record"])
            distances=[sum(source[k]!=flatten(c["input_record"])[k] for k in source) for c in public]
            neighbors=[i for i,d in enumerate(distances) if d==min(distances)]
            require(neighbors==row["public_neighbor_indices"],"Hamming neighbor selection differs")
            votes=Counter(public[i]["expected_label"] for i in neighbors)
            label=min(votes,key=lambda x:(-votes[x],"uncertain" if x is None else x))
        else:
            require(row["method"]=="minilm_1nn" and len(row["public_neighbor_indices"])==1,"Invalid lookup method")
            label=public[row["public_neighbor_indices"][0]]["expected_label"]
        gold=task["cases"][row["case_index"]]["gold"]
        require(row["predicted"]==label and row["gold"]==gold and row["correct"]==(label==gold),"Lookup rescoring differs")
        counts[(row["cohort"],row["method"])][0]+=label==gold
        counts[(row["cohort"],row["method"])][1]+=1
    require(len(seen)==2268,"Incomplete lookup grid")
    return {"conditions":{c+"/"+m:accuracy(*v) for (c,m),v in counts.items()},
        "scope":"Both use 54 public labels. Supplementary protocol specified after observing the completed Llama cohort. Hamming selection recomputed; embedding labels checked against recorded neighbor IDs, without regenerating encoder embeddings."}


QWEN_RUNS={
    "counting_adapter":"2026-10-07_000448_bbh_bbh_counting_qwen25_3b_adapter_20261006",
    "counting_textgrad":"2026-10-06_230851_bbh_bbh_counting_qwen25_3b_textgrad_tgd_20261006",
    "counting_transfer":"2026-10-07_001100_bbh_bbh_counting_qwen25_3b_textgrad_prompt_20261006",
    "superni":"2026-10-07_001402_superni_superni_qwen25_3b_fixed_llama70_development_20261006",
}


def nlp_rows(path,score,expected_tasks,per_task=20):
    predicted=rows(path);seen=set();counts=defaultdict(lambda:[0,0]);refs={}
    for row in predicted:
        key=(row["task_id"],row["item_id"],row["condition"])
        require(key not in seen,"Duplicate NLP prediction");seen.add(key)
        refkey=key[:2];references=tuple(row["references"])
        require(refkey not in refs or refs[refkey]==references,"NLP reference changes across conditions")
        refs[refkey]=references
        correct=score["response_matches"](row["response"],references)
        require(correct==row["correct"],"NLP response rescoring differs")
        counts[row["condition"]][0]+=correct;counts[row["condition"]][1]+=1
    require(len({t for t,_ in refs})==expected_tasks and len(refs)==expected_tasks*per_task,"NLP task/item inventory differs")
    require(len(counts)==5 and all(v[1]==expected_tasks*per_task for v in counts.values()),"Incomplete NLP condition grid")
    return predicted,refs,{k:accuracy(*v) for k,v in counts.items()}


def qwen_adapters(root,score):
    counting={};ids=None
    for method,name in QWEN_RUNS.items():
        if method=="superni":continue
        data=rows(root/"runs"/name/"artifacts/predictions.jsonl")
        require(len(data)==100,"Incomplete Qwen counting panel")
        mapping={row["item_id"]:row for row in data}
        require(len(mapping)==100 and (ids is None or set(mapping)==ids),"Unpaired Qwen counting cases")
        ids=set(mapping);correct=initial=corrections=0
        for row in data:
            value=score["parse_answer"]("object_counting",row["raw_output"])
            require(value==row["parsed_output"] and row["correct"]==(value==row["gold"]),"Qwen counting rescoring differs")
            correct+=value==row["gold"]
            initial+=score["parse_answer"]("object_counting",row["initial_output"] or row["raw_output"])==row["gold"]
            corrections+=bool(row["local_correction_used"])
        counting[method]={**accuracy(correct,100),"initial_correct":initial,"local_corrections":corrections,
            'all_preparation_and_final_attempts':attempt_accounting(root/'runs'/name/'artifacts/model_calls.jsonl')}
    _,_,nlp=nlp_rows(root/"runs"/QWEN_RUNS["superni"]/"artifacts/predictions.jsonl",score,4)
    interrupted=root/'runs/2026-10-06_230554_bbh_bbh_counting_qwen25_3b_textgrad_tgd_20261006/artifacts'
    require((interrupted/'termination.json').exists(),'Interrupted predecessor lacks its termination record')
    return {"counting":counting,"nlp_conditions":nlp,'nlp_attempts':attempt_accounting(root/'runs'/QWEN_RUNS['superni']/'artifacts/model_calls.jsonl'),
        'interrupted_predecessor_attempts':attempt_accounting(interrupted/'model_calls.jsonl',True),
        "scope":"Submitted exploratory Qwen adapter panels; counting has unequal runtime assistance. Positive-example fallback is retained in the four-task NLP panel. Completed algorithm runs and interrupted project overhead are shown separately."}


def reserved(root,score):
    name="2026-11-20_115943_superni_reviewer1_reserved20_llama_20261120"
    run=root/"runs"/name
    predicted,refs,conditions=nlp_rows(run/"artifacts/predictions.jsonl",score,20)
    artifacts=[read(path) for path in (run/"artifacts").glob("task*.json")]
    require(len(artifacts)==20 and all(a["final_representation"] is None for a in artifacts),"Reserved task fallback inventory differs")
    starts,ends=call_ledger(run/"artifacts/model_calls.jsonl")
    roles=Counter(r["role"] for r in starts.values())
    require(roles["teacher"]==20 and roles["student"]==800,"Reserved call accounting differs")
    protocol=read(root/"protocols/reviewer1_reserved_superni_20261120_v1.json")
    means={t['task_id']:{condition:sum(r['correct'] for r in predicted if r['task_id']==t['task_id'] and r['condition']==condition)/20 for condition in conditions} for t in protocol['tasks']}
    contrasts={}
    for comparator in ['raw_definition','positive_examples','parent_one_shot','validated_full']:
        differences=[v['validated_routed']-v[comparator] for v in means.values()]
        contrasts['validated_routed minus '+comparator]=bootstrap(differences,100000,20260727)
    source_checked=0
    for task in protocol["tasks"]:
        source=root/"data/superni/tasks"/(task["task_id"]+".json")
        if not source.exists():continue
        require(digest(source)==task["source_sha256"],"Downloaded upstream NLP task hash differs")
        official=read(source)
        for item in official["Instances"][:20]:
            require(tuple(item["output"])==refs[(task["task_id"],item["id"])],"NLP references differ from upstream instances")
        source_checked+=1
    schema=[];schema_requests=[]
    for name in ["2026-11-20_123413_superni_reviewer1_reserved20_schema70b_llama_20261120","2026-11-20_123628_superni_reviewer1_reserved20_schema70b_transport1_llama_20261120"]:
        path=root/"runs"/name
        first,last=call_ledger(path/"artifacts/model_calls.jsonl")
        require(len(first)==len(last)==1 and all(r["event"]=="failed" and "429" in r["error"] for r in last.values()),"Schema endpoint failure differs")
        require(not (path/"artifacts/predictions.jsonl").exists(),"Unavailable endpoint has fabricated final observations")
        schema_requests.append(next(iter(first.values()))['request'])
        schema.append({"failed_teacher_attempts":1,"successful_responses":0,"final_observations":0})
    require(schema_requests[0]==schema_requests[1],'Transport restart changed schema-teacher request content')
    return {"conditions":conditions,"fallback_tasks":20,"physical_attempts_by_role":dict(roles),
        'contrasts':contrasts,
        "source_instance_validation":{"tasks_checked":source_checked,"expected_tasks":20,"status":"passed" if source_checked==20 else "not_complete; saved-row references rescored, upstream files not redistributed"},
        "schema_enabled_configuration":{"status":"unavailable","attempts":schema,"accuracy":None},
        "scope":"All twenty compilation-format failures and the two HTTP-429 attempts retained. No positive independent-transfer claim."}


def teacher_divergence(root):
    sys.path.insert(0,str(root/'src'))
    from taskcompiler.dsl.parser import parse_program
    from taskcompiler.dsl.interpreter import run
    base=root/'reports/matched_original/development_20261006_v1'
    public={t['task_id']:t for t in read(base/'public_tasks.json')}
    final=read(base/'sealed/final_cases.json')
    output=[]
    for number in range(8):
        tid=f'policy-{number:03d}';request=[];response=[];programs=[]
        for model in ['qwen2.5_3b','llama3.2_3b']:
            events=[r for r in rows(base/'calls'/f'{model}-{tid}-full.jsonl') if r.get('role')=='teacher_optimization']
            request.append(next(r['body'] for r in events if r['event']=='transport_request'))
            response.append(next(r['payload']['message']['content'] for r in events if r['event']=='transport_response'))
            prepared=read(base/'prepared'/f'{model}-{tid}-full.json')
            source=prepared['method']['trace'].get('dsl',{}).get('initial_source')
            programs.append(parse_program(source) if source else None)
        domain=[c['input_record'] for c in public[tid]['build_cases']+public[tid]['audit_cases']+final[tid]]
        require(len(domain)==len({json.dumps(r,sort_keys=True) for r in domain})==81,'Divergence replay domain differs')
        differences=sum(run(programs[0],record).output_label!=run(programs[1],record).output_label for record in domain) if all(p is not None for p in programs) else None
        output.append({'task':tid,'identical_initial_request':request[0]==request[1],'identical_initial_response':response[0]==response[1],
            'initial_program_available':[p is not None for p in programs],'domain_inputs':81,'program_output_disagreements':differences})
    require(all(r['identical_initial_request'] for r in output),'Initial teacher request pairing differs')
    require([r['task'] for r in output if not r['identical_initial_response']]==['policy-001','policy-004','policy-007'],'Teacher response-difference inventory differs')
    require(output[1]['program_output_disagreements']==12,'Policy-001 program disagreement differs')
    return {'tasks':output,'scope':'Separate generated responses differed before student selection. This replay does not identify a hardware/kernel root cause.'}


def program_repairs(root):
    sys.path.insert(0,str(root/"src"))
    from taskcompiler.dsl.parser import parse_program
    from taskcompiler.dsl.interpreter import run
    tasks=read(root/"reports/reviewer1_20261120/original_ablation/prepared_inputs.json")
    protocol=read(root/'reports/reviewer1_20261120/original_ablation/protocol.json')
    retained=list(protocol['retained_input_sha256'])
    accepted=0;counts=Counter();program_scores=defaultdict(lambda:Counter());conversion=Counter()
    for task in tasks:
        if task["syntax_failure"]:continue
        group=program_scores[task["cohort"]]
        programs={}
        for phase in ['initial','final']:
            suffix=f"{task['task_id']}.teacher.{phase}.dsl"
            candidates=[p for p in retained if p.endswith(suffix) and (('gemma' in p)==(task['cohort']=='gemma'))]
            require(len(candidates)==1,'Ambiguous retained program source')
            programs[phase]=parse_program((root/candidates[0]).read_text(encoding='utf-8'))
        for case in task["cases"]:
            initial=run(programs['initial'],case['record']).output_label
            final=run(programs['final'],case['record']).output_label
            require(initial==case['initial_program_label'] and final==case['final_program_label'],'Program execution differs from retained labels')
            group['initial_correct']+=initial==case['gold']
            group['final_correct']+=final==case['gold']
            group['total']+=1
        public=task["public_build"]+task["public_audit"]
        if task['approved']:
            domain=[c['input_record'] for c in public]+[c['record'] for c in task['cases']]
            require(len(domain)==len({json.dumps(c,sort_keys=True) for c in domain})==81,'Finite task domain is not 81 disjoint records')
            decision_lines=task['representations']['final_list'][1:]
            for record in domain:
                flat={k:str(v) for fields in record.values() for k,v in fields.items()}
                labels=set()
                for line in decision_lines:
                    match=re.fullmatch(r'If (.*), return ([A-Za-z_]+)\.',line)
                    require(bool(match),'Unrecognized retained decision entry')
                    conditions,label=match.groups()
                    if conditions=='any input' or all(flat.get(k.split('.')[-1])==v for k,v in (part.split('=',1) for part in conditions.split(', '))):labels.add(label)
                expected=run(programs['final'],record).output_label or 'uncertain'
                require(labels=={expected},'Rendered decision list changes program output')
            conversion[task['cohort']]+=1
        for repair in task["repair_records"]:
            if not repair["accepted"]:continue
            before=parse_program(repair["original_source"]);after=parse_program(repair["proposed_source"])
            old={i for i,c in enumerate(public) if run(before,c["input_record"]).output_label==c["expected_label"]}
            new={i for i,c in enumerate(public) if run(after,c["input_record"]).output_label==c["expected_label"]}
            require(old<new,"Accepted repair violates strict public non-regression")
            accepted+=1
            name=repair["rule_name"]
            kind="teacher" if name=="<complete-program>" else "condition_narrowing" if name.startswith("local-exclusion:") else "insertion"
            counts[(task["cohort"],kind)]+=1
    require(accepted==72,"Accepted repair count differs")
    return {"accepted_changes_replayed":accepted,"repair_counts":{c+"/"+k:v for (c,k),v in counts.items()},
        "program_only_scores_reexecuted":{k:dict(v) for k,v in program_scores.items()},'approved_decision_lists_checked_over_81_inputs':dict(conversion),
        "scope":"All accepted changes executed on 54 public cases. Program-only totals reexecuted from saved DSL; include unapproved parsed candidates and exclude two missing Gemma programs. Conversion agrees with generated programs, not necessarily with reference labels everywhere."}


def metric_rows(data,prefix=""):
    if isinstance(data,dict):
        if set(data)>={"correct","total","accuracy"}:
            yield [prefix,data['correct'],data['total'],data['accuracy']]
        for key,value in data.items():
            yield from metric_rows(value,prefix+('/' if prefix else '')+str(key))
    elif isinstance(data,list):
        for i,value in enumerate(data):yield from metric_rows(value,prefix+'/'+str(i))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'task-compiler')
    parser.add_argument('--output',type=Path,default=Path.cwd()/'reproduced_results')
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    require((root/'src/taskcompiler').is_dir(),'--root must name the task-compiler directory')
    score,hashes=scoring_functions(root)
    result={'status':'passed','mode':'offline saved-record rescoring; no inference or network','scoring_source_sha256':hashes}
    checks=[('submitted',lambda:legacy_results(root)),('current_native_gepa',lambda:current_gepa(root)),
            ('qwen_adapters',lambda:qwen_adapters(root,score)),('new_rule_ablation',lambda:original_ablation(root,score)),
            ('new_common_checker',lambda:counting_control(root,score)),('public_example_lookup',lambda:lookup_control(root)),
            ('reserved_nlp',lambda:reserved(root,score)),('program_repair_replay',lambda:program_repairs(root)),
            ('teacher_compilation_divergence',lambda:teacher_divergence(root))]
    for name,check in checks:
        result[name]=check();print('Passed: '+name,flush=True)
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (output/'accuracy.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['section','correct','total','accuracy']);writer.writerows(metric_rows(result))
    print(json.dumps({'status':'passed','model_generation_calls':0,'output':str(output),'upstream_nlp_source_validation':result['reserved_nlp']['source_instance_validation']},indent=2))


if __name__=='__main__':
    try:main()
    except (VerificationError,FileNotFoundError,KeyError) as error:
        print('Verification failed: '+str(error),file=sys.stderr);raise SystemExit(1)
