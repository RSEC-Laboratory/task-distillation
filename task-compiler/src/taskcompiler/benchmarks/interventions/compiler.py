"""Local, frozen-model clause updates and matched candidate-pool controls."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import time

from .program import RULES, execute_events, format_only, prompt_for, render, output_schema, variants
from .teaching import editor_request, parse_edit, split_groups,editing_schema
from .response import diagnostics, parse_output, projection, response_residual, select_candidate


def summarize_usage(rows):
    result={"attempted_calls":len(rows),"failed_calls":sum(r["error"] is not None for r in rows),
            "summed_request_seconds":sum(r["elapsed_seconds"] for r in rows)}
    for key in ("input_tokens","output_tokens"):
        unknown=sum(r[key] is None for r in rows)
        known=sum(r[key] for r in rows if r[key] is not None)
        result[key]=None if unknown else known
        result[key+"_known_lower_bound"]=known
        result[key+"_unknown_calls"]=unknown
    return result


class Recorder:
    def __init__(self, solver, editor, *, limit=340, on_record=None):
        self.solver,self.editor = solver,editor
        self.limit,self.rows,self.on_record = limit,[],on_record

    def ask(self, system, prompt, schema, *, phase, candidate, group=None, state=None, editing=False):
        if len(self.rows) >= self.limit:
            raise ValueError("Frozen compilation call cap exceeded")
        model = self.editor if editing else self.solver
        row = {"phase":phase,"candidate":candidate,"group":group,"state":state,
               "role":"editor" if editing else "solver", "system":system,"prompt":prompt,
               "structured_decoding":bool(schema),"output":None,"error":None,
               "input_tokens":None,"output_tokens":None}
        started=time.perf_counter()
        try:
            response=model.generate(system,prompt,schema)
            row.update(asdict(response))
            row["output"]=row.pop("text")
        except ValueError:
            # Invalid experiment inputs/configurations are not model failures.
            raise
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}",
                       input_tokens=getattr(exc,"input_tokens",None),output_tokens=getattr(exc,"output_tokens",None))
        finally:
            row["elapsed_seconds"]=time.perf_counter()-started
            self.rows.append(row)
            if self.on_record:
                self.on_record(row)
        return row

    def summary(self):
        return summarize_usage(self.rows)


def compile_pack(pack, solver, editor, *, seed, limit=340, on_record=None):
    program=pack["program"]
    partitions=split_groups(pack,seed)
    ledger=Recorder(solver,editor,limit=limit,on_record=on_record)
    references={(g["id"],state):execute_events(program,g,rules)
                for groups in partitions.values() for g in groups for state,rules in variants().items()}
    candidates={"format":{r:"{rule}" for r in RULES},"teacher":deepcopy(pack["clauses"])}
    predictions={}
    request_cache={}
    request_aliases=[]

    def measure(key,phase):
        values=predictions.setdefault(key,{})
        for group in partitions[phase]:
            for state,rules in variants().items():
                system=render(program,candidates[key],rules)
                prompt=prompt_for(group)
                cache_key=(phase,system,prompt)
                if cache_key not in request_cache:
                    row=ledger.ask(system,prompt,output_schema(program),
                                   phase=phase,candidate=key,group=group["id"],state=state)
                    request_cache[cache_key]=(row,len(ledger.rows)-1)
                row,index=request_cache[cache_key]
                request_aliases.append({"phase":phase,"candidate":key,"group":group["id"],"state":state,"physical_call_index":index})
                values[(group["id"],state)]=parse_output(row["output"],program)

    for key in candidates:
        measure(key,"optimization")
    optimization=[g["id"] for g in partitions["optimization"]]
    def evidence_for(keys):
        # Dialogue appears once, with every rule setting and the same common
        # candidates' observed per-cell outputs. Both joint editors can reconstruct
        # cross-edit correlations; neither receives additional underlying evidence.
        return [{"group":g["id"],"family":g["family"],"input":prompt_for(g),
                 "variant_observations":[{"task_rules":rules,"reference":references[(g["id"],state)],
                       "student_outputs":{k:predictions[k][(g["id"],state)] for k in keys}}
                       for state,rules in variants().items()]}
                for g in partitions["optimization"]]
    evidence=evidence_for(["teacher"])
    # The same two initial edit strings, labels and observations are available to
    # both subsequent updates. The only new information is a processed signal.
    edit_errors={}
    for rule in RULES:
        system,prompt=editor_request(program,candidates["teacher"],[rule],evidence,
                                    {"kind":"single_clause_repair","rule":rule})
        row=ledger.ask(system,prompt,editing_schema(candidates["teacher"],[rule]),phase="proposal",candidate="single_"+rule,editing=True)
        key="single_"+rule
        try:
            candidates[key]=parse_edit(row["output"],candidates["teacher"],[rule])
        except (ValueError,TypeError) as exc:
            candidates[key]=deepcopy(candidates["teacher"])
            edit_errors[key]=f"{type(exc).__name__}: {exc}"
        measure(key,"optimization")
    candidates["composed"]={r:candidates["single_"+r][r] for r in RULES}
    measure("composed","optimization")
    projection_report=projection(predictions["teacher"],{r:predictions["single_"+r] for r in RULES},references,optimization)
    measured_joint=diagnostics(predictions["composed"],references,optimization)
    # The first-order approximation is explicitly falsifiable. A rewritten joint
    # candidate is always remeasured; no estimated effect acts as a certificate.
    residual=response_residual(predictions["teacher"],references,optimization)
    vector_errors=[{"group":k[0],"left":k[1],"right":k[2],"rule":k[3],"slot":k[4],"value":k[5],"signed_error":v}
                   for k,v in sorted(residual.items()) if v]
    response_signal={"kind":"semantic_response_projection", "projection":projection_report,
                     "largest_response_errors":sorted(vector_errors,key=lambda r:(-abs(r["signed_error"]),r["group"],r["rule"],r["slot"],r["value"]))[:24],
                     "actual_composed_scores":measured_joint,
                     "teacher_scores":diagnostics(predictions["teacher"],references,optimization),
                     "single_scores":{r:diagnostics(predictions["single_"+r],references,optimization) for r in RULES},
                     "instruction":"Repair missed rule effects and unwanted changes on invariance controls. Jointly explain the rules without introducing a different original task."}
    scalar_signal={"kind":"ordinary_augmented_accuracy", "candidate_scores":{
        key:diagnostics(values,references,optimization) for key,values in predictions.items()},
        "instruction":"Improve ordinary exact accuracy across all public task variants, preserving the original task."}
    # Scalar control gets ALL scores, even the response scores: information access
    # is matched; the difference is the declared update objective and direction.
    response_parent=deepcopy(candidates["teacher"])
    for rule in projection_report["selected"]:
        response_parent[rule]=candidates["single_"+rule][rule]
    scalar_parent_id=select_candidate(predictions,references,optimization,objective="augmented_accuracy",baseline="format")["selected"]
    joint_evidence=evidence_for(sorted(candidates))
    for key,signal,parent in (("joint_response",response_signal,response_parent),
                               ("joint_accuracy",scalar_signal,candidates[scalar_parent_id])):
        system,prompt=editor_request(program,parent,list(RULES),joint_evidence,signal)
        row=ledger.ask(system,prompt,editing_schema(parent,RULES),phase="proposal",candidate=key,editing=True)
        try:
            candidates[key]=parse_edit(row["output"],parent,RULES)
        except (ValueError,TypeError) as exc:
            candidates[key]=deepcopy(parent)
            edit_errors[key]=f"{type(exc).__name__}: {exc}"
        measure(key,"optimization")
    for key in candidates:
        measure(key,"selection")
    selection=[g["id"] for g in partitions["selection"]]
    common={"format","teacher","single_order","single_source","composed"}
    response_pool={k:v for k,v in predictions.items() if k in common|{"joint_response"}}
    scalar_pool={k:v for k,v in predictions.items() if k in common|{"joint_accuracy"}}
    decisions={
        "response_update":select_candidate(response_pool,references,selection,objective="response",baseline="format"),
        "matched_accuracy_update":select_candidate(scalar_pool,references,selection,objective="augmented_accuracy",baseline="format"),
        "response_pool_accuracy":select_candidate(response_pool,references,selection,objective="augmented_accuracy",baseline="format"),
        "original_accuracy":select_candidate(response_pool,references,selection,objective="original_accuracy",baseline="format"),
    }
    selected={"raw_format":"format","teacher":"teacher",**{k:d["selected"] for k,d in decisions.items()}}
    # A public synthetic audit measures only chosen strings; it never changes any
    # selection. Identical candidates share calls, with aliases retained.
    audit_groups=[g["id"] for g in partitions["synthetic_audit"]]
    for key in sorted(set(selected.values())):
        measure(key,"synthetic_audit")
    return {"version":1,"candidates":candidates,"selected":selected,"decisions":decisions,
            "projection":projection_report,"scalar_parent_id":scalar_parent_id,
            "response_parent":response_parent,"actual_composed_scores":measured_joint,"edit_errors":edit_errors,
            "synthetic_audit":{k:diagnostics(predictions[v],references,audit_groups) for k,v in selected.items()},
            "split_ids":{p:[g["id"] for g in groups] for p,groups in partitions.items()},
            "request_aliases":request_aliases,
            "predictions":[{"candidate":k,"group":g,"state":s,"output":value}
                           for k,values in predictions.items() for (g,s),value in values.items()],
            "usage":ledger.summary(),"calls":ledger.rows,
            "limitations":["Generated surface annotations lack independent semantic adjudication.",
                           "Projection uses first-order measured effects; composition is remeasured.",
                           "Response objective differs from augmented accuracy but novelty is unestablished.",
                           "This compares bounded local clause updates, not native GEPA/MIPRO/SAMMO."]}
