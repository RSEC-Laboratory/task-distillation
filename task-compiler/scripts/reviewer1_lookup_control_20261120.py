"""Frozen public-example lookup controls; no teacher/student generation."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"reports/reviewer1_20261120"
OUT=BASE/"lookup_control"
INPUT=BASE/"original_ablation/prepared_inputs.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


def canonical(record):
    return json.dumps(record,sort_keys=True,separators=(",", ":"))


def features(record,prefix=""):
    result={}
    for key,value in record.items():
        name=prefix+key
        if isinstance(value,dict):
            result.update(features(value,name+"."))
        else:
            result[name]=str(value)
    return result


def vote(labels):
    counts=Counter(labels)
    return min(counts,key=lambda label:(-counts[label],"uncertain" if label is None else label))


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/"protocol.json").exists():
        raise RuntimeError("Lookup protocol already frozen")
    write(OUT/"protocol.json",{
        "id":"reviewer1-public-lookup-20261120-v1",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "question":"How do fixed public-example lookups perform on the same complete original suite?",
        "scope":"Supplementary descriptive control on historical task families, specified after viewing the completed Llama ablation. No confirmatory or independent-generalization claim.",
        "public_data":"All 27 build plus 27 audit inputs and reference labels per task; no teacher program, final reference labels, or tuned parameters enter prediction.",
        "methods":{
            "hamming_nearest_vote":"Flatten the four declared fields; find all public cases at minimum unweighted Hamming distance; majority vote their labels; break label-vote ties lexicographically.",
            "minilm_1nn":"Encode canonical sorted compact JSON using the existing all-MiniLM-L6-v2 snapshot, normalized embeddings and cosine similarity. Choose one highest-similarity public case; exact numerical ties use the first build-then-audit index. Copy its reference label."
        },
        "task_coverage":{"llama":24,"gemma":18},"cases_per_task":27,
        "embedding":"Existing cached sentence-transformers/all-MiniLM-L6-v2, CPU only, local_files_only; record all snapshot-file hashes before scoring.",
        "analysis":"Counts and whole-task descriptive bootstrap, 10000 resamples, seed 20261120. No neighbor-count/metric search or result-dependent task selection.",
        "limits":{"cpu_wall_seconds":180,"teacher_calls":0,"student_generation_calls":0},
        "input_sha256":sha(INPUT),"runner_sha256":sha(__file__),
        "stopping":"Complete the entire grid once or report an operational failure; never substitute a different lookup after observing accuracy."
    })
    print("Lookup protocol frozen; no outcomes computed",flush=True)


def run():
    import numpy as np
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    protocol=json.loads((OUT/"protocol.json").read_text())
    assert sha(INPUT)==protocol["input_sha256"] and sha(__file__)==protocol["runner_sha256"]
    if (OUT/"predictions.jsonl").exists():
        raise RuntimeError("No automatic rerun")
    started=time.monotonic()
    snapshot=Path(snapshot_download("sentence-transformers/all-MiniLM-L6-v2",local_files_only=True))
    write(OUT/"embedding_snapshot.json",{"snapshot_id":snapshot.name,
        "sha256":{p.relative_to(snapshot).as_posix():sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}})
    model=SentenceTransformer(str(snapshot),device="cpu",local_files_only=True)
    data=json.loads(INPUT.read_text())
    predicted=[]
    for task in data:
        if time.monotonic()-started>180:
            raise RuntimeError("Frozen lookup wall-time bound reached")
        public=task["public_build"]+task["public_audit"]
        assert len(public)==54
        public_records=[r["input_record"] for r in public]
        public_features=[features(record) for record in public_records]
        final_records=[r["record"] for r in task["cases"]]
        assert not set(map(canonical,public_records))&set(map(canonical,final_records))
        encoded=model.encode([canonical(record) for record in public_records+final_records],normalize_embeddings=True,show_progress_bar=False)
        similarities=encoded[54:]@encoded[:54].T
        for index,record in enumerate(final_records):
            fs=features(record)
            distances=[sum(fs[k]!=p[k] for k in fs) for p in public_features]
            nearest=[i for i,d in enumerate(distances) if d==min(distances)]
            neighbor=int(np.argmax(similarities[index]))
            for method,label,ids in (
                ("hamming_nearest_vote",vote([public[i]["expected_label"] for i in nearest]),nearest),
                ("minilm_1nn",public[neighbor]["expected_label"],[neighbor]),
            ):
                predicted.append({"cohort":task["cohort"],"task_id":task["task_id"],"case_index":index,
                    "method":method,"predicted":label,"public_neighbor_indices":ids})
    # Only after every prediction is fixed, attach final reference labels.
    gold={(t["cohort"],t["task_id"],c["index"]):c["gold"] for t in data for c in t["cases"]}
    for row in predicted:
        row["gold"]=gold[(row["cohort"],row["task_id"],row["case_index"])]
        row["correct"]=row["predicted"]==row["gold"]
    (OUT/"predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in predicted))
    summary={"scope":protocol["scope"],"status":"complete","model_generation_calls":0,"cohorts":{}}
    for cohort in ("llama","gemma"):
        conditions={}
        for method in protocol["methods"]:
            rows=[r for r in predicted if r["cohort"]==cohort and r["method"]==method]
            means={tid:sum(r["correct"] for r in rows if r["task_id"]==tid)/27 for tid in sorted({r["task_id"] for r in rows})}
            rng=np.random.default_rng(20261120)
            values=np.array(list(means.values()))
            draws=rng.choice(values,(10000,len(values))).mean(axis=1)
            conditions[method]={"correct":sum(r["correct"] for r in rows),"total":len(rows),
                "accuracy":float(values.mean()),"descriptive_task_bootstrap_95":np.quantile(draws,[.025,.975]).tolist(),"per_task":means}
        summary["cohorts"][cohort]=conditions
    summary["elapsed_seconds"]=time.monotonic()-started
    write(OUT/"summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    {"prepare":prepare,"run":run}[sys.argv[1]]()
