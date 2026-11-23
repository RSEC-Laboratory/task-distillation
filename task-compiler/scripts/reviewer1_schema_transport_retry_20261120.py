"""One documented rate-limit restart, with identical scientific configuration."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"reports/reviewer1_20261120"
OUT=BASE/"queue_schema_retry"
PROTOCOL=ROOT/"protocols/reviewer1_reserved_schema_superni_20261120_v2_transport.json"
CONFIG=ROOT/"configs/reviewer1_20261120/superni_reserved_schema_transport_llama.yaml"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


def now():
    return datetime.now(timezone.utc)


def status(state,**extra):
    data={"at_utc":now().isoformat(),"state":state,"pid":os.getpid(),**extra}
    write(OUT/"status.json",data);print(json.dumps(data),flush=True)


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not PROTOCOL.exists(),"Transport amendment already frozen"
    runs=list((ROOT/"runs").glob("*reviewer1_reserved20_schema70b_llama_20261120"))
    assert len(runs)==1
    run=runs[0]
    calls=[json.loads(s) for s in (run/"artifacts/model_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    starts=[r for r in calls if r["event"]=="started"]
    fails=[r for r in calls if r["event"]=="failed"]
    assert len(starts)==len(fails)==1 and starts[0]["role"]=="teacher" and "429" in fails[0]["error"]
    assert not any(r["event"]=="completed" for r in calls)
    assert not (run/"artifacts/predictions.jsonl").exists()
    config_path=ROOT/"configs/reviewer1_20261120/superni_reserved_schema_llama.yaml"
    original=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config=dict(original,run_label="reviewer1_reserved20_schema70b_transport1_llama_20261120")
    assert {k:v for k,v in config.items() if k!="run_label"}=={k:v for k,v in original.items() if k!="run_label"}
    CONFIG.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    protocol=read(ROOT/"protocols/reviewer1_reserved_schema_superni_20261120_v1.json")
    deadline=datetime.fromisoformat(starts[0]["at_utc"])+timedelta(seconds=7200)
    protocol.update(id="reviewer1-reserved-schema-superni-20261120-v2-transport",frozen_at_utc=now().isoformat(),
        config_sha256={CONFIG.relative_to(ROOT).as_posix():hashlib.sha256(CONFIG.read_bytes()).hexdigest()},
        transport_amendment={"first_run":run.relative_to(ROOT).as_posix(),"first_call_log_sha256":hashlib.sha256((run/"artifacts/model_calls.jsonl").read_bytes()).hexdigest(),
            "reason":"First provider request returned HTTP 429 with no successful response or student call. One cooldown restart is allowed for transport availability, not accuracy.",
            "successful_responses_before_amendment":0,"failed_physical_teacher_attempts_retained":1,"usage_for_failed_attempt":"unknown; charged as an attempt",
            "scientific_changes":0,"administrative_change":"Unique run label only; same model, provider, system/user prompts, format, source, seed, cases and scoring.",
            "minimum_cooldown_seconds":60,"retry_not_before":(datetime.fromisoformat(fails[0]["at_utc"])+timedelta(seconds=60)).isoformat(),
            "absolute_deadline_utc":deadline.isoformat(),"maximum_transport_restarts":1,"no_further_restart_or_provider_change":True})
    protocol["limits"]["teacher_attempts_maximum_including_prior_transport_failure"]=61
    protocol["source_sha256"][Path(__file__).relative_to(ROOT).as_posix()]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(PROTOCOL,protocol)
    status("frozen_transport_amendment",deadline_utc=deadline.isoformat())


def run():
    protocol=read(PROTOCOL)
    assert not (OUT/"launch.json").exists(),"Only one restart permitted"
    amendment=protocol["transport_amendment"]
    first_log=ROOT/amendment["first_run"]/"artifacts/model_calls.jsonl"
    assert hashlib.sha256(first_log.read_bytes()).hexdigest()==amendment["first_call_log_sha256"]
    not_before=datetime.fromisoformat(amendment["retry_not_before"])
    while now()<not_before:
        time.sleep(min(5,(not_before-now()).total_seconds()))
    command=[sys.executable,"-B",str(ROOT/"scripts/run_external_comparison.py"),"--config",str(CONFIG),"--protocol",str(PROTOCOL)]
    write(OUT/"launch.json",{"at_utc":now().isoformat(),"command":command})
    deadline=datetime.fromisoformat(amendment["absolute_deadline_utc"])
    with (OUT/"stdout.log").open("w",encoding="utf-8") as stdout,(OUT/"stderr.log").open("w",encoding="utf-8") as stderr:
        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ),stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        while child.poll() is None:
            if now()>deadline:
                child.terminate();child.wait(timeout=30)
                status("operational_stop",reason="Original absolute configuration deadline");return
            status("running",child_pid=child.pid)
            time.sleep(30)
    status("finished",exit_code=child.returncode)


if __name__=="__main__":
    action=sys.argv[1]
    {"prepare":prepare,"run":run}[action]()
