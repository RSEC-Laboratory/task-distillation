"""Check retained SuperNI evaluation records for the reserved task IDs.

This records evidence available in this workspace, not an assurance about an
author's unrecorded knowledge. It never selects or excludes a task and makes
no model calls.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/reviewer1_20261120"
protocol = json.loads((ROOT / "protocols/reviewer1_reserved_superni_20261120_v1.json").read_text())
reserved = {task["task_id"] for task in protocol["tasks"]}
exposure = {task: [] for task in sorted(reserved)}
records = []
historical_runs = []
for run in sorted((ROOT / "runs").iterdir()):
    if not run.is_dir() or "reviewer1_reserved20_llama_20261120" in run.name:
        continue
    config = run / "config.yaml"
    summary = run / "summary.json"
    candidate = "superni" in run.name.lower()
    if not candidate and summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            candidate = data.get("task_metrics", {}).get("dataset") == "superni"
        except (ValueError, AttributeError):
            pass
    if not candidate:
        continue
    historical_runs.append(run.name)
    paths = [config, summary, run / "artifacts/suite_manifest.json", run / "artifacts/predictions.jsonl"]
    for path in paths:
        if not path.exists():
            continue
        raw = path.read_bytes()
        content = raw.decode("utf-8")
        relative = path.relative_to(ROOT).as_posix()
        hits = sorted(task for task in reserved if task in content)
        records.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "reserved_ids_present": hits})
        for task in hits:
            exposure[task].append(relative)
result = {
    "at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Available historical SuperNI run configurations, summaries, manifests and prediction records; excludes this newly frozen study.",
    "historical_runs": historical_runs,
    "files_checked": len(records),
    "reserved_tasks": len(reserved),
    "reserved_ids_in_prior_records": sum(bool(files) for files in exposure.values()),
    "prior_records_per_task": exposure,
    "records": records,
    "limitations": ["Absence from retained records does not establish absence of unrecorded author exposure.", "Reserved tasks can share datasets and linguistic phenomena with development tasks."],
    "model_calls": 0,
}
out = BASE / "retained_audit/prior_reserved_exposure.json"
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key not in {"records", "prior_records_per_task"}}, indent=2))
