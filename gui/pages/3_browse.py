"""Browse past runs, metrics, logs, evidence, and predictions."""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from taskcompiler.config import list_runs

st.title("Browse past runs")

runs = list_runs()
if not runs:
    st.info("No runs yet. Launch one from the Configure page.")
    st.stop()

rows = []
for run_dir in runs:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = {"run": run_dir.name, "wall_seconds": round(summary.get("total_wall_seconds", 0), 1)}
    row.update(summary.get("task_metrics", {}))
    teacher_tokens = summary.get("teacher_tokens", {})
    row["teacher_tokens"] = teacher_tokens.get("input", 0) + teacher_tokens.get("output", 0)
    rows.append(row)

st.dataframe(rows, width="stretch")

selected_name = st.selectbox("Inspect a run", [r.name for r in runs])
selected = next(r for r in runs if r.name == selected_name)

config_path = selected / "config.yaml"
if config_path.exists():
    st.subheader("Config")
    st.code(config_path.read_text(encoding="utf-8"), language="yaml")

evidence_path = selected / "evidence_graph.jsonl"
if evidence_path.exists() and evidence_path.stat().st_size > 0:
    st.subheader("Task Evidence Graph")
    records = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    st.dataframe(records, width="stretch")

artifacts_dir = selected / "artifacts"
if artifacts_dir.exists():
    predictions_path = artifacts_dir / "predictions.jsonl"
    if predictions_path.exists() and predictions_path.stat().st_size > 0:
        st.subheader("Answers and predictions")
        predictions = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        st.dataframe(predictions, width="stretch")

    guidance_events_path = artifacts_dir / "guidance_events.jsonl"
    if guidance_events_path.exists() and guidance_events_path.stat().st_size > 0:
        st.subheader("Guidance generation events")
        guidance_events = [
            json.loads(line)
            for line in guidance_events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        st.dataframe(guidance_events, width="stretch")

    st.subheader("Artifacts")
    for artifact in sorted(artifacts_dir.iterdir()):
        with st.expander(artifact.name):
            st.text(artifact.read_text(encoding="utf-8")[:5000])
