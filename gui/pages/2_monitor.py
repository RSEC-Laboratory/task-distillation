"""Show logs and metrics for a selected run."""

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from taskcompiler.config import list_runs

st.title("Monitor")

runs = list_runs()
if not runs:
    st.info("No runs yet. Launch one from the Configure page.")
    st.stop()

selected = st.selectbox("Run", runs, format_func=lambda p: p.name, index=0)

if st.button("Refresh"):
    st.rerun()

st.caption(f"Last refreshed: {time.strftime('%H:%M:%S')}")

log_path = selected / "log.txt"
if log_path.exists():
    st.subheader("Execution log")
    st.text(log_path.read_text(encoding="utf-8"))
else:
    st.info("No log written yet.")

metrics_path = selected / "metrics.jsonl"
if metrics_path.exists():
    st.subheader("Stage metrics")
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if rows:
        st.dataframe(rows)

summary_path = selected / "summary.json"
if summary_path.exists():
    st.subheader("Summary")
    st.json(json.loads(summary_path.read_text(encoding="utf-8")))
