"""Show the Streamlit landing page and link to the experiment pages."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

st.set_page_config(page_title="Task Compiler", layout="wide")

st.title("Task Compiler")
st.write(
    "Configure and launch experiments, monitor a running experiment, "
    "and browse past results using the pages in the sidebar."
)
st.write(
    "Tracks: rules_track (ContractNLI, DSL-verified) and guidance_track "
    "(reusable task-level or on-demand question-level guidance for BBH, "
    "open-ended questions, and Docker-isolated HumanEval)."
)
st.info(
    "In guidance_track, task-level mode compiles one reusable artifact. "
    "Question-level mode lets the teacher see each incoming question and is "
    "reported separately as online teacher assistance."
)
