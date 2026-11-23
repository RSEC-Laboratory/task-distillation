TASK DISTILLATION: CODE AND REPRODUCTION

This artifact includes the implementation, retained programs and predictions,
and scripts for checking the reported results. Submitted experiments and
additional controlled measurements are identified separately in the output.

QUICK START (PYTHON 3.11 OR NEWER)

Extract the complete archive, or clone the repository, then run:

    python -B run.py

The default command needs only the Python standard library. It verifies file
hashes, reconstructs large JSONL records in outputs/workspace, executes the
retained DSL programs with the included interpreter, and recomputes numerical
results from saved predictions. It does not call models, contact an API, or
require a GPU. Results are written to outputs/results.

Large records are split into files smaller than 6 MB for anonymous hosting.
run.py reassembles them byte for byte. Do not copy only the source folder.

WHAT IS CHECKED

1. Program execution, accepted public repairs, and rendered finite decisions.
2. Main Llama/Gemma conditions and the original routing comparison.
3. The native matched GEPA panel and symmetric execution fallbacks.
4. Counting, TextGrad, and Super-NaturalInstructions comparison records.
5. Additional matched-label/rendering/routing controls, a common-checker
   counting control, public-example lookups, and reserved NLP outcomes.

Failed compilation, task-description fallbacks, interrupted predecessor calls,
and unsuccessful reserved-task conditions are retained. Program-only replay
counts are distinguished from deployed student accuracy. Descriptive intervals
over the reused task families do not establish independent-family transfer.
The submitted Gemma aggregates use the registered description-fallback estimates
for two syntax-failure tasks; the new matched pass measures its fallbacks anew.

CODE BUILD AND TESTS

The source package is in task-compiler/src/taskcompiler. All 81 source modules
are retained byte-identically so the frozen source hashes remain checkable.
No model weights or credentials are distributed.

The dependency-free interpreter checks can also be run separately:

    python -B -S check_core.py --root outputs/workspace/task-compiler

For the full Python package and selected software tests, create an environment:

    python -m venv .venv

Activate it, then run:

    python -m pip install -r requirements-inference.txt
    python -m pip install --no-deps -e ./task-compiler
    python -m pytest -q task-compiler/tests -o addopts=

The recorded environment used Python 3.12, torch 2.5.1 with CUDA 12.1, and the
versions pinned in requirements-inference.txt. CPU installations can use the
appropriate torch 2.5.1 wheel. Offline record reproduction requires none of
these optional packages. Run the package in editable mode because experiment
data paths are resolved beside src.

NEW STUDENT PREDICTIONS (OPTIONAL)

The retained predictions support numerical reproduction, not a guarantee that
external model providers regenerate identical responses. The optional runner
rerun_student.py sends recorded preparation prompts to an explicitly selected
local Ollama endpoint and writes fresh outputs separately:

    python rerun_student.py --help

For example, after materializing the evidence and installing the recorded
checkpoint in a local Ollama server:

    python rerun_student.py --root outputs/workspace/task-compiler --cohort llama --arm selected_routed --limit 5 --output outputs/fresh-llama

Add --plan-only to inspect the selected requests without contacting the server.

It uses frozen task preparation and makes new student calls; it does not
regenerate teacher programs. A compatible model must already be installed.
No download or inference is triggered by the default reproduction command.
The included historical experiment drivers preserve the study implementation;
their original output paths and provider/cache assumptions are not portable
entrypoints. Use the release runner for a fresh local student check.

UPSTREAM NLP DATA (OPTIONAL)

The default artifact checks recorded references and predictions without
redistributing complete upstream task corpora. For independent source-instance
revalidation or new NLP inference, read external/superni.json and then run:

    python fetch_data.py
    python -B run.py

The downloader uses a fixed upstream commit, verifies download hashes, and
reconstructs the recorded checkout newline convention. Original task-instance
licenses remain applicable; the metadata license does not relicense instances.
NLP request prompts containing upstream instances are represented by SHA-256
digests in the published call evidence. Outputs, reference labels, call counts,
errors, and chronology are retained. PROVENANCE.json records these transformations.

FILE LAYOUT

    run.py                   Integrity check and offline reproduction entrypoint
    check_core.py            Execute the actual retained DSL programs
    reproduce.py             Saved-prediction scoring and statistical checks
    rerun_student.py         Optional new local student predictions
    fetch_data.py            Optional pinned upstream data download
    MANIFEST.json            Public release file hashes
    DATA_PARTS.json          Split-record reconstruction instructions
    PROVENANCE.json          Documented release transformations
    external/                Upstream data pins and license notices
    task-compiler/src/       Original implementation
    task-compiler/reports/   Retained study records and configurations
    task-compiler/runs/      Required run artifacts and comparison records

The top-level entrypoints are authoritative for this release. Historical
support data under reviewer_artifact are consumed by the scoring scripts;
older development comparisons are not substituted for the native GEPA panel.
Third-party license text and original model/data attribution are preserved.
