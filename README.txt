TASK DISTILLATION

Implementation and evaluation of checked task preparation for frozen language
models. Source code, experiment definitions, saved predictions, and calculated
results are separated below.

START HERE

    evaluation/results/main_studies.csv       Main Llama/Gemma and counting results
    evaluation/results/matched_gepa.csv       Matched native GEPA comparison
    evaluation/results/instruction_ablation.csv
                                             Label, rendering, and routing controls
    evaluation/results/README.txt             Full results index and interpretation
    experiments/index.csv                    Experiment-to-record/result crosswalk

REPRODUCE THE RESULTS

Python 3.11 or newer is sufficient. From the repository root:

    python -B -m evaluation

This checks release hashes, executes retained DSL programs, checks public
repairs and rendering equivalence, and recomputes every saved-prediction panel.
No model, API key, GPU, or network connection is needed. Recomputed tables and
reports are written to evaluation/results/reproduced/. Distributed reference
tables remain in evaluation/results/.

    python -B -m evaluation --verify-only

REPOSITORY LAYOUT

    src/taskcompiler/          Task compiler, DSL, checking, routing, and adapters
    experiments/configs/       Configurations for the included experiment families
    experiments/protocols/     Frozen experiment definitions and baseline artifacts
    experiments/prepared/      Retained programs and instruction bundles
    experiments/drivers/       Experiment execution and matched-baseline drivers
    evaluation/               Scoring, statistics, program replay, and inference CLI
    evaluation/records/       Raw predictions, calls, failures, and frozen inputs
    evaluation/results/       Named CSV tables and detailed JSON results
    tests/                    Compiler, baseline-boundary, and evaluation tests
    data/                     Benchmark inputs, source pins, and upstream licenses
    provenance/               File hashes and mappings of immutable record references

INSTALL THE IMPLEMENTATION

Offline evaluation above uses only the standard library. For full experiment
execution and software tests, create and activate a virtual environment, then:

    python -m pip install -r requirements-inference.txt
    python -m pip install --no-deps -e .
    python -m pytest -q

The recorded inference environment used Python 3.12 and torch 2.5.1/CUDA 12.1;
CPU execution can use the corresponding CPU torch wheel. Model weights and
credentials are not included. Editable installation preserves the package's
paths to benchmark inputs in this checkout.

FRESH STUDENT INFERENCE

With the recorded checkpoint already available in a local Ollama server:

    python -B -m evaluation.run_student --cohort llama --arm selected_routed --limit 5

Add --plan-only to inspect requests without making model calls. Output defaults
to a new evaluation/results/fresh_<timestamp>/ directory. This command uses the
frozen preparation and generates new student responses; it does not rerun the
teacher. It checks the checkpoint digest and retains every physical request.

See experiments/README.txt for full experiment execution and required inputs.

EVIDENCE AND SCOPE

Saved-output reproduction and fresh model generation are distinct operations.
The evaluation retains failed compilations, fallbacks, interrupted calls, and
negative NLP results. Historical main-study values and additional controlled
measurements are identified separately in the results index. Program execution
is reported separately from student accuracy. All 81 framework source modules
retain their original bytes; evaluation code and entrypoints use this layout.

Large call logs are split below 6 MB per part for anonymous hosting and are
reassembled with hash checks in evaluation/.cache/. Original references inside
saved records are resolved through provenance/layout.json. Optional upstream
NLP source checks require downloading the pinned datasets; see data/superni/.
Third-party attributions and license information are in THIRD_PARTY_NOTICES.txt.
