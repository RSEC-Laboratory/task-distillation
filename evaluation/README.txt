EVALUATION

Run from the repository root:

    python -B -m evaluation

The command verifies distributed files, executes the retained rule programs,
and rescores all nine experiment panels. It checks the resulting numerical
report against results/reference.json. All work is local and uses Python's
standard library. Output goes to results/reproduced/ under this directory.

CODE

    __main__.py       Evaluation entrypoint and reference comparison
    core.py           DSL execution, public repair gate, finite rendering checks
    reproduce.py      Panel orchestration and CSV/JSON export
    submitted.py      Original main, routing, counting, and diagnostic panels
    matched_gepa.py   Matched native GEPA scoring and call ceilings
    controls.py       Label/rendering/routing, common checker, and lookup controls
    adapters.py       Qwen counting/TextGrad/NLP and reserved NLP outcomes
    programs.py       Program repairs and paired teacher-response comparisons
    common.py         Input readers, scoring-function loading, and call accounting
    bootstrap.py      Frozen random streams and descriptive bootstrap intervals
    layout.py         Hash verification and immutable-reference resolution
    run_student.py    Optional new local student calls on frozen preparation

DATA AND OUTPUTS

records/ holds the saved observations and experiment inputs. It is not used
as an output directory. Every panel in results/ is named after its experiment;
results/README.txt gives the table-to-record crosswalk. Reproduced and fresh
outputs are ignored by Git to keep saved references separate from new runs.

Individual operations:

    python -B -m evaluation.core
    python -B -m evaluation.reproduce
    python -B -m evaluation.run_student --help

Reproduction checks saved labels, raw responses, request linkage, case pairing,
budgets, programs, and accepted public repairs. It does not regenerate model
responses or sentence-encoder neighbors. Optional upstream NLP instance checks
are enabled when the pinned source files are present under data/superni/tasks/.
