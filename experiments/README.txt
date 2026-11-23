EXPERIMENTS

index.csv maps each experiment to its configuration, saved records, and result
table. Use evaluation/results/ to inspect calculated results and
"python -B -m evaluation" to recompute them without inference.

FULL EXECUTION

Install the implementation and inference dependencies as described in README.txt.
Local configurations use a loopback Ollama endpoint; change the YAML endpoint
to the server you intend to use. Teacher calls require OPENROUTER_API_KEY in
the environment or project .env. Model weights must already be available.

For example, generate a new object-counting run with frozen guidance:

    python -m taskcompiler.run --config experiments/configs/bbh_object_counting_llama32_3b_verified_parent70_test100.yaml

Object-counting inputs and the published demonstration prompt are included in
data/bbh_official/. Fresh NLP runs require the pinned task corpora first:

    python scripts/fetch_data.py

This downloads source datasets only. It does not make teacher or student calls.
Instance licenses listed in data/superni/sources.json remain applicable.

The main synthetic studies use generated task definitions and the frozen
protocols in protocols/. Their drivers are run_t9_confirmation.py for Llama
and run_t14_gemma_generalized_confirmation.py for Gemma. The latter shares
implementation with run_t10_gemma_replication.py. Stage 1 performs public
preparation; stage 2 evaluates the separately defined final cases. Inspect each
driver's --help before a live run. The matched GEPA driver freezes current
source and model/dependency identities before optimization and final evaluation.
Fresh model runs are not expected to reproduce provider responses byte for byte.

prepared/ contains retained teacher programs, syntax-recovery test fixtures,
and instruction bundles. protocols/ preserves the disclosed task splits,
budgets, and baseline artifacts. Saved protocol identifiers and run IDs retain
their original values; provenance/layout.json maps their old file references
to the current paths without changing observations.
