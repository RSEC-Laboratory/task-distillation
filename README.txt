What's in here

Scoped to what draft.tex actually reports: the main confirmation, the Gemma
sensitivity check, the routing ablation, the TextGrad/BBH Object Counting
comparison, and the GEPA comparison. Models: Llama 3.1 70B (teacher) / Llama
3.2 3B (student), Gemma 2 27B / Gemma 2 9B, and Qwen 2.5 1.5B (GEPA's frozen
student, per the paper text). Nothing else -- RuleArena, SLR-Bench,
Super-NaturalInstructions, ContractNLI, APE, and a broader BBH screen were
explored during development (their code paths are still in src/, since
that's the real implementation) but never reported, so their configs,
protocol records, and data aren't shipped, and neither is any DeepSeek or
Qwen3 config -- draft.tex doesn't mention either.

Layout mirrors the working repo (src/, data/, configs/, ... at the top
level, not wrapped in a code/ folder) because the package resolves its data
directory as the parent of src/, and won't find data/ through a wrapper.

src/, tests/, gui/
    the implementation and its pytest suite, kept whole -- trimming it to
    only the five reported studies would mean hand-tracing every shared
    import for not much payoff, since it's already small (src/ is 1.5 MB)
    and doesn't ship any data itself

scripts/
    just the six entry points that actually produced a reported number:
    run_t9_confirmation.py (main confirmation), run_t10_gemma_replication.py
    (base class run_t14 imports), run_t14_gemma_generalized_confirmation.py
    (Gemma sensitivity), run_routing_ablation.py, audit_gepa_synthetic_
    comparison.py (recomputes the retained GEPA comparison), and
    download_data.py (how data/ below was fetched)

configs/
    eight run configs, matched one-by-one against the run ids and
    compiled_guidance_path fields in the retained predictions: four for the
    routing ablation (tasks 0 and 26, full-list vs. routed), four for BBH
    Object Counting (compile the artifact, evaluate it, TextGrad matched
    rerun, TextGrad prompt transfer)

protocols/, frozen/
    the frozen protocol registrations and compiled instruction bundle that
    those configs actually read (checked each compiled_guidance_path by
    hand rather than including the whole folder)

data/
    the two real inputs behind the reported numbers: BBH Object Counting
    and the TextGrad reference implementation we ran our matched rerun
    against. The main confirmation, Gemma sensitivity, routing ablation,
    and GEPA comparison all run on generated synthetic tasks, so there's no
    dataset to ship for those -- src/taskcompiler/rules_track/
    synthetic_policies.py builds them from a task id at run time, same id
    in, same task out (checked before packaging this, see below).

results/
    offline recomputation of the five reported numbers from retained
    per-item predictions -- no model server or network needed, cd in and
    run verify.py. This is a trimmed copy of the full offline-evidence
    package: it originally also carried recomputation for four development
    screens (RuleArena, Super-NI, APE, a broader BBH screen) that never
    made the reported table; verify.py here only recomputes the five that
    did.

Retesting

1. pip install -e .[dev,research]     (from this folder; needs Python 3.11+
   -- research pulls in diskcache and gepa, both needed for the TextGrad
   TGD loop and a live GEPA rerun respectively; verify.py below needs
   neither)
2. cd results && python verify.py     recomputes the five reported studies
   offline, straight from the retained predictions. No dependencies beyond
   the standard library for this step.
3. To rerun a live experiment you need your own Ollama endpoint serving the
   student models -- the shipped configs point ollama_base_url at
   127.0.0.1 as a placeholder, change it to your host, then e.g.:
       python -m taskcompiler.run --config configs/bbh_object_counting_llama32_3b_verified_parent70_test100.yaml
   Before packaging this, we ran that exact command against the repo copy
   of this config and confirmed it gets through config loading, loading the
   compiled instruction bundle, loading the BBH data, and building the
   student, and only stops where it tries to reach an Ollama server -- so
   the run path itself works, you just need somewhere for it to call. The
   main confirmation and Gemma sensitivity runs (run_t9_confirmation.py /
   run_t14_gemma_generalized_confirmation.py) additionally need an
   OpenRouter key for the teacher model (see .env.example) -- neither could
   be smoke-tested here for that reason, only their config/data loading and
   the underlying task generator's determinism were checked.
