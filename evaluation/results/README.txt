EXPERIMENT RESULTS

CSV accuracy values are fractions in [0, 1]. "correct" and "total" retain the
numerator and denominator. JSON reports include task-level results, intervals,
call counts, fallback behavior, and the interpretation of each comparison.

TABLE / REPORT                 EXPERIMENT AND SOURCE RECORDS

main_studies.csv                Original main Llama/Gemma and Llama counting
submitted.json/.csv             Complete original studies and diagnostics
routing.csv                    Original two-task full-list/routing comparison
                               records/submitted/

matched_gepa.json/.csv          Matched native GEPA and direct-execution policies
                               records/matched_gepa/

instruction_ablation.json/.csv  Matched allowed-label, rendering, and routing arms
                               records/instruction_ablation/

counting_control.json/.csv      Same runtime checker applied to both guidance methods
                               records/counting_control/

public_lookup.json/.csv         Public-example Hamming and encoder-neighbor controls
                               records/public_lookup/

qwen_adapters.json/.csv         Qwen counting, TextGrad, prompt transfer, and NLP
                               records/qwen_adapters/

reserved_nlp.json/.csv          Reserved 20-task NLP study, fallbacks, unavailable calls
                               records/reserved_nlp/

program_repairs.json            Accepted repairs and program-only execution counts
                               experiments/prepared/programs/ and instruction inputs

teacher_divergence.json         Paired teacher requests and program disagreements
                               records/matched_gepa/

summary.csv                    Every accuracy metric across all panels
reference.json                 Full reference report used by the reproduction check

REPRODUCING

Run "python -B -m evaluation" at the repository root. The same named files and
the complete results.json/accuracy.csv are generated in reproduced/ here.

INTERPRETING THE PANELS

The original main Gemma aggregate uses the recorded task-description fallback
estimates for two syntax failures. The instruction ablation measures its own
fallbacks anew, so its Llama totals should not be substituted for original rates.
Additional controls reuse the retained task families; their intervals describe
those tasks and do not establish independent-family generalization.

Direct program execution and student inference have separate result columns.
Counting adapters have different runtime assistance in the original comparisons;
the counting_control panel applies the same checker policy to both methods.
Qwen interrupted predecessor calls remain charged as project overhead, with
unknown unfinished usage left unknown. Reserved NLP compilation failures and
both HTTP-429 schema attempts are retained; no accuracy is assigned to an
unavailable endpoint. Upstream source-instance validation is reported separately
from saved-row scoring. These scope statements are also retained in each JSON.
