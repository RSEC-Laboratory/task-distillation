"""Freeze one schema-compatible teacher configuration before final-score access."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from taskcompiler.config import load_config
from taskcompiler.lineage import validate_lineage_config

BASE=ROOT/"reports/reviewer1_20261120"
old_protocol=ROOT/"protocols/reviewer1_reserved_superni_20261120_v1.json"
new_protocol=ROOT/"protocols/reviewer1_reserved_schema_superni_20261120_v1.json"
old_config=ROOT/"configs/reviewer1_20261120/superni_reserved_llama.yaml"
new_config=ROOT/"configs/reviewer1_20261120/superni_reserved_schema_llama.yaml"
if new_protocol.exists():
    raise RuntimeError("Additional configuration already frozen")
config=yaml.safe_load(old_config.read_text(encoding="utf-8"))
config.update(run_label="reviewer1_reserved20_schema70b_llama_20261120",
    teacher_model="meta-llama/llama-3.3-70b-instruct",teacher_selection="manual",
    teacher_structured_outputs=True,teacher_openrouter_provider="DeepInfra")
new_config.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
loaded=load_config(new_config)
assert validate_lineage_config(loaded) is None
protocol=json.loads(old_protocol.read_text(encoding="utf-8"))
protocol.update(
    id="reviewer1-reserved-schema-superni-20261120-v1",
    frozen_at_utc=datetime.now(timezone.utc).isoformat(),
    question="Evaluate the unchanged existing NLP adapter on the same reserved tasks using an endpoint that supports its original structured-output contract.",
    config_sha256={new_config.relative_to(ROOT).as_posix():hashlib.sha256(new_config.read_bytes()).hexdigest()},
    teacher_request_format="Original teacher_structured_outputs=true; JSON-schema enforcement. DeepInfra advertises response_format and structured_outputs for Llama 3.3 70B.",
    teacher_selection="Existing manual configuration mode; Llama-family 70B teacher, not a claim of documented student-training-parent matching.",
    runtime="Same frozen Llama3.2:3b, CPU ranker and cases; launches only after the first reserved configuration, counting control and original-ablation continuation have ended.",
    reason="The first available Llama3.1 70B provider produced prose and representation-schema failures on public compilation prompts. This additional configuration is selected for API compatibility, before inspecting final-case responses or accuracy.",
    outcome_access="Only initial-teacher public compilation responses/failure flags and operational event counts have been inspected from the first reserved run; no final-case responses, references or accuracy outputs have been inspected.",
    relation_to_first_configuration="Additional endpoint/teacher configuration, not a replacement. Retain and score every first-configuration outcome and fallback. Same twenty task IDs and first twenty instances; no task selection or prompt retuning.",
    compatibility_source="https://openrouter.ai/api/v1/models/meta-llama/llama-3.3-70b-instruct/endpoints",
    selection_rule="Preserve 70B teacher scale and Llama family; choose the lowest-priced endpoint advertising both response_format and structured_outputs. No model accuracy probe was used.",
    further_configuration_search_allowed=False,
)
protocol["source_sha256"][Path(__file__).relative_to(ROOT).as_posix()]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
new_protocol.write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"protocol":str(new_protocol),"frozen_at_utc":protocol["frozen_at_utc"],"teacher":config["teacher_model"],"provider":config["teacher_openrouter_provider"],"new_model_calls":0},indent=2))
