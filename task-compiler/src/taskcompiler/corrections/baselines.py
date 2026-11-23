"""Official local optimizers adapted to the correction-study information budget.

Both receive 16 optimization cases plus the 8 dedicated demonstrations as
training data, the same 8 selection cases, and no synthetic holdout access.
MIPRO may select up to 3 labeled and 1 metric-accepted bootstrapped
demonstration from this 24-case pool. This is
an access-constrained official implementation, not a published-result replica.
"""
from __future__ import annotations

from taskcompiler.procedural.experiment import LocalModel, SearchConfig, common_contract as legacy_contract
from taskcompiler.procedural.gepa import optimize_gepa
from taskcompiler.procedural.mipro import optimize_mipro
from taskcompiler.procedural.schema import content_hash, validate_prepared_splits

from .schema import all_cases, as_procedural_pack, common_contract, split_cases, validate_pack


def compile_baseline(pack: dict, model: LocalModel, method: str, config: SearchConfig) -> dict:
    if method not in {"gepa_local", "mipro_local"}:
        raise ValueError("Expected official gepa_local or mipro_local baseline")
    validate_pack(pack)
    converted = as_procedural_pack(pack)
    if common_contract(pack) != legacy_contract(converted):
        raise ValueError("Baseline and correction arms must share the identical task/output contract")
    partitions = split_cases(pack, config.seed)
    practice_ids = {row["id"] for rows in partitions.values() for row in rows}
    demonstration_cases = [row for row in all_cases(pack) if row["id"] not in practice_ids]
    partitions["optimization"] = partitions["optimization"] + demonstration_cases
    partitions = validate_prepared_splits(converted["cases"], partitions)
    if method == "gepa_local":
        result = optimize_gepa(converted, model, config, prepared_splits=partitions)
    else:
        result = optimize_mipro(
            converted, model, config, prepared_splits=partitions,
            max_labeled_demos=3, max_bootstrapped_demos=1,
        )
    bundle = result["bundle"]
    bundle["converted_pack_hash"] = bundle["task_pack_hash"]
    bundle["task_pack_hash"] = content_hash(pack)
    protocol = bundle["optimizer_protocol"]
    protocol["study_protocol"] = "instruction_corrections"
    protocol["prepared_split_sizes"] = {name: len(rows) for name, rows in partitions.items()}
    protocol["dedicated_demonstration_case_ids"] = [row["id"] for row in demonstration_cases]
    protocol["training_access"] = "16 optimization cases plus 8 dedicated demonstration cases"
    protocol["selection_access"] = "8 selection cases; synthetic holdout inaccessible"
    protocol["demonstration_pool"] = (
        "24 generated training cases; up to 3 labeled plus 1 metric-accepted bootstrapped demonstration"
        if method == "mipro_local" else "training inputs available to reflection, no dedicated demo selector"
    )
    protocol["adaptations"].append(
        "coverage-stratified prepared split; dedicated demonstrations also available as training cases"
    )
    if method == "mipro_local":
        protocol["adaptations"].append(
            "native demonstrations enabled: maximum 3 labeled plus 1 metric-accepted bootstrapped"
        )
    bundle["bundle_hash"] = content_hash({k: v for k, v in bundle.items() if k != "bundle_hash"})
    return result
