"""Pinned official IFEval/BBH data, with outcome-independent pilot selection.

This module never calls a model. Benchmark inputs and verifier metadata are
local evaluation material and must not be included in remote teacher requests.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import unicodedata
from urllib.request import Request, urlopen

IFEVAL_COMMIT = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
BBH_COMMIT = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
BBH_TASKS = ("logical_deduction_three_objects", "boolean_expressions", "object_counting")
DEFAULT_SEED = 20261005
DEFAULT_COUNT = 16


@dataclass(frozen=True)
class Source:
    benchmark: str
    repository: str
    commit: str
    path: str
    sha256: str
    bytes: int

    @property
    def url(self):
        return f"https://raw.githubusercontent.com/{self.repository}/{self.commit}/{self.path}"

    @property
    def cache_path(self):
        if self.benchmark == "ifeval" and self.path.endswith(".py"):
            return f"vendor/{self.commit}/{self.path}"
        return f"sources/{self.benchmark}/{self.commit}/{self.path}"


# Exact bytes fetched from the official repositories before any pilot outcomes.
SOURCE_FILES = (
    Source("bbh", "suzgunmirac/BIG-Bench-Hard", "9ee07bd481feebf959a6b59d61ea57bdcf30964d", "bbh/boolean_expressions.json", "ea6c754ec005e2d3f2d085d349a740f593b5764f32ea4638ffed4cfc0061b12a", 17172),
    Source("bbh", "suzgunmirac/BIG-Bench-Hard", "9ee07bd481feebf959a6b59d61ea57bdcf30964d", "bbh/logical_deduction_three_objects.json", "e590fae72e3a6041fbd23bd8ff6b74baf44ca563f6ba40a041c55e7b867c1681", 112213),
    Source("bbh", "suzgunmirac/BIG-Bench-Hard", "9ee07bd481feebf959a6b59d61ea57bdcf30964d", "bbh/object_counting.json", "8acef14dbbdd40d2326ef7755750b6b5fe919fef06276af1bab3f1c297f022ff", 35890),
    Source("bbh", "suzgunmirac/BIG-Bench-Hard", "9ee07bd481feebf959a6b59d61ea57bdcf30964d", "LICENSE", "4ef2ff4295d26ab6211235039c132697408cec5391d2c091b41e983771978db8", 1068),
    Source("bbh", "suzgunmirac/BIG-Bench-Hard", "9ee07bd481feebf959a6b59d61ea57bdcf30964d", "README.md", "9c41105f4a349fcba7f87a422df0e744a9ed8e7a30c2db6676a9ecad372bc9db", 3475),
    Source("ifeval", "google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0", "instruction_following_eval/data/input_data.jsonl", "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49", 207111),
    Source("ifeval", "google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0", "instruction_following_eval/evaluation_lib.py", "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c", 6984),
    Source("ifeval", "google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0", "instruction_following_eval/evaluation_main.py", "e6df07a04d25a0e7134933ba2400c26e0129a243e449c88df55043e583fc4b4a", 2449),
    Source("ifeval", "google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0", "instruction_following_eval/instructions_registry.py", "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc", 7240),
    Source("ifeval", "google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0", "instruction_following_eval/instructions_util.py", "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b", 19538),
    Source("ifeval", "google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0", "instruction_following_eval/instructions.py", "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf", 55162),
    Source("ifeval", "google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0", "instruction_following_eval/README.md", "fef3c44408e4b39837af345095ccce7da798853fd60ea94a1de63afa9d3d67b3", 1457),
    Source("ifeval", "google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0", "instruction_following_eval/requirements.txt", "1b1716c9ac21b9cc2bd0d7354ed8f7b6982ccf35598529601a7c08508acec94a", 35),
    Source("ifeval", "google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0", "LICENSE", "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30", 11358),
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalized_input_hash(text):
    return sha256(" ".join(unicodedata.normalize("NFC", text).split()).encode("utf-8"))


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Non-finite JSON constant")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def download_bytes(url):
    request = Request(url, headers={"User-Agent": "taskcompiler-pinned-benchmark-acquisition/1"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _immutable_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"Existing pinned artifact differs: {path.name}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def acquire_sources(cache_root, *, sources=SOURCE_FILES, fetch=None):
    """Reuse verified bytes or download exact pinned bytes; never repair a bad cache."""
    root = Path(cache_root).resolve()
    fetch = download_bytes if fetch is None else fetch
    records = []
    for source in sources:
        path = root / source.cache_path
        data = path.read_bytes() if path.exists() else fetch(source.url)
        if len(data) != source.bytes or sha256(data) != source.sha256:
            raise ValueError(f"Pinned source checksum mismatch: {source.cache_path}")
        _immutable_write(path, data)
        records.append({"benchmark": source.benchmark, "repository": source.repository,
                        "commit": source.commit, "source_path": source.path,
                        "url": source.url, "cache_path": source.cache_path,
                        "sha256": source.sha256, "bytes": source.bytes})
    return records


def _check_cases(cases):
    ids, inputs = set(), set()
    for case in cases:
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in ids:
            raise ValueError("Missing or duplicated benchmark case ID")
        if not isinstance(case["input"], str) or not case["input"].strip():
            raise ValueError("Benchmark input must be nonempty text")
        digest = normalized_input_hash(case["input"])
        if digest in inputs:
            raise ValueError("Duplicate normalized benchmark input")
        ids.add(case["id"])
        inputs.add(digest)
    return cases


def parse_ifeval(data):
    cases = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = strict_json(line)
        if type(row.get("key")) is not int:
            raise ValueError("IFEval key must be an integer")
        instructions, kwargs = row.get("instruction_id_list"), row.get("kwargs")
        if (not isinstance(instructions, list) or not instructions
                or any(not isinstance(item, str) or not item for item in instructions)
                or not isinstance(kwargs, list) or len(kwargs) != len(instructions)
                or any(not isinstance(item, dict) for item in kwargs)):
            raise ValueError("IFEval verifier metadata is malformed")
        cases.append({"id": f"ifeval_{row['key']}", "benchmark": "ifeval", "task": "ifeval",
                      "input": row["prompt"], "key": row["key"],
                      "instruction_id_list": instructions, "kwargs": kwargs})
    return _check_cases(cases)


def parse_bbh(data, task):
    if task not in BBH_TASKS:
        raise ValueError("Unsupported BBH task family")
    rows = strict_json(data.decode("utf-8"))["examples"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("BBH examples must be a nonempty list")
    cases = []
    for index, row in enumerate(rows):
        target = row.get("target")
        if not isinstance(target, str) or not target:
            raise ValueError("BBH reference must be nonempty text")
        cases.append({"id": f"{task}_{index:04d}", "benchmark": "bbh", "task": task,
                      "source_index": index, "input": row["input"], "reference": target})
    return _check_cases(cases)


def select_cases(cases, *, task_id, count=DEFAULT_COUNT, seed=DEFAULT_SEED):
    """Hash-rank stable IDs; no model outputs, references or eligibility filters."""
    _check_cases(cases)
    if type(seed) is not int:
        raise ValueError("Selection seed must be an integer")
    if count is not None and (type(count) is not int or not 0 < count <= len(cases)):
        raise ValueError("Count must be positive and no larger than the source, or None for all")
    def rank(case):
        return sha256(f"{seed}\0{task_id}\0{case['id']}".encode("utf-8")), case["id"]
    return sorted(cases, key=rank)[:count]


def prepare_general_benchmarks(cache_root, output_dir, *, count=DEFAULT_COUNT, seed=DEFAULT_SEED, fetch=None):
    """Create a new immutable pilot manifest and case files, before model scoring."""
    root, out = Path(cache_root).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError("Prepared benchmark output must be a new directory")
    if not out.is_relative_to(root) or out == root:
        raise ValueError("Prepared output must be a child of the benchmark cache root")
    if any(out.is_relative_to(root / name) for name in ("sources", "vendor")):
        raise ValueError("Prepared output cannot be inside cached official sources")
    records = acquire_sources(root, fetch=fetch)
    source_manifest = {"version": 1, "sources": records,
                       "privacy_scope": "Local benchmark/verifier material; never a remote teacher input."}
    _immutable_write(root / "source_manifest.json", json_bytes(source_manifest))
    vendor = root / "vendor" / IFEVAL_COMMIT
    vendor_records = [{**record, "path": Path(record["cache_path"]).relative_to(Path("vendor") / IFEVAL_COMMIT).as_posix()}
                      for record in records if record["cache_path"].startswith(f"vendor/{IFEVAL_COMMIT}/")]
    _immutable_write(vendor / "vendor_manifest.json", json_bytes({"version": 1, "commit": IFEVAL_COMMIT,
                                                               "repository": "google-research/google-research",
                                                               "files": vendor_records}))
    ifeval_path = root / f"sources/ifeval/{IFEVAL_COMMIT}/instruction_following_eval/data/input_data.jsonl"
    task_cases = {"ifeval": parse_ifeval(ifeval_path.read_bytes())}
    for task in BBH_TASKS:
        task_cases[task] = parse_bbh((root / f"sources/bbh/{BBH_COMMIT}/bbh/{task}.json").read_bytes(), task)
    expected = {"ifeval": 541, **{task: 250 for task in BBH_TASKS}}
    if {task: len(cases) for task, cases in task_cases.items()} != expected:
        raise ValueError("Pinned official benchmark counts changed")
    tasks, outputs, all_inputs = [], {}, set()
    for task, cases in task_cases.items():
        chosen = select_cases(cases, task_id=task, count=count, seed=seed)
        blob = json_bytes({"version": 1, "task_id": task, "benchmark": chosen[0]["benchmark"], "cases": chosen})
        name = task + ".json"
        hashes = [normalized_input_hash(case["input"]) for case in chosen]
        if all_inputs.intersection(hashes):
            raise ValueError("Selected inputs overlap across benchmark families")
        all_inputs.update(hashes)
        outputs[name] = blob
        tasks.append({"task_id": task, "benchmark": chosen[0]["benchmark"], "cases_path": name,
                      "cases_sha256": sha256(blob), "source_count": len(cases), "selected_count": len(chosen),
                      "selected_case_ids": [case["id"] for case in chosen], "normalized_input_sha256": hashes})
    manifest = {"version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "prepared_before_model_outcomes", "selection_seed": seed,
                "requested_count_per_family": count, "selection_algorithm": "sha256(str(seed)+NUL+task_id+NUL+stable_case_id), ascending; ties by ID",
                "total_selected_inputs": sum(row["selected_count"] for row in tasks),
                "unique_normalized_inputs": len(all_inputs),
                "source_manifest_path": Path(os.path.relpath(root / "source_manifest.json", out)).as_posix(),
                "source_manifest_sha256": sha256(json_bytes(source_manifest)),
                "ifeval_vendor_path": Path(os.path.relpath(vendor, out)).as_posix(),
                "ifeval_vendor_manifest_sha256": sha256((vendor / "vendor_manifest.json").read_bytes()),
                "tasks": tasks, "sources": records,
                "benchmark_scope": "IFEval plus three BBH families; a deterministic pilot, not the full BBH suite.",
                "teacher_access": "Public generic task descriptions only; no benchmark cases, verifier kwargs, labels or evaluation feedback.",
                "prior_benchmark_development": "No benchmark outcomes used to select this pilot. Public benchmark pretraining contamination is not excluded."}
    out.mkdir(parents=True, exist_ok=False)
    for name, blob in outputs.items():
        _immutable_write(out / name, blob)
    _immutable_write(out / "manifest.json", json_bytes(manifest))
    return manifest


def load_prepared(manifest_path):
    """Validate retained commitments and return task-ID to case-list mapping."""
    path = Path(manifest_path).resolve()
    manifest = strict_json(path.read_bytes())
    source_path = (path.parent / manifest["source_manifest_path"]).resolve()
    if sha256(source_path.read_bytes()) != manifest["source_manifest_sha256"]:
        raise ValueError("Source manifest checksum mismatch")
    source_root = source_path.parent
    source_manifest = strict_json(source_path.read_bytes())
    if source_manifest["sources"] != manifest["sources"]:
        raise ValueError("Source manifest records differ")
    for source in source_manifest["sources"]:
        candidate = (source_root / source["cache_path"]).resolve()
        if not candidate.is_relative_to(source_root):
            raise ValueError("Source path escapes benchmark cache")
        data = candidate.read_bytes()
        if len(data) != source["bytes"] or sha256(data) != source["sha256"]:
            raise ValueError("Retained official source checksum mismatch")
    vendor_path = (path.parent / manifest["ifeval_vendor_path"] / "vendor_manifest.json").resolve()
    if sha256(vendor_path.read_bytes()) != manifest["ifeval_vendor_manifest_sha256"]:
        raise ValueError("Verifier manifest checksum mismatch")
    result = {}
    for task in manifest["tasks"]:
        candidate = (path.parent / task["cases_path"]).resolve()
        if not candidate.is_relative_to(path.parent):
            raise ValueError("Case path escapes prepared dataset")
        data = candidate.read_bytes()
        if sha256(data) != task["cases_sha256"]:
            raise ValueError("Prepared benchmark case checksum mismatch")
        payload = strict_json(data)
        cases = _check_cases(payload["cases"])
        if (payload["task_id"] != task["task_id"] or len(cases) != task["selected_count"]
                or [case["id"] for case in cases] != task["selected_case_ids"]
                or [normalized_input_hash(case["input"]) for case in cases] != task["normalized_input_sha256"]):
            raise ValueError("Prepared benchmark identity mismatch")
        if task["task_id"] in result:
            raise ValueError("Duplicate prepared task")
        result[task["task_id"]] = cases
    total = sum(map(len, result.values()))
    unique = {normalized_input_hash(case["input"]) for cases in result.values() for case in cases}
    if total != manifest["total_selected_inputs"] or len(unique) != manifest["unique_normalized_inputs"] or len(unique) != total:
        raise ValueError("Prepared benchmark denominator mismatch")
    return result
