"""Native-text scorers for the bounded general-benchmark pilot.

IFEval runs the unmodified, checksum-pinned official strict/loose implementation.
The declared adaptations are langdetect seed 0 and pinned English NLTK resources.
Verifier/configuration errors raise; they are never converted to model failures.
BBH uses a fixed reference-independent final-line parser, not an official harness.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import random
import re
import sys
import threading
import types

from .datasets import BBH_TASKS, IFEVAL_COMMIT, SOURCE_FILES, strict_json

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VENDOR = ROOT / "data/general_benchmarks/vendor" / IFEVAL_COMMIT
DEPENDENCIES = {"absl-py": "2.3.1", "langdetect": "1.0.9", "nltk": "3.9.2", "immutabledict": "4.2.2"}
NLTK_COMMIT = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
TOKENIZER_SHA256 = {
    "abbrev_types.txt": "92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779",
    "collocations.tab": "8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf",
    "ortho_context.tab": "4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c",
    "sent_starters.txt": "f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac",
}
PRACTICE_INSTRUCTIONS = {
    "keywords:existence": {"keywords"},
    "keywords:forbidden_words": {"forbidden_words"},
    "length_constraints:number_words": {"num_words", "relation"},
    "detectable_format:number_bullet_lists": {"num_bullets"},
    "change_case:english_lowercase": set(),
    "detectable_format:title": set(),
}
_LOCK = threading.RLock()
_MODULES = {}


class VerifierError(ValueError):
    """Invalid verifier configuration, missing resource, or verifier malfunction."""


def _require(condition, message):
    if not condition:
        raise VerifierError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _checked_manifest(root, name, expected, commit):
    try:
        data = (root / name).read_bytes()
        manifest = strict_json(data)
        _require(manifest.get("version") == 1 and manifest.get("commit") == commit,
                 f"Unrecognized {name} version/revision")
        rows = manifest.get("files", [])
        _require(isinstance(rows, list) and len(rows) == len(expected), f"Unexpected {name} file inventory")
        _require({r.get("path") for r in rows} == set(expected), f"Unexpected {name} paths")
        for row in rows:
            relative = row["path"]
            path = (root / relative).resolve()
            _require(path.is_relative_to(root) and path.is_file(), "Verifier path escapes vendor root or is missing")
            blob = path.read_bytes()
            _require(row.get("sha256") == expected[relative] == _digest(blob), f"Verifier checksum mismatch: {relative}")
            _require(row.get("bytes") == len(blob), f"Verifier size mismatch: {relative}")
        return _digest(data)
    except VerifierError:
        raise
    except Exception as exc:
        raise VerifierError(f"Cannot validate {name}: {type(exc).__name__}: {exc}") from exc


def _load_official(root):
    root = Path(root).resolve()
    expected = {r.path: r.sha256 for r in SOURCE_FILES if r.benchmark == "ifeval" and r.path.endswith(".py")}
    vendor_hash = _checked_manifest(root, "vendor_manifest.json", expected, IFEVAL_COMMIT)
    tokenizer_files = {"nltk_data/tokenizers/punkt_tab/english/" + p: h for p, h in TOKENIZER_SHA256.items()}
    tokenizer_hash = _checked_manifest(root, "tokenizer_manifest.json", tokenizer_files, NLTK_COMMIT)
    try:
        versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
        _require(versions == DEPENDENCIES, f"Verifier dependency versions differ: expected {DEPENDENCIES}; found {versions}")
        import langdetect
        import nltk
        package_path = root / "instruction_following_eval"
        with _LOCK:
            if _MODULES:
                _require(_MODULES["root"] == root, "Another IFEval vendor directory is already loaded in this process")
                return _MODULES["evaluation"], _MODULES["registry"], copy.deepcopy(_MODULES["identity"])
            _require(not any(n == "instruction_following_eval" or n.startswith("instruction_following_eval.") for n in sys.modules),
                     "An unverified instruction_following_eval package is already imported")
            package = types.ModuleType("instruction_following_eval")
            package.__path__ = [str(package_path)]
            package.__package__ = "instruction_following_eval"
            sys.modules[package.__name__] = package
            loaded = []
            try:
                for short in ("instructions_util", "instructions", "instructions_registry", "evaluation_lib"):
                    name = "instruction_following_eval." + short
                    spec = importlib.util.spec_from_file_location(name, package_path / (short + ".py"))
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    loaded.append(name)
                    spec.loader.exec_module(module)
                    setattr(package, short, module)
                # Preload only the verified English sentence parameters. No downloads
                # or fallback to mutable user/global NLTK directories during scoring.
                old_paths = list(nltk.data.path)
                try:
                    nltk.data.path[:] = [str(root / "nltk_data")]
                    nltk.tokenize._get_punkt_tokenizer.cache_clear()
                    package.instructions_util._get_sentence_tokenizer()
                finally:
                    nltk.data.path[:] = old_paths
                langdetect.DetectorFactory.seed = 0
            except Exception:
                for name in loaded + [package.__name__]:
                    sys.modules.pop(name, None)
                raise
            identity = {
                "implementation": "native_text_scorers_v1", "ifeval_commit": IFEVAL_COMMIT,
                "vendor_manifest_sha256": vendor_hash, "source_sha256": expected,
                "dependencies": versions, "nltk_resource_commit": NLTK_COMMIT,
                "tokenizer_manifest_sha256": tokenizer_hash, "tokenizer_sha256": tokenizer_files,
                "langdetect_seed": 0,
                "adaptations": ["Set langdetect.DetectorFactory.seed=0 before each score.",
                                "Load English sentence tokenizer only from hash-pinned local resources."],
                "upstream_semantics": "Official internal exception handling, including language-detection fallback, is unchanged.",
            }
            _MODULES.update(root=root, evaluation=package.evaluation_lib,
                            registry=package.instructions_registry, identity=identity)
            return package.evaluation_lib, package.instructions_registry, copy.deepcopy(identity)
    except VerifierError:
        raise
    except Exception as exc:
        raise VerifierError(f"Cannot initialize official IFEval verifier: {type(exc).__name__}: {exc}") from exc


def extract_bbh_answer(task, response):
    """Read the final nonempty line only; accept a bare token or an answer marker.

    Optional markers: 'Answer:', 'Final answer:', 'The answer is', or
    'So the answer is'. One trailing period is allowed. Choices are A/B/C,
    optionally parenthesized; booleans ignore case; counts are canonical unsigned
    integers. No answer is inferred from arbitrary prose or from the reference.
    """
    _require(task in BBH_TASKS, "Unsupported BBH task")
    _require(isinstance(response, str), "Model response must be text")
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return None
    final = re.sub(r"^(?:(?:final\s+)?answer\s*:|(?:so\s+)?the\s+answer\s+is\s*:?)\s*", "", lines[-1], flags=re.IGNORECASE)
    if final.endswith("."):
        final = final[:-1].rstrip()
    if task == "logical_deduction_three_objects":
        if re.fullmatch(r"[A-C]|\([A-C]\)", final, flags=re.IGNORECASE):
            return "(" + final.strip("()").upper() + ")"
    elif task == "boolean_expressions":
        if final.lower() in {"true", "false"}:
            return final.capitalize()
    elif re.fullmatch(r"0|[1-9][0-9]*", final):
        return final
    return None


class BenchmarkEvaluator:
    def __init__(self, ifeval_vendor=None):
        self.vendor = Path(ifeval_vendor) if ifeval_vendor is not None else DEFAULT_VENDOR
        self._official = None
        self._validated = set()

    def _load(self):
        if self._official is None:
            self._official = _load_official(self.vendor)
        return self._official

    def identity(self):
        identity = copy.deepcopy(self._load()[2])
        identity.update(evaluator_sha256=_digest(Path(__file__).read_bytes()),
                        bbh_parser="final_nonempty_line_v1",
                        ifeval_primary="prompt_level_strict",
                        practice_only_constraint_ids=sorted(PRACTICE_INSTRUCTIONS))
        return identity

    def scorer_id(self, task_id):
        if task_id == "ifeval":
            fingerprint = _digest(json.dumps(self.identity(), sort_keys=True).encode())
            return f"ifeval_official_strict:{IFEVAL_COMMIT}:seed0:{fingerprint}"
        _require(task_id in BBH_TASKS, "Unknown benchmark task")
        return "bbh_fixed_final_answer_v1:" + _digest(Path(__file__).read_bytes())

    def validate_case(self, case):
        _require(isinstance(case, dict), "Benchmark case must be an object")
        _require(isinstance(case.get("id"), str) and bool(case["id"].strip()), "Case ID must be nonempty text")
        _require(isinstance(case.get("input"), str) and bool(case["input"].strip()), "Case input must be nonempty text")
        benchmark, task = case.get("benchmark"), case.get("task")
        if benchmark == "bbh":
            _require(task in BBH_TASKS, "Unknown BBH task")
            reference = case.get("reference")
            pattern = {"logical_deduction_three_objects": r"\([A-C]\)",
                       "boolean_expressions": r"True|False", "object_counting": r"0|[1-9][0-9]*"}[task]
            _require(isinstance(reference, str) and re.fullmatch(pattern, reference) is not None,
                     "BBH reference does not follow its canonical answer format")
            return case
        _require(benchmark == "ifeval" and task == "ifeval", "Unknown benchmark/task pairing")
        ids, kwargs = case.get("instruction_id_list"), case.get("kwargs")
        _require(isinstance(ids, list) and bool(ids) and all(isinstance(i, str) for i in ids), "IFEval instruction IDs must be a nonempty list")
        _require(len(set(ids)) == len(ids), "Duplicate IFEval instruction IDs")
        _require(isinstance(kwargs, list) and len(kwargs) == len(ids) and all(isinstance(k, dict) for k in kwargs),
                 "IFEval kwargs must align with instruction IDs")
        _require("key" not in case or type(case["key"]) is int, "IFEval key must be an integer")
        try:
            fingerprint = json.dumps([case["input"], ids, kwargs], sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise VerifierError("IFEval arguments must be finite JSON values") from exc
        if fingerprint in self._validated:
            return case
        _, registry, _ = self._load()
        with _LOCK:
            random_state = random.getstate()
            try:
                for identifier, arguments in zip(ids, kwargs):
                    _require(identifier in registry.INSTRUCTION_DICT, f"Unknown IFEval instruction: {identifier}")
                    instruction = registry.INSTRUCTION_DICT[identifier](identifier)
                    inspect.signature(instruction.build_description).bind(**arguments)
                    for key, value in arguments.items():
                        if key in {"keywords", "forbidden_words"}:
                            _require(isinstance(value, list) and bool(value) and
                                     all(isinstance(v, str) and bool(v.strip()) for v in value),
                                     f"IFEval {key} must be a nonempty list of strings")
                        elif key.startswith("num_") or key in {"frequency", "let_frequency", "capital_frequency", "nth_paragraph"}:
                            _require(type(value) is int and value >= 0, f"IFEval {key} must be a nonnegative integer")
                        else:
                            _require(isinstance(value, str) and bool(value.strip()), f"IFEval {key} must be nonempty text")
                    instruction.build_description(**copy.deepcopy(arguments))
                    resolved = instruction.get_instruction_args()
                    if resolved and "prompt" in resolved:
                        instruction.build_description(prompt=case["input"])
                _require(random.getstate() == random_state,
                         "IFEval arguments invoked random verifier defaults; supply explicit valid parameters")
            except VerifierError:
                raise
            except Exception as exc:
                raise VerifierError(f"Invalid IFEval verifier arguments: {type(exc).__name__}: {exc}") from exc
            finally:
                random.setstate(random_state)
        self._validated.add(fingerprint)
        return case

    def validate_practice_case(self, case):
        self.validate_case(case)
        if case["benchmark"] == "ifeval":
            ids, kwargs = case["instruction_id_list"], case["kwargs"]
            _require(1 <= len(ids) <= 2, "IFEval practice requires one or two constraints")
            for identifier, arguments in zip(ids, kwargs):
                _require(identifier in PRACTICE_INSTRUCTIONS, "Constraint is outside the generated-practice contract")
                _require(set(arguments) == PRACTICE_INSTRUCTIONS[identifier], "Generated verifier kwargs are incomplete or unexpected")
                if identifier.startswith("keywords:"):
                    words = next(iter(arguments.values()))
                    _require(isinstance(words, list) and 1 <= len(words) <= 8 and
                             all(isinstance(w, str) and re.fullmatch(r"[A-Za-z]+", w) for w in words),
                             "Generated keywords must be nonempty lists of simple alphabetic words")
                elif identifier == "length_constraints:number_words":
                    _require(type(arguments["num_words"]) is int and 10 <= arguments["num_words"] <= 60,
                             "Generated word thresholds must be integers between 10 and 60")
                    _require(arguments["relation"] in {"less than", "at least"}, "Invalid generated word-count relation")
                elif identifier == "detectable_format:number_bullet_lists":
                    _require(type(arguments["num_bullets"]) is int and 1 <= arguments["num_bullets"] <= 6,
                             "Generated bullet counts must be integers between 1 and 6")
            reference = case.get("reference")
            _require(isinstance(reference, str) and bool(reference.strip()), "Generated practice needs an exemplar reference")
            from instruction_following_eval.instructions_util import count_words
            _require(count_words(reference) < 100, "Generated IFEval exemplars must contain fewer than 100 words")
            _require(self(case, reference)["correct"], "Generated exemplar fails official strict IFEval verification")
        return case

    def __call__(self, case, response):
        self.validate_case(case)
        _require(isinstance(response, str), "Model response must be text")
        if case["benchmark"] == "bbh":
            answer = extract_bbh_answer(case["task"], response)
            passed = answer is not None and answer == case["reference"]
            return {"passed": passed, "score": float(passed), "correct": passed,
                    "format_valid": answer is not None, "parsed": answer,
                    "parse_error": None if answer is not None else "No unambiguous final-line answer",
                    "metric": "bbh_fixed_answer_exact_match", "native": {"answer_exact_match": passed}}
        evaluation, _, _ = self._load()
        inp = evaluation.InputExample(key=case.get("key", 0), prompt=case["input"],
                                      instruction_id_list=list(case["instruction_id_list"]), kwargs=copy.deepcopy(case["kwargs"]))
        try:
            import langdetect
            with _LOCK:
                langdetect.DetectorFactory.seed = 0
                strict = evaluation.test_instruction_following_strict(inp, {inp.prompt: response})
                loose = evaluation.test_instruction_following_loose(inp, {inp.prompt: response})
            native = self._native(strict.follow_instruction_list, loose.follow_instruction_list)
            _require(bool(strict.follow_all_instructions) == native["prompt_level_strict"] and
                     bool(loose.follow_all_instructions) == native["prompt_level_loose"], "Official verifier returned inconsistent metrics")
        except VerifierError:
            raise
        except Exception as exc:
            raise VerifierError(f"IFEval verifier failed for case {case['id']}: {type(exc).__name__}: {exc}") from exc
        passed = native["prompt_level_strict"]
        return {"passed": passed, "score": float(passed), "correct": passed,
                "format_valid": bool(response.strip()), "parsed": response, "parse_error": None if response.strip() else "Empty response",
                "metric": "ifeval_prompt_level_strict", "native": native}

    @staticmethod
    def _native(strict, loose):
        _require(len(strict) == len(loose) and bool(strict), "Official verifier returned invalid denominators")
        _require(all(type(v) is bool for v in [*strict, *loose]), "Official verifier returned nonboolean judgments")
        return {"prompt_level_strict": all(strict), "prompt_level_loose": all(loose),
                "instruction_level_strict": sum(strict) / len(strict),
                "instruction_level_loose": sum(loose) / len(loose),
                "instruction_count": len(strict), "strict_instruction_passed": sum(strict),
                "loose_instruction_passed": sum(loose),
                "strict_follow_instruction_list": list(strict), "loose_follow_instruction_list": list(loose)}

    def failure_score(self, case, error):
        """Model-call failure, retaining all verifier denominators; not a verifier error."""
        self.validate_case(case)
        native = ({"answer_exact_match": False} if case["benchmark"] == "bbh" else
                  self._native([False] * len(case["instruction_id_list"]), [False] * len(case["instruction_id_list"])))
        return {"passed": False, "correct": False, "score": 0.0, "format_valid": False,
                "parsed": None, "parse_error": str(error), "native": native,
                "metric": "ifeval_prompt_level_strict" if case["benchmark"] == "ifeval" else "bbh_fixed_answer_exact_match"}


def score(case, response, *, ifeval_vendor=None):
    return BenchmarkEvaluator(ifeval_vendor)(case, response)


def validate_case(case, *, ifeval_vendor=None):
    return BenchmarkEvaluator(ifeval_vendor).validate_case(case)


def validate_practice_case(case, *, ifeval_vendor=None):
    return BenchmarkEvaluator(ifeval_vendor).validate_practice_case(case)
