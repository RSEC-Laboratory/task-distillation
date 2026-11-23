"""Integrity and result-comparison checks independent of inference packages."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.__main__ import compare_reference
from evaluation.layout import prepare_records, resolve_path, verify_manifest


class RecordIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "provenance").mkdir()

    def write(self, path, value):
        (self.root / path).write_text(json.dumps(value), encoding="utf-8")

    def test_manifest_rejects_changed_record(self):
        (self.root / "input.json").write_bytes(b"original")
        self.write("provenance/manifest.json", {"files": [{"path": "input.json", "bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}]})
        self.assertEqual(verify_manifest(self.root), 1)
        (self.root / "input.json").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            verify_manifest(self.root)

    def test_parts_reassemble_exact_bytes_and_reject_corruption(self):
        (self.root / "part1").write_bytes(b"first\n")
        (self.root / "part2").write_bytes(b"second\n")
        content = b"first\nsecond\n"
        record = {"path": "records/calls.jsonl", "cache_path": "cache/calls.jsonl", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "parts": [{"path": name, "sha256": hashlib.sha256((self.root / name).read_bytes()).hexdigest()} for name in ("part1", "part2")]}
        self.write("provenance/large_files.json", {"files": [record]})
        self.write("provenance/layout.json", {"files": {"old/calls.jsonl": "records/calls.jsonl"}, "directories": {}})
        prepare_records(self.root)
        self.assertEqual(resolve_path(self.root, "old/calls.jsonl").read_bytes(), content)
        (self.root / "cache/calls.jsonl").unlink()
        (self.root / "part2").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Split record hash differs"):
            prepare_records(self.root)

    def test_reference_resolution_rejects_outside_paths(self):
        with self.assertRaisesRegex(ValueError, "escapes project"):
            resolve_path(self.root, "../outside")


class ReferenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"accuracy": 0.5, "reserved_nlp": {"source_instance_validation": {"tasks_checked": 0, "expected_tasks": 20, "status": "not_complete"}}}

    def test_downloaded_source_coverage_can_increase_without_changing_scores(self):
        actual = deepcopy(self.expected)
        actual["reserved_nlp"]["source_instance_validation"].update(tasks_checked=20, status="passed")
        compare_reference(actual, self.expected)
        self.assertEqual(self.expected["reserved_nlp"]["source_instance_validation"]["tasks_checked"], 0)

    def test_changed_accuracy_is_rejected(self):
        actual = deepcopy(self.expected)
        actual["accuracy"] = 0.6
        with self.assertRaisesRegex(ValueError, "differ from"):
            compare_reference(actual, self.expected)


if __name__ == "__main__":
    unittest.main()
