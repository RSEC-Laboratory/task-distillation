"""Offline regression checks for replay gates and fresh-request accounting."""
from __future__ import annotations

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import core, run_student


class ReplayGateTests(unittest.TestCase):
    def test_repair_must_preserve_every_previously_correct_case(self):
        parser = SimpleNamespace(parse_program=lambda source, task_spec: json.loads(source))
        public = [{"input_record": {"index": index}, "expected_label": "allow"}
                  for index in range(3)]
        execute = lambda program, record: program[record["index"]]
        repair = {"accepted": True, "original_source": '["allow", "deny", "deny"]',
                  "proposed_source": '["allow", "allow", "deny"]'}
        task = {"repair_records": [repair]}
        self.assertEqual(core.check_repairs(task, public, None, parser, execute), 1)
        # Same accuracy with a different correct set is a regression, as is a
        # higher accuracy obtained by losing one previously correct answer.
        for proposed in ('["deny", "allow", "deny"]', '["deny", "allow", "allow"]'):
            with self.subTest(proposed=proposed):
                repair["proposed_source"] = proposed
                with self.assertRaises(core.CheckError):
                    core.check_repairs(task, public, None, parser, execute)

    def test_rendering_requires_complete_and_unambiguous_program_agreement(self):
        domains = {path: {"a", "b", "c"}
                   for path in ("user.kind", "user.group", "context.zone", "action.kind")}
        task = {"representations": {"selected": ["If any input, return allow."]}}
        execute = lambda program, record: "allow"
        self.assertEqual(core.check_rendering(task, domains, None, execute), 81)
        task["representations"]["selected"].append("If user.kind=a, return deny.")
        with self.assertRaises(core.CheckError):
            core.check_rendering(task, domains, None, execute)
        task["representations"]["selected"] = ["If user.kind=a, return allow."]
        with self.assertRaises(core.CheckError):
            core.check_rendering(task, domains, None, execute)


@contextlib.contextmanager
def model_server(digest=None, response_status=200):
    """A loopback fixture: no model process, checkpoint, or external service."""
    state = {"posts": [], "gets": []}
    model, expected_digest = run_student.MODELS["llama"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_json(self, status, value):
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state["gets"].append(self.path)
            if self.path == "/api/version":
                self.send_json(200, {"version": "fixture-only"})
            elif self.path == "/api/tags":
                self.send_json(200, {"models": [{"name": model, "digest": digest or expected_digest}]})
            else:
                self.send_json(404, {"error": "unknown fixture path"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["posts"].append(json.loads(body))
            self.send_json(response_status, {"model": model, "done": True,
                "message": {"role": "assistant", "content": "deny"},
                "prompt_eval_count": 12, "eval_count": 1})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        yield run_student.LocalServer(address, timeout=2), state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class StudentRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        model, digest = run_student.MODELS["llama"]
        self.plan = {"model": model, "required_digest": digest}
        request = {"model": model, "messages": [{"role": "user", "content": "fixture input"}],
                   "stream": False, "think": False,
                   "options": {"temperature": 0, "seed": 0, "num_predict": 16, "num_ctx": 8192}}
        self.selected = [{"task_id": "fixture_task", "case_index": 0, "arm": "raw_labels",
            "source_call_id": "fixture_call", "source_request_hash": "fixture_hash",
            "gold": "deny", "labels": ["allow", "deny"], "request": request}]

    def tearDown(self):
        self.temporary.cleanup()

    def test_fresh_request_and_terminal_record_are_preserved(self):
        with model_server() as (server, state), contextlib.redirect_stdout(io.StringIO()):
            predictions, status, error = run_student.execute_requests(
                server, self.output, self.plan, self.selected)
        self.assertEqual((status, error), ("complete", None))
        self.assertEqual(state["posts"], [self.selected[0]["request"]])
        self.assertTrue(predictions[0]["correct"])
        ledger = [json.loads(line) for line in (self.output / "requests.jsonl").read_text().splitlines()]
        self.assertEqual([row["event"] for row in ledger], ["started", "completed"])
        self.assertTrue((self.output / "server_before.json").is_file())
        self.assertTrue((self.output / "server_after.json").is_file())

    def test_checkpoint_mismatch_prevents_inference(self):
        with model_server(digest="incorrect-checkpoint") as (server, state):
            predictions, status, error = run_student.execute_requests(
                server, self.output, self.plan, self.selected)
        self.assertEqual(status, "incomplete")
        self.assertIn("digest", error)
        self.assertEqual(predictions, [])
        self.assertEqual(state["posts"], [])

    def test_failed_request_is_charged_once_and_stops_the_batch(self):
        selected = self.selected + [dict(self.selected[0], case_index=1)]
        with model_server(response_status=500) as (server, state):
            predictions, status, error = run_student.execute_requests(
                server, self.output, self.plan, selected)
        self.assertEqual(status, "incomplete")
        self.assertIsNotNone(error)
        self.assertEqual(len(state["posts"]), 1)
        self.assertEqual(len(predictions), 1)
        self.assertFalse(predictions[0]["correct"])
        self.assertEqual(predictions[0]["event"], "failed")
        ledger = [json.loads(line) for line in (self.output / "requests.jsonl").read_text().splitlines()]
        self.assertEqual([row["event"] for row in ledger], ["started", "failed"])

    def test_existing_output_is_refused_before_loading_records(self):
        options = SimpleNamespace(root=self.output / "absent-project", output=self.output)
        with self.assertRaisesRegex(run_student.RunError, "already exists"):
            run_student.run(options)

    def test_selected_request_uses_started_prompt_and_ignores_archived_answer(self):
        records = self.output / "evaluation/records/instruction_ablation"
        records.mkdir(parents=True)
        case = {"index": 0, "gold": "deny", "record": {"message": "fixture input"}}
        prepared = [{"cohort": "llama", "task_id": "fixture_task",
                     "labels": ["allow", "deny"], "cases": [case]}]
        (records / "prepared_inputs.json").write_text(json.dumps(prepared), encoding="utf-8")
        primary = {"cohort": "llama", "task_id": "fixture_task", "case_index": 0,
                   "arm": "raw_labels", "sentinel": False, "gold": "deny",
                   "call_id": "fixture_call", "request_hash": "fixture_hash"}
        (records / "predictions.jsonl").write_text(json.dumps(primary) + "\n", encoding="utf-8")
        model, digest = run_student.MODELS["llama"]
        identity = {"model": model, "digest": digest, "backend": "ollama",
                    "temperature": 0, "seed": 0, "context_tokens": 8192, "thinking": False}
        started = {"event": "started", "call_id": "fixture_call", "model_identity": identity,
                   "prompt": f"Input: {case['record']}", "system": "fixture instruction",
                   "request_hash": "fixture_hash", "sentinel": False, "max_answer_tokens": 16}
        completed = {"event": "completed", "call_id": "fixture_call", "raw_output": "allow"}
        ledger = records / "calls.jsonl"
        ledger.write_text(json.dumps(started) + "\n" + json.dumps(completed) + "\n", encoding="utf-8")
        with patch.object(run_student, "resolve_path", return_value=ledger):
            selected, available, hashes = run_student.select_cases(
                self.output, "llama", ["raw_labels"], limit=1)
        self.assertEqual(available, 1)
        self.assertIn("started_request_ledger", hashes)
        self.assertEqual(selected[0]["request"]["messages"], [
            {"role": "system", "content": "fixture instruction"},
            {"role": "user", "content": f"Input: {case['record']}"}])
        self.assertNotIn("raw_output", selected[0])
        with model_server() as (server, state), contextlib.redirect_stdout(io.StringIO()):
            predictions, status, _error = run_student.execute_requests(
                server, self.output, self.plan, selected)
        self.assertEqual(status, "complete")
        self.assertEqual(predictions[0]["raw_output"], "deny")
        self.assertEqual(state["posts"], [selected[0]["request"]])


if __name__ == "__main__":
    unittest.main()
