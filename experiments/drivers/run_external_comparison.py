"""Run an existing external adapter with a durable model-call audit.

This changes transport accounting only: automatic HTTP retries are disabled,
and each model request and response is retained. Prompts, selection, scoring,
and compiler behavior remain in the existing guidance-track implementations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from requests.adapters import HTTPAdapter

from taskcompiler.config import load_config, new_run_dir, save_config
from taskcompiler.guidance_track.pipeline import execute
from taskcompiler.metrics import RunRecorder
from taskcompiler.student import StudentModel
from taskcompiler.teacher import TeacherClient


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def install_audit(path: Path) -> None:
    """Record physical requests without logging client credentials."""
    sequence = 0

    def append(event: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()

    for client, method_name, role in (
        (TeacherClient, "call", "teacher"),
        (StudentModel, "ask", "student"),
    ):
        original_init = client.__init__
        original_call = getattr(client, method_name)
        signature = inspect.signature(original_call)

        def init(self, *args, _init=original_init, _role=role, **kwargs):
            if _role == "teacher":
                kwargs["max_retries"] = 0
            _init(self, *args, **kwargs)
            # Student identity resolution is a read-only GET. All inference
            # POSTs use this zero-retry adapter and one charged attempt.
            if hasattr(self, "_session"):
                for scheme in ("http://", "https://"):
                    self._session.mount(scheme, HTTPAdapter(max_retries=0))

        def call(self, *args, _call=original_call, _sig=signature,
                 _role=role, **kwargs):
            nonlocal sequence
            sequence += 1
            call_id = sequence
            bound = _sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            request = {k: v for k, v in bound.arguments.items() if k != "self"}
            append({"event": "started", "call_id": call_id, "role": _role,
                    "at_utc": utc_now(), "request": request,
                    "model_identity": dict(self.identity)})
            started = time.monotonic()
            try:
                response = _call(self, *args, **kwargs)
            except Exception as exc:
                append({"event": "failed", "call_id": call_id, "role": _role,
                        "at_utc": utc_now(), "elapsed_seconds": time.monotonic() - started,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "response_usage_unknown": True})
                raise
            append({"event": "completed", "call_id": call_id, "role": _role,
                    "at_utc": utc_now(), "elapsed_seconds": time.monotonic() - started,
                    "response": asdict(response), "model_identity": dict(self.identity)})
            return response

        client.__init__ = init
        setattr(client, method_name, call)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    config_path = args.config.resolve()
    relative = config_path.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if protocol["config_sha256"].get(relative) != digest:
        raise ValueError("Config is absent from the frozen protocol or has changed")
    for relative_source, expected in protocol["source_sha256"].items():
        if hashlib.sha256((ROOT / relative_source).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {relative_source}")
    config = load_config(config_path)
    run_dir = new_run_dir(config)
    save_config(config, run_dir / "config.yaml")
    (run_dir / "artifacts" / "frozen_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8")
    install_audit(run_dir / "artifacts" / "model_calls.jsonl")
    recorder = RunRecorder(run_dir)
    recorder.log(f"audited run started: {run_dir.relative_to(ROOT)}")
    try:
        execute(config, recorder, run_dir)
    except Exception as exc:
        recorder.log(f"run failed: {type(exc).__name__}: {exc}")
        raise
    else:
        recorder.log("run finished")
    finally:
        recorder.finalize()
    print(f"run finished: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
