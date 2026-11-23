"""Download pinned SuperNI tasks without redistributing their complete corpora.

Every response must match its LF download hash. The downloader then restores
recorded CRLF bytes and verifies the source hash used by the experiments.
Dataset instance licenses are recorded in data/superni/sources.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = PROJECT_ROOT / "data/superni/sources.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_records(manifest: dict, group: str) -> list[dict]:
    records = [
        record
        for record in manifest["tasks"]
        if group == "all" or record["group"] == group
    ]
    expected = manifest["groups"][group]["task_count"]
    if len(records) != expected or len({r["path"] for r in records}) != expected:
        raise ValueError("Source manifest group inventory differs: " + group)
    return records


def restore_recorded_bytes(raw: bytes, record: dict) -> bytes:
    """Check the downloaded bytes before applying the documented newline change."""
    if sha256(raw) != record["download_sha256"]:
        raise ValueError("Pinned upstream download hash differs: " + record["path"])
    recorded = raw
    if sha256(recorded) != record["recorded_input_sha256"]:
        if not record["byte_equivalent_after_CRLF_to_LF"] or b"\r\n" in raw:
            raise ValueError(
                "Unsupported recorded-byte transformation: " + record["path"]
            )
        recorded = raw.replace(b"\n", b"\r\n")
    if (
        sha256(recorded) != record["recorded_input_sha256"]
        or len(recorded) != record["bytes"]
    ):
        raise ValueError("Recorded source hash or size differs: " + record["path"])
    return recorded


def fetch_record(record: dict, output: Path) -> str:
    relative = Path(record["path"])
    if (
        relative.parts[0] != "tasks"
        or len(relative.parts) != 2
        or relative.suffix != ".json"
    ):
        raise ValueError("Unexpected source task path")
    target = output / relative.name
    if (
        target.exists()
        and sha256(target.read_bytes()) == record["recorded_input_sha256"]
    ):
        return "Present: " + relative.name
    request = urllib.request.Request(
        record["fetch_url"], headers={"User-Agent": "task-distillation-reproducibility"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read()
    recorded = restore_recorded_bytes(raw, record)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output, prefix=".download-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(recorded)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "Fetched and verified: " + relative.name


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=("reserved", "development", "all"),
        default="all",
        help="Pinned task group to fetch (default: all 24 tasks).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/superni/tasks",
        help="Destination directory for task JSON files.",
    )
    args = parser.parse_args(argv)
    manifest = json.loads(SOURCES.read_text(encoding="utf-8"))
    records = select_records(manifest, args.group)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for record in records:
        print(fetch_record(record, output), flush=True)
    print(f"Ready: {len(records)} {args.group} tasks with recorded source hashes.")
    print("Run python -B run.py to include optional reserved-source validation.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
        print("Data fetch failed: " + str(error))
        raise SystemExit(1)
