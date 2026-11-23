"""Resolve archived references without reproducing the old directory layout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def within_root(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes project directory: {relative}")
    return path


def verify_manifest(root: Path = PROJECT_ROOT) -> int:
    """Verify distributed inputs and code before reproducing results."""
    manifest = read_json(root / "provenance/manifest.json")
    for record in manifest["files"]:
        path = within_root(root, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing or changed file: {record['path']}")
        if digest(path) != record["sha256"]:
            raise ValueError(f"File hash differs: {record['path']}")
    return len(manifest["files"])


def prepare_records(root: Path = PROJECT_ROOT) -> None:
    """Reassemble split call logs in an ignored cache, checking every byte."""
    root = root.resolve()
    for record in read_json(root / "provenance/large_files.json")["files"]:
        target = within_root(root, record["cache_path"])
        if target.is_file() and digest(target) == record["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.stem, suffix=".incomplete", delete=False) as output:
            temporary = Path(output.name)
            for part in record["parts"]:
                source = within_root(root, part["path"])
                if digest(source) != part["sha256"]:
                    raise ValueError(f"Split record hash differs: {part['path']}")
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        if temporary.stat().st_size != record["bytes"] or digest(temporary) != record["sha256"]:
            temporary.unlink()
            raise ValueError(f"Reassembled log differs: {record['path']}")
        temporary.replace(target)


def resolve_path(root: Path, relative: str | Path) -> Path:
    """Resolve a clean path or an immutable reference inside a saved record."""
    root = root.resolve()
    candidate = within_root(root, relative)
    key = candidate.relative_to(root).as_posix()
    if key.startswith("task-compiler/"):
        key = key.removeprefix("task-compiler/")
    layout = read_json(root / "provenance/layout.json")
    key = layout["files"].get(key, key)
    for old, new in sorted(layout["directories"].items(), key=lambda item: -len(item[0])):
        if key == old or key.startswith(old + "/"):
            key = new + key[len(old):]
            break
    for record in read_json(root / "provenance/large_files.json")["files"]:
        if key == record["path"]:
            prepare_records(root)
            return within_root(root, record["cache_path"])
    return within_root(root, key)


def run_directory(root: Path, recorded_run_id: str) -> Path:
    if Path(recorded_run_id).name != recorded_run_id or "\\" in recorded_run_id:
        raise ValueError("A recorded run identifier must be a single directory name")
    return resolve_path(root, "runs/" + recorded_run_id)
