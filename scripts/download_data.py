"""Download and cache the project benchmarks."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests
from datasets import load_dataset

from taskcompiler.config import DATA_DIR

CONTRACT_NLI_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
BBH_RAW_ROOT = (
    "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main"
)
RULEARENA_COMMIT = "3b9e2256294644beca66732babc5e1055855a576"
TEXTGRAD_COMMIT = "75e912e210864b61999781778cdf756d4468120f"


def download_contract_nli(target_dir: Path = DATA_DIR / "contract_nli") -> None:
    if (target_dir / "dev.json").exists():
        print(f"contract_nli already present at {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading ContractNLI from {CONTRACT_NLI_URL}")
    response = requests.get(CONTRACT_NLI_URL, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for name in archive.namelist():
            if name.endswith(".json"):
                data = archive.read(name)
                (target_dir / Path(name).name).write_bytes(data)
    print(f"ContractNLI ready at {target_dir}")


def download_bbh(target_dir: Path = DATA_DIR / "bbh") -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"bbh already present at {target_dir}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("lukaemon/bbh", "logical_deduction_five_objects")
    dataset.save_to_disk(str(target_dir))
    print(f"BBH ready at {target_dir}")


def download_bbh_official(
    target_dir: Path = DATA_DIR / "bbh_official",
) -> None:
    """Cache the official files used by the TextGrad BBH protocol."""
    tasks = ("object_counting", "word_sorting")
    expected = [
        target_dir / folder / f"{task}.{'json' if folder == 'bbh' else 'txt'}"
        for folder in ("bbh", "cot-prompts")
        for task in tasks
    ]
    if all(path.exists() for path in expected):
        print(f"official BBH files already present at {target_dir}")
        return
    for folder in ("bbh", "cot-prompts"):
        (target_dir / folder).mkdir(parents=True, exist_ok=True)
        suffix = "json" if folder == "bbh" else "txt"
        for task in tasks:
            url = f"{BBH_RAW_ROOT}/{folder}/{task}.{suffix}"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            (target_dir / folder / f"{task}.{suffix}").write_bytes(
                response.content
            )
    print(f"official BBH files ready at {target_dir}")


def download_humaneval(target_dir: Path = DATA_DIR / "humaneval") -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"humaneval already present at {target_dir}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("openai/openai_humaneval")
    dataset.save_to_disk(str(target_dir))
    print(f"HumanEval ready at {target_dir}")


def _download_github_archive(
    repository: str,
    commit: str,
    target_dir: Path,
) -> None:
    url = f"https://github.com/{repository}/archive/{commit}.zip"
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts[1:]
            if not parts or ".." in parts:
                continue
            destination = target_dir.joinpath(*parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))


def download_rulearena(
    target_dir: Path = DATA_DIR / "rulearena",
) -> None:
    marker = target_dir / "airline" / "synthesized_problems" / "comp_0.jsonl"
    if marker.exists():
        print(f"RuleArena already present at {target_dir}")
        return
    _download_github_archive(
        "SkyRiver-2000/RuleArena",
        RULEARENA_COMMIT,
        target_dir,
    )
    print(f"RuleArena {RULEARENA_COMMIT} ready at {target_dir}")


def download_textgrad_source(
    target_dir: Path = DATA_DIR / "textgrad_official",
) -> None:
    marker = target_dir / "textgrad" / "optimizer" / "optimizer.py"
    if marker.exists():
        print(f"TextGrad source already present at {target_dir}")
        return
    _download_github_archive(
        "zou-group/textgrad",
        TEXTGRAD_COMMIT,
        target_dir,
    )
    print(f"TextGrad {TEXTGRAD_COMMIT} ready at {target_dir}")


def main() -> None:
    download_contract_nli()
    download_bbh()
    download_bbh_official()
    download_humaneval()
    download_rulearena()
    download_textgrad_source()


if __name__ == "__main__":
    main()
