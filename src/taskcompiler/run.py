"""Run an experiment from the CLI or GUI."""

from __future__ import annotations

import argparse
from pathlib import Path

from taskcompiler.config import GUIDANCE_TRACK, RULES_TRACK, RunConfig, load_config, new_run_dir, save_config
from taskcompiler.metrics import RunRecorder

STUB_TRACK = "stub_track"


def execute_run(config: RunConfig) -> Path:
    run_dir = new_run_dir(config)
    save_config(config, run_dir / "config.yaml")
    recorder = RunRecorder(run_dir)
    recorder.log(f"run started: track={config.track} dataset={config.dataset}")

    try:
        if config.track == STUB_TRACK:
            from taskcompiler.stub_track import execute as run_stub

            run_stub(config, recorder)
        elif config.track == RULES_TRACK:
            from taskcompiler.rules_track.pipeline import execute as run_rules_track

            run_rules_track(config, recorder, run_dir)
        elif config.track == GUIDANCE_TRACK:
            from taskcompiler.guidance_track.pipeline import execute as run_guidance_track

            run_guidance_track(config, recorder, run_dir)
        else:
            raise ValueError(f"unknown track: {config.track}")
    except Exception as exc:
        recorder.log(f"run failed: {type(exc).__name__}: {exc}")
        recorder.finalize()
        raise
    else:
        recorder.log("run finished")
        recorder.finalize()

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a task-compiler experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a run config YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = execute_run(config)
    print(f"run finished: {run_dir}")


if __name__ == "__main__":
    main()
