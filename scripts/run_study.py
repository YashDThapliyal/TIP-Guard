"""Run every study configuration and write one marker per arm.

Sequential on purpose: the response cache is a single SQLite connection, and
a shared cache is what makes a re-run after a fix nearly free.
"""

import json
import sys
import time
from pathlib import Path

from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import ExperimentConfig
from tipguard.evaluation.runner import run_experiment

OUT = Path("reports/study")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    configs = sorted(Path("experiments/study").glob("*.yaml"))
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    if only:
        configs = [c for c in configs if only in c.name]
    print(f"{len(configs)} configurations", flush=True)
    for index, path in enumerate(configs, 1):
        config = load_yaml_model(path, ExperimentConfig)
        marker = OUT / f"{path.stem}.json"
        if marker.exists():
            print(f"[{index}/{len(configs)}] {path.stem}: already done", flush=True)
            continue
        started = time.time()
        try:
            artifacts = run_experiment(config, path)
        except Exception as exc:
            print(
                f"[{index}/{len(configs)}] {path.stem}: FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        marker.write_text(
            json.dumps({"run_dir": str(artifacts.run_dir), "config": path.name}, indent=2)
        )
        elapsed = time.time() - started
        print(
            f"[{index}/{len(configs)}] {path.stem}: {elapsed:.0f}s -> {artifacts.run_dir}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
