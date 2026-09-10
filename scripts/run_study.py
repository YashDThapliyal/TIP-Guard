"""Run every study configuration and write one marker per arm.

Sequential on purpose: the response cache is a single SQLite connection, and
a shared cache is what makes a re-run after a fix nearly free.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import ExperimentConfig
from tipguard.evaluation.run_dir import RUN_COMPLETE_MARKER
from tipguard.evaluation.runner import run_experiment
from tipguard.run_id import config_hash

OUT = Path("reports/study")


def _marker_is_current(marker: Path, config: ExperimentConfig) -> tuple[bool, str]:
    """Whether an arm's marker still describes a finished run of *this* config.

    A marker is only a note that an arm was run once. Skipping on its mere
    existence is wrong three ways: the config may have changed since (this
    study edited `tip_guard`'s params mid-sweep), the run directory it points
    at may have been deleted, and that directory may hold a crashed run that
    never wrote its results. Any of those leaves `analyse_study.py` quietly
    reading stale numbers, or omitting the arm, while the driver reports
    success -- the failure mode that corrupts a comparison without ever
    looking like an error.
    """
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"marker unreadable ({type(exc).__name__})"
    recorded_dir = payload.get("run_dir")
    if not isinstance(recorded_dir, str) or not recorded_dir:
        # Checked before constructing the Path: `Path("")` is `Path(".")`,
        # the working directory, which is a real directory -- so a marker
        # naming no run would have been carried past this check and refused
        # later for the wrong reason.
        return False, "marker names no run directory"
    run_dir = Path(recorded_dir)
    if not run_dir.is_dir():
        return False, "run directory is gone"
    if not (run_dir / RUN_COMPLETE_MARKER).exists():
        return False, "run never completed"
    results = run_dir / "results.jsonl"
    if not results.exists():
        return False, "run has no results"
    # Existence is not enough: a run whose results file is empty or holds
    # only blank lines has nothing for `analyse_study.py` to read, and an
    # arm silently contributing zero records to a table is indistinguishable
    # in that table from an arm that was never run.
    try:
        if not any(line.strip() for line in results.read_text(encoding="utf-8").splitlines()):
            return False, "results file is empty"
    except OSError as exc:
        return False, f"results unreadable ({type(exc).__name__})"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return False, "run has no manifest"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"manifest unreadable ({type(exc).__name__})"
    if manifest.get("config_hash") != config_hash(config):
        return False, "config changed since the run"
    # The config hash covers the config, which names the dataset by *path*.
    # Regenerating the benchmark leaves every config byte-identical while
    # making every stored result stale, so the content hash is checked too.
    #
    # Required, not optional. An earlier version skipped this when the
    # manifest carried no digest, which is precisely backwards: a manifest
    # that cannot be checked is not evidence the run is current, and treating
    # absence as agreement lets exactly the stale results this guards against
    # through. Every check here fails closed.
    recorded = manifest.get("dataset_sha256")
    if not recorded:
        return False, "manifest records no dataset digest"
    try:
        current = hashlib.sha256(config.dataset.read_bytes()).hexdigest()
    except OSError as exc:
        return False, f"dataset unreadable ({type(exc).__name__})"
    if recorded != current:
        return False, "dataset changed since the run"
    return True, ""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    configs = sorted(Path("experiments/study").glob("*.yaml"))
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    if only:
        configs = [c for c in configs if only in c.name]
    print(f"{len(configs)} configurations", flush=True)
    failed: list[str] = []
    for index, path in enumerate(configs, 1):
        marker = OUT / f"{path.stem}.json"
        started = time.time()
        try:
            # Loading is inside the guard: a malformed or invalid YAML used to
            # raise before the per-arm `try` and abort the whole sweep, which
            # is exactly the salvage behaviour the rest of this loop provides.
            config = load_yaml_model(path, ExperimentConfig)
            if marker.exists():
                current, reason = _marker_is_current(marker, config)
                if current:
                    print(f"[{index}/{len(configs)}] {path.stem}: already done", flush=True)
                    continue
                print(
                    f"[{index}/{len(configs)}] {path.stem}: re-running ({reason})",
                    flush=True,
                )
                marker.unlink()
            artifacts = run_experiment(config, path)
        except Exception as exc:
            print(
                f"[{index}/{len(configs)}] {path.stem}: FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )
            failed.append(path.stem)
            continue
        marker.write_text(
            json.dumps({"run_dir": str(artifacts.run_dir), "config": path.name}, indent=2)
        )
        elapsed = time.time() - started
        print(
            f"[{index}/{len(configs)}] {path.stem}: {elapsed:.0f}s -> {artifacts.run_dir}",
            flush=True,
        )
    if failed:
        # One arm's failure must not stop the others -- a long study should
        # salvage every arm it can -- but the run as a whole did not succeed,
        # and batch automation reading only the exit status would otherwise
        # treat an incomplete study as a complete one.
        print(f"incomplete: {len(failed)} arm(s) failed: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
