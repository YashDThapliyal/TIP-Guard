"""JSONL read/write for benchmark cases."""

import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from tipguard.benchmark.schema import BenchmarkCase


class DatasetError(ValueError):
    """Raised when a dataset file cannot be parsed or validated."""


def load_cases(path: Path) -> tuple[BenchmarkCase, ...]:
    if not path.is_file():
        raise DatasetError(f"{path}: file not found")
    cases: list[BenchmarkCase] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    cases.append(BenchmarkCase.model_validate_json(line))
                except ValidationError as exc:
                    raise DatasetError(f"{path} line {number}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{path}: cannot read file: {exc}") from exc
    return tuple(cases)


def write_cases(path: Path, cases: Sequence[BenchmarkCase]) -> None:
    """Write the cases as JSONL, replacing `path` only once all of them are on
    disk.

    `tipguard split` rewrites its input in place, so a plain truncating write
    would destroy the dataset if anything failed midway. The lines go to a
    temporary file beside the destination and are moved onto it with an atomic
    rename; a failure leaves the previous file exactly as it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for case in cases:
                handle.write(case.model_dump_json() + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
