"""JSONL read/write for benchmark cases."""

import json
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
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(BenchmarkCase.model_validate_json(line))
            except (ValidationError, json.JSONDecodeError) as exc:
                raise DatasetError(f"{path} line {number}: {exc}") from exc
    return tuple(cases)


def write_cases(path: Path, cases: Sequence[BenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")
