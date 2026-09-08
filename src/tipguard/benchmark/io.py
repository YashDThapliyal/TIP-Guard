"""JSONL read/write for benchmark cases."""

import os
import secrets
import stat
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from tipguard.benchmark.schema import BenchmarkCase

#: The mode the temporary file is created with. The kernel applies the process
#: umask to it, so an ordinary `open()` and this produce the same permissions
#: without the umask ever being read or cleared.
TEMPORARY_FILE_MODE = 0o666

#: How many names to try before giving up. A collision needs two 64-bit random
#: suffixes to match, so one retry is already generous.
TEMPORARY_NAME_ATTEMPTS = 8


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


def _unique_suffix() -> str:
    return secrets.token_hex(8)


def _create_temporary(directory: Path, name: str) -> tuple[int, Path]:
    """Create a fresh file beside the destination and return its descriptor.

    Opened with `O_EXCL` so an existing file is never taken over, and with a
    permissive mode so the kernel narrows it by the umask itself. Reading the
    umask instead, which is the usual idiom, would mean clearing it process-wide
    for an instant and letting any concurrent file creation escape it.
    """
    for _ in range(TEMPORARY_NAME_ATTEMPTS):
        candidate = directory / f".{name}.{_unique_suffix()}.tmp"
        try:
            descriptor = os.open(
                candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, TEMPORARY_FILE_MODE
            )
        except FileExistsError:
            continue
        return descriptor, candidate
    raise OSError(f"{directory}: cannot create a temporary file for {name}")


def _match_destination_mode(temporary: Path, destination: Path) -> None:
    """Give the replacement the permissions of the file it replaces.

    Only when there is one: a new destination keeps what the kernel derived
    from the umask, which is what a plain `open()` would have produced.
    """
    try:
        mode = stat.S_IMODE(destination.stat().st_mode)
    except OSError:
        return
    temporary.chmod(mode)


def write_cases(path: Path, cases: Sequence[BenchmarkCase]) -> None:
    """Write the cases as JSONL, replacing `path` only once all of them are on
    disk.

    `tipguard split` rewrites its input in place, so a plain truncating write
    would destroy the dataset if anything failed midway. The lines go to a
    temporary file beside the destination and are moved onto it with an atomic
    rename; a failure leaves the previous file exactly as it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _create_temporary(path.parent, path.name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(case.model_dump_json() + "\n")
        _match_destination_mode(temporary, path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
