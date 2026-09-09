"""Reserving, resuming and completing a run directory.

A run directory has to distinguish five states, because the directory alone
cannot tell them apart and one of the confusions overwrites results: absent,
finished, held by a live run, finished before completion markers existed, and
the wreckage of a run that died part-way. Two marker files carry that
information, and creating the ownership one is the atomic claim.
"""

import contextlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from tipguard.config.loader import ConfigError

__all__ = [
    "RESERVE_ATTEMPTS",
    "RUN_ARTIFACTS",
    "RUN_COMPLETE_MARKER",
    "RUN_OWNER_MARKER",
    "mark_complete",
    "release_run_dir",
    "reserve_run_dir",
]

#: Dropped into a run directory once every artifact is on disk. Its
#: presence is what separates a finished run, which must never be
#: overwritten, from the wreckage of a crashed one, which is resumable.
RUN_COMPLETE_MARKER = ".complete"
#: Written when a run claims a directory and removed when it completes or
#: fails, so a directory carrying one belongs to a run that is either live
#: or was killed outright. Creating it is the atomic claim: two processes
#: drawing the same id race on this file, not on the directory, because
#: generated ids collide at one-second resolution.
RUN_OWNER_MARKER = ".running"
#: What a complete run directory holds, besides the two markers.
RUN_ARTIFACTS = ("results.jsonl", "summary.json", "manifest.json")
#: How many times a reservation restarts after the directory vanishes
#: under it. More than one such collision is a concurrent run creating
#: and removing the same id; a bound turns that into an error rather
#: than a spin.
RESERVE_ATTEMPTS = 3


def _missing_artifacts(run_dir: Path) -> tuple[str, ...]:
    return tuple(name for name in RUN_ARTIFACTS if not (run_dir / name).exists())


def _owner_detail(run_dir: Path) -> str:
    """Whatever the ownership file says about its owner, best effort."""
    try:
        owner = json.loads((run_dir / RUN_OWNER_MARKER).read_text(encoding="utf-8"))
        return f"pid {owner['pid']} on {owner['host']} since {owner['started_at']}"
    except (OSError, ValueError, KeyError, TypeError):
        return "owner unknown"


def _in_progress_error(run_dir: Path) -> ConfigError:
    """Refuse a directory another run holds.

    A leftover ownership file from a process that died outright is
    indistinguishable from a live one, and it is refused the same way on
    purpose: guessing "dead" and being wrong overwrites a running
    experiment's results, which no error message can undo.
    """
    return ConfigError(
        f"another run with this id appears to be in progress: {run_dir} "
        f"({_owner_detail(run_dir)}); if no such run is live, delete that directory "
        "or choose another --run-id"
    )


def _complete_run_error(run_dir: Path) -> ConfigError | None:
    """The refusal a directory carrying the completion marker deserves."""
    if not (run_dir / RUN_COMPLETE_MARKER).exists():
        return None
    missing = _missing_artifacts(run_dir)
    if missing:
        return ConfigError(
            f"run directory {run_dir} is marked complete but is missing "
            f"{', '.join(missing)}; delete it or choose another --run-id"
        )
    return ConfigError(f"run directory already exists and is complete: {run_dir}")


def _legacy_run_error(run_dir: Path) -> ConfigError | None:
    """The refusal a directory holding every artifact but no marker deserves."""
    if _missing_artifacts(run_dir):
        return None
    return ConfigError(
        f"run directory {run_dir} holds every artifact but no completion marker, so it is "
        "a run completed before completion markers existed; delete it or choose another "
        "--run-id"
    )


def _classify_existing_run_dir(run_dir: Path) -> bool:
    """What an existing run directory means. Returns True to resume, else raises.

    Ordered by how much is known: the completion marker is the strongest
    signal, then an ownership file, then the artifacts themselves. Only the
    last case — no marker, nobody holding it, and not everything written — is
    the wreckage of a crashed run, and only that is resumed.
    """
    complete = _complete_run_error(run_dir)
    if complete is not None:
        raise complete
    if (run_dir / RUN_OWNER_MARKER).exists():
        raise _in_progress_error(run_dir)
    legacy = _legacy_run_error(run_dir)
    if legacy is not None:
        raise legacy
    return True


def _confirm_still_claimable(run_dir: Path) -> None:
    """Re-check the directory once the claim is held.

    Classification runs before the claim, so a concurrent run can finish in
    between — and finishing *releases* its ownership file, which is exactly
    what leaves this pass free to claim a directory that is now complete.
    Holding the claim, no other run can reach that state from here, so this
    is both the last moment the check is needed and the first at which its
    answer is stable. Only another run's work can be found: this pass has
    written nothing but the ownership file yet.
    """
    for error in (_complete_run_error(run_dir), _legacy_run_error(run_dir)):
        if error is not None:
            raise error


def _open_owner_file(run_dir: Path) -> TextIO:
    """Create the ownership file, or refuse: exclusive creation is the claim."""
    try:
        return (run_dir / RUN_OWNER_MARKER).open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise _in_progress_error(run_dir) from exc


def _claim_ownership(run_dir: Path) -> None:
    """Take the directory atomically, or refuse.

    `FileNotFoundError` — the directory removed since the caller looked at
    it — is deliberately not handled here. Recovering in place would mean
    recreating the directory, and a tolerant recreate accepts a directory
    another process has meanwhile recreated *and completed*: completion
    releases ownership, so the exclusive open below would succeed against a
    finished run and this one would overwrite it. The caller restarts the
    whole reservation instead, so the new directory is classified from
    scratch.
    """
    with _open_owner_file(run_dir) as handle:
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "started_at": datetime.now(UTC).isoformat(),
                }
            )
        )


def _attempt_reservation(run_dir: Path) -> bool | None:
    """One reservation pass: True to resume, False if fresh, None to start over.

    `mkdir(exist_ok=False)` is what distinguishes the two openings, and it is
    exclusive on every pass including a retry: creating the directory is
    proof it was not there, so claiming it is safe. If it was there, the
    directory is classified before anything is claimed — which is what makes
    a concurrently completed run refused as complete rather than overwritten.
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        created, resumed = True, False
    except FileExistsError as exc:
        if not run_dir.is_dir():
            raise ConfigError(f"run directory path is not a directory: {run_dir}") from exc
        created, resumed = False, _classify_existing_run_dir(run_dir)
    try:
        _claim_ownership(run_dir)
    except FileNotFoundError:
        # Removed between the look and the claim, by a concurrent run
        # cleaning up after itself. Nothing known about it still holds, so
        # the caller starts the pass over rather than patching it up here.
        return None
    except BaseException:
        # A directory this pass created and never claimed would otherwise be
        # left behind, and the next run would announce that it is resuming a
        # run that never started.
        if created:
            _remove_run_dir_if_empty(run_dir)
        raise
    try:
        _confirm_still_claimable(run_dir)
    except BaseException:
        release_run_dir(run_dir)
        raise
    return resumed


def reserve_run_dir(run_dir: Path) -> bool:
    """Claim run_dir, returning True when a crashed run is being resumed.

    Five states, because a directory alone cannot tell them apart and one of
    the confusions loses results. Absent: created and claimed. Marked
    complete: refused — a finished run is never overwritten. Held by an
    ownership file: refused as in progress. Holding every artifact but no
    marker: refused as a run finished before markers existed, which is what
    every run predating this code looks like. Anything else: the partial
    write of a run that died, which is resumed.

    Creating the ownership file with mode "x" is the atomic claim; `mkdir`
    only distinguishes a fresh directory from an existing one. A directory
    that vanishes mid-pass restarts the classification, bounded by
    `RESERVE_ATTEMPTS` so a concurrent run creating and removing the same id
    cannot spin here forever.
    """
    for _ in range(RESERVE_ATTEMPTS):
        outcome = _attempt_reservation(run_dir)
        if outcome is not None:
            return outcome
    raise ConfigError(
        f"run directory {run_dir} kept disappearing while it was being claimed "
        f"({RESERVE_ATTEMPTS} attempts); another process is creating and removing this "
        "run id. Retry, or choose another --run-id"
    )


def _remove_run_dir_if_empty(run_dir: Path) -> None:
    """Drop a run directory nothing was ever written into, if it still exists."""
    with contextlib.suppress(OSError):
        if run_dir.is_dir() and not any(run_dir.iterdir()):
            run_dir.rmdir()


def release_run_dir(run_dir: Path) -> None:
    """Undo the claim after a failed or interrupted run.

    The ownership file goes first, so a retry with the same id sees the
    wreckage rather than a run that looks live; the directory follows if
    nothing was ever written into it. Every step tolerates the directory
    having already vanished, because a concurrent invocation may be doing
    the same cleanup. A process killed outright runs none of this, which is
    exactly when the leftover file should refuse the id.
    """
    with contextlib.suppress(OSError):
        (run_dir / RUN_OWNER_MARKER).unlink(missing_ok=True)
    _remove_run_dir_if_empty(run_dir)


def mark_complete(run_dir: Path) -> None:
    """Declare the run finished, and let go of it.

    Called only once every artifact is on disk: a marker written any earlier
    would declare a half-written run complete. Ownership is released after
    it, so a directory is never both complete and held.
    """
    (run_dir / RUN_COMPLETE_MARKER).write_text("", encoding="utf-8")
    (run_dir / RUN_OWNER_MARKER).unlink(missing_ok=True)
