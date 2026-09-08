import os
import stat
from pathlib import Path

import pytest

from tests.benchmark.test_schema import make_case
from tipguard.benchmark.io import DatasetError, load_cases, write_cases


def test_roundtrip(tmp_path: Path) -> None:
    cases = (make_case(), make_case(case_id="tip-base64-l1-0002", prompt="other"))
    path = tmp_path / "d.jsonl"
    write_cases(path, cases)
    assert load_cases(path) == cases


def test_invalid_line_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('{"case_id": "x"}\n')
    with pytest.raises(DatasetError, match="line 1"):
        load_cases(path)


def test_malformed_json_line_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text("not json at all\n")
    with pytest.raises(DatasetError, match="line 1"):
        load_cases(path)


def test_non_utf8_content_raises_dataset_error(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(DatasetError, match=str(path)):
        load_cases(path)


def test_directory_path_raises_dataset_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=str(tmp_path)):
        load_cases(tmp_path)


def test_smoke_dataset_loads(repo_root: Path) -> None:
    cases = load_cases(repo_root / "data" / "generated" / "smoke.jsonl")
    assert len(cases) == 6


def test_write_cases_replaces_the_destination_only_once_complete(tmp_path: Path) -> None:
    """`tipguard split` rewrites its input in place, so a failure midway must
    leave the previous dataset intact rather than a truncated one."""
    path = tmp_path / "d.jsonl"
    write_cases(path, (make_case(),))
    before = path.read_text(encoding="utf-8")
    with pytest.raises(AttributeError):
        write_cases(path, [make_case(), object()])  # type: ignore[list-item]
    assert path.read_text(encoding="utf-8") == before
    assert list(tmp_path.iterdir()) == [path]


def test_write_cases_leaves_no_temporary_file_behind_when_the_target_is_a_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "already-a-directory"
    directory.mkdir()
    with pytest.raises(OSError):
        write_cases(directory, (make_case(),))
    assert list(tmp_path.iterdir()) == [directory]


def test_write_cases_keeps_the_destination_file_mode(tmp_path: Path) -> None:
    """The rename must not hand the dataset the temporary file's 0600."""
    path = tmp_path / "d.jsonl"
    write_cases(path, (make_case(),))
    path.chmod(0o644)
    write_cases(path, (make_case(), make_case(case_id="tip-base64-l1-0002", prompt="other")))
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_write_cases_creates_a_file_with_the_usual_mode(tmp_path: Path) -> None:
    path = tmp_path / "new.jsonl"
    write_cases(path, (make_case(),))
    umask = os.umask(0)
    os.umask(umask)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode != 0o600
    assert mode == 0o666 & ~umask


def test_write_cases_never_touches_the_process_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading the umask means clearing it process-wide for an instant, which
    would widen the permissions of any file another thread creates meanwhile."""

    def _forbidden(mask: int) -> int:
        raise AssertionError("write_cases must not query or change the umask")

    monkeypatch.setattr(os, "umask", _forbidden)
    path = tmp_path / "d.jsonl"
    write_cases(path, (make_case(),))
    write_cases(path, (make_case(),))
    assert load_cases(path) == (make_case(),)


def test_write_cases_never_takes_over_an_existing_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Names are exclusive, so a leftover temporary is a clean failure rather
    than a file two writers share."""
    from tipguard.benchmark import io as io_module

    monkeypatch.setattr(io_module, "_unique_suffix", lambda: "fixed")
    path = tmp_path / "d.jsonl"
    write_cases(path, (make_case(),))
    before = path.read_text(encoding="utf-8")
    squatter = tmp_path / ".d.jsonl.fixed.tmp"
    squatter.write_text("someone else's", encoding="utf-8")
    with pytest.raises(OSError, match="cannot create a temporary file"):
        write_cases(path, (make_case(case_id="tip-base64-l1-0002", prompt="other"),))
    assert path.read_text(encoding="utf-8") == before
    assert squatter.read_text(encoding="utf-8") == "someone else's"
