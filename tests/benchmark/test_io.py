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


def test_smoke_dataset_loads(repo_root: Path) -> None:
    cases = load_cases(repo_root / "data" / "generated" / "smoke.jsonl")
    assert len(cases) == 6
