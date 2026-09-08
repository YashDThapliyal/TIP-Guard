"""Composition tables and the dataset digest."""

import hashlib
from pathlib import Path

import pytest

from tests.benchmark.test_schema import make_case
from tipguard.benchmark.io import DatasetError
from tipguard.benchmark.schema import CaseType, Decision, Split
from tipguard.benchmark.stats import dataset_sha256, format_stats


def test_dataset_sha256_matches_the_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_bytes(b"one\ntwo\n")
    assert dataset_sha256(path) == hashlib.sha256(b"one\ntwo\n").hexdigest()


def test_dataset_sha256_reports_an_unreadable_path(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="cannot read file"):
        dataset_sha256(tmp_path)


def _cases() -> tuple[object, ...]:
    benign = make_case(
        case_id="b1",
        case_type=CaseType.BENIGN_TRANSFORMATION,
        policy_id=None,
        expected_decision=Decision.ALLOW,
        protected_value_hash=None,
        transformation="reverse",
        difficulty=2,
        split=Split.TRAIN,
        expected_answer="x",
    )
    return (make_case(), make_case(case_id="t2"), benign)


def test_format_stats_totals_every_table_at_the_case_count() -> None:
    cases = _cases()
    report = format_stats(cases, "deadbeef")  # type: ignore[arg-type]
    assert "total: 3" in report
    assert "sha256: deadbeef" in report
    totals = [line for line in report.splitlines() if line.startswith("| total |")]
    assert len(totals) == 2
    assert all(line.endswith("3 |") for line in totals)


def test_format_stats_crosses_type_against_transformation_and_difficulty() -> None:
    report = format_stats(_cases(), "d")  # type: ignore[arg-type]
    assert "| case type | base64 | reverse | total |" in report
    assert "| tip | 2 | 0 | 2 |" in report
    assert "| benign_transformation | 0 | 1 | 1 |" in report
    assert "| case type | 1 | 2 | total |" in report


def test_format_stats_shows_the_block_and_allow_balance_of_each_split() -> None:
    """Two split conditions carry no allow-side cases, which a totals-only
    table hides; the card quotes this straight from the command."""
    report = format_stats(_cases(), "d")  # type: ignore[arg-type]
    assert "| split | cases | expect block | expect allow |" in report
    assert "| test | 2 | 2 | 0 |" in report
    assert "| train | 1 | 0 | 1 |" in report
