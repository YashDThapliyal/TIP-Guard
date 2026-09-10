"""The report's numbers must be regenerable from the committed artifacts.

This exists because the claim was twice asserted and twice false. The report
first said its tables came from `analyse_study.py` when most of them came from
throwaway shell scripts that were never committed; the correction then missed
two values because the table meant to cover them silently excluded
attack-only transformation families.

A prose claim about reproducibility cannot be checked by reading. This checks
it: every confidence interval printed in the report must appear in the output
of the committed scripts, except for a named allow-list of values that are
deliberately not artifact-derived.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPORT = Path("docs/report.md")
MARKERS = Path("artifacts/study-v1/markers")
SCRIPTS = (Path("scripts/analyse_study.py"), Path("scripts/report_tables.py"))

#: Intervals in the report that no script can regenerate, and why.
#:
#: These are the 40-case-per-type risk-v1/v2 pilot. It ran before the full
#: arms, was never written to a run directory, and its entire value is that it
#: predicted the full arm's result in advance -- back-deriving it from that arm
#: would destroy the only thing it demonstrates. Anything else appearing here
#: is a number in the report that nothing can check, which is a defect.
PILOT_ONLY = {
    "1.00 [1.00-1.00]",
    "0.59 [0.49-0.70]",
    "0.97 [0.93-1.00]",
    "0.29 [0.19-0.39]",
}

#: `\u2013` is EN DASH, written escaped because matching the report's own
#: typography is the point -- it types intervals with en-dashes while the
#: scripts print hyphens.
EN_DASH = "\u2013"
_INTERVAL = re.compile(r"\d\.\d{2} \[\d\.\d{2}[-\u2013]\d\.\d{2}\]")


def _normalise(value: str) -> str:
    """The report types intervals with en-dashes; the scripts print hyphens."""
    return value.replace(EN_DASH, "-")


@pytest.fixture(scope="module")
def generated() -> str:
    if not MARKERS.is_dir():
        pytest.skip(f"no committed artifacts at {MARKERS}")
    out = []
    for script in SCRIPTS:
        result = subprocess.run(
            [sys.executable, str(script), str(MARKERS)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{script} failed:\n{result.stderr}"
        out.append(result.stdout)
    return "\n".join(out)


def test_every_reported_interval_is_regenerable(generated: str) -> None:
    reported = {_normalise(v) for v in _INTERVAL.findall(REPORT.read_text(encoding="utf-8"))}
    assert reported, "found no intervals in the report; the pattern is probably wrong"
    missing = sorted(v for v in reported if v not in generated and v not in PILOT_ONLY)
    assert not missing, (
        "the report quotes intervals that no committed script reproduces, so a reader "
        f"cannot check them: {missing}"
    )


def test_the_pilot_allow_list_has_not_gone_stale(generated: str) -> None:
    """If a pilot value becomes regenerable, the allow-list should shrink.

    Otherwise the exemption quietly grows into a place to hide unverifiable
    numbers.
    """
    now_covered = sorted(v for v in PILOT_ONLY if v in generated)
    assert not now_covered, (
        f"these are now regenerable and should be removed from PILOT_ONLY: {now_covered}"
    )


def test_the_report_names_its_exceptions() -> None:
    """The two tables that are not artifact-derived must say so in the report,
    not only in this test file."""
    text = REPORT.read_text(encoding="utf-8")
    assert "not regenerable from `artifacts/study-v1/`" in text
    assert "report_tables.py" in text


def test_both_scripts_are_referenced_by_the_artifact_readme() -> None:
    readme = Path("artifacts/README.md")
    if not readme.exists():
        pytest.skip("no artifact readme")
    text = readme.read_text(encoding="utf-8")
    for script in SCRIPTS:
        assert script.name in text, f"{script.name} is not documented for a reader"
