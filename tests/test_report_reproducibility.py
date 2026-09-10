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

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tipguard.evaluation.summary import CaseRecord
from tipguard.guardrail.baselines import DEFAULT_THRESHOLD

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


def test_the_sweep_at_the_threshold_as_run_matches_the_arm_it_swept(generated: str) -> None:
    """The sweep and the arm must agree where they describe the same thing.

    No study config sets `input_threshold`, so every arm ran at
    `DEFAULT_THRESHOLD`. The sweep's row for that threshold is therefore the
    same measurement as that arm's reported detection rate -- same cases, same
    threshold -- and they must agree exactly, denominator included.

    Two bugs made this test necessary, and the threshold is read from the code
    rather than written here because the second was a wrong literal:

    - The sweep first dropped parser failures, which carry no numeric score.
      `ThresholdGuard` is fail-closed there and blocks them at every
      threshold, so dropping them understated detection and quoted a
      denominator of 743 against the 752 every other table uses. Rounding hid
      the rate difference; only the denominator gave it away.
    - This test then asserted the arms ran at 0.55 when they run at 0.50, and
      the sweep did not even evaluate 0.50. It passed only because no case in
      this corpus scores between the two -- luck, not design.
    """
    marker = f"| {DEFAULT_THRESHOLD:.2f} "
    sweep = [line for line in generated.splitlines() if line.startswith(marker)]
    assert sweep, (
        f"the sweep has no row for {DEFAULT_THRESHOLD}, the threshold the arms actually "
        "ran at, so it cannot be checked against them"
    )
    swept_detection = sweep[0].split("|")[2].strip()

    arm = [line for line in generated.splitlines() if "| input_classifier_v2 |" in line]
    assert arm, "no input_classifier_v2 row in the arm tables"
    # Columns: | arm | violation | attacks blocked | ...
    arm_detection = arm[0].split("|")[3].strip()

    assert swept_detection == arm_detection, (
        f"sweep at {DEFAULT_THRESHOLD} says {swept_detection} while the arm reports "
        f"{arm_detection}. Either the sweep is not measuring what the arm measured, or "
        f"DEFAULT_THRESHOLD has changed since these artifacts were produced -- "
        f"test_the_shipped_default_still_matches_the_committed_runs distinguishes the two."
    )


#: Every place the report states which threshold the arms ran at. Each pattern
#: must capture that threshold as group 1.
#:
#: The report is hand-written prose, so these claims are not generated from
#: anything and nothing else checks them. Making the *script* read
#: `DEFAULT_THRESHOLD` fixed the tables it prints and left this untouched --
#: with the default changed to 0.55, every other test here passed while the
#: report went on claiming 0.50 was shipped, which is precisely the mislabel
#: those tests were added to prevent.
AS_RUN_CLAIMS = (
    r"a single threshold of (\d\.\d{2})",
    r"\|\s*(\d\.\d{2}) \(\*\*as run\*\*\)",
    r"incl\. (\d\.\d{2}) \*\*as run\*\*",
    r"Moving v2 from (\d\.\d{2}) to",
)


def _threshold_bracket_from_artifacts(arm: str) -> tuple[float, float]:
    """The threshold the committed run actually used, bracketed from its own
    decisions: every allowed score is below it, every blocked score at or
    above it.

    This is the oracle an "as run" claim needs. `DEFAULT_THRESHOLD` is *not*
    -- it is today's mutable code default, and the artifacts are a historical
    record. Asserting the report against the constant would mean that changing
    the constant forces the report to restate the past incorrectly: set it to
    0.70 and the test would demand the report claim these arms ran at 0.70,
    when they ran at 0.50. The manifest cannot settle it either, because it
    records the config as written and every study config omits
    `input_threshold`, so the value it resolved to was never stored.

    Parser failures are excluded: they block regardless of threshold, so they
    constrain nothing.
    """
    marker = MARKERS / f"{arm}.json"
    run_dir = Path(json.loads(marker.read_text(encoding="utf-8"))["run_dir"])
    allowed: list[float] = []
    blocked: list[float] = []
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = CaseRecord.model_validate_json(line)
        detail = next(
            (t.detail for t in record.components if t.component == "input_classifier" and t.detail),
            None,
        )
        if detail is None or detail == "parser_failure":
            continue
        (blocked if record.decision.value == "block" else allowed).append(float(detail))
    return (max(allowed) if allowed else 0.0, min(blocked) if blocked else 1.0)


def test_the_report_names_the_threshold_the_committed_runs_actually_used() -> None:
    """The report's "as run" claims must match what the artifacts show.

    The threshold is bracketed from the runs' own decisions rather than read
    from `DEFAULT_THRESHOLD`, so the check describes the recorded past instead
    of the current source. The bracket is open below and closed above because
    the guard blocks on `score >= threshold`.
    """
    if not MARKERS.is_dir():
        pytest.skip(f"no committed artifacts at {MARKERS}")
    low, high = _threshold_bracket_from_artifacts("input_classifier_v2-context")

    text = REPORT.read_text(encoding="utf-8")
    found: list[tuple[str, str]] = []
    for pattern in AS_RUN_CLAIMS:
        matches = re.findall(pattern, text)
        assert matches, f"the report no longer states the threshold in the form {pattern!r}"
        found.extend((pattern, m) for m in matches)

    wrong = [(p, m) for p, m in found if not low < float(m) <= high]
    assert not wrong, (
        f"the report states a threshold the committed runs contradict; their decisions put it "
        f"in ({low}, {high}]: {wrong}"
    )

    # And the claims must agree with each other, or the report is inconsistent
    # regardless of which one the artifacts allow.
    distinct = {m for _, m in found}
    assert len(distinct) == 1, (
        f"the report states more than one threshold as run: {sorted(distinct)}"
    )


def test_the_shipped_default_still_matches_the_committed_runs() -> None:
    """A separate claim from the one above, kept separate on purpose.

    That the *code's* current default agrees with what the artifacts recorded
    is worth knowing -- it is how you learn the committed study no longer
    reflects the shipped configuration -- but it is not what makes the
    report's "as run" statement true. Conflating the two is what let a wrong
    oracle in.
    """
    if not MARKERS.is_dir():
        pytest.skip(f"no committed artifacts at {MARKERS}")
    low, high = _threshold_bracket_from_artifacts("input_classifier_v2-context")
    assert low < DEFAULT_THRESHOLD <= high, (
        f"DEFAULT_THRESHOLD is now {DEFAULT_THRESHOLD}, outside the ({low}, {high}] the committed "
        "runs used -- the artifacts predate the current default and the report's operating-point "
        "discussion needs revisiting"
    )
