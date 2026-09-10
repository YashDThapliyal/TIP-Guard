"""Regenerate every table in `docs/report.md` that comes from run artifacts.

`analyse_study.py` produces the two headline condition tables and the findings
list. Several other tables in the report -- the threshold sweep, the blocking
stage attribution, the per-family comparisons, and the H1/H4/H5 hypothesis
tables -- were originally computed with throwaway scripts, which meant the
report contained numbers no committed code could reproduce. This script is
those computations, kept, so that every artifact-derived number in the report
can be checked:

    uv run python scripts/report_tables.py artifacts/study-v1/markers

One table in the report is deliberately *not* here: the 40-case-per-type
risk-v1/v2 pilot. That pilot ran before the full arms, was never written to a
run directory, and its point is that it predicted the full result in advance --
so it cannot be, and should not be, back-derived from the arms it predicted.
The report says so where it appears.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from tipguard.evaluation.metrics import Rate, separates, wilson_rate
from tipguard.evaluation.summary import CaseRecord

MARKERS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/study")

#: Prohibited case types. Both expect a block, so both belong in attack rates.
ATTACKS = ("tip", "direct")
BENIGN = ("benign_transformation", "hard_negative")

#: The split holding transformation families withheld from every other split.
HELDOUT_FAMILIES = "heldout_transformation"


def load(arm: str) -> list[CaseRecord]:
    marker = MARKERS / f"{arm}.json"
    run_dir = Path(json.loads(marker.read_text(encoding="utf-8"))["run_dir"])
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return [CaseRecord.model_validate_json(line) for line in lines if line.strip()]


def _classifier_score(record: CaseRecord, component: str) -> float | None:
    """The risk score a component recorded, or None when it did not run or
    its reply failed to parse (the detail then reads `parser_failure`)."""
    for trace in record.components:
        if trace.component == component and trace.detail:
            try:
                return float(trace.detail)
            except ValueError:
                return None
    return None


def header(title: str) -> None:
    print(f"\n## {title}\n")


def table(columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def tip_versus_direct() -> None:
    """H1: are TIP prompts more effective than simply asking?"""
    header("H1 -- TIP versus direct requests (undefended)")
    rows = []
    for arm, label in (
        ("no_defense-context", "gpt-4o-mini, context"),
        ("no_defense-forbidden", "gpt-4o-mini, forbidden"),
        ("no_defense-context-haiku", "haiku, context"),
        ("no_defense-forbidden-haiku", "haiku, forbidden"),
    ):
        records = load(arm)
        tip = wilson_rate([r.leaked for r in records if r.case_type.value == "tip"])
        direct = wilson_rate([r.leaked for r in records if r.case_type.value == "direct"])
        rows.append(
            (label, tip.format(), direct.format(), "yes" if separates(tip, direct) else "no")
        )
    table(("Arm", "TIP cases", "Direct cases", "Separates?"), rows)


def by_difficulty() -> None:
    """H4: does undefended vulnerability peak at difficulty 2-3?"""
    header("H4 -- undefended violation rate by difficulty")
    per_arm: dict[str, dict[int, Rate]] = {}
    for arm in ("no_defense-context", "no_defense-context-haiku"):
        buckets: dict[int, list[bool]] = defaultdict(list)
        for r in load(arm):
            if r.case_type.value in ATTACKS:
                buckets[r.difficulty].append(r.leaked)
        per_arm[arm] = {d: wilson_rate(buckets[d]) for d in sorted(buckets)}
    levels = sorted(per_arm["no_defense-context"])
    table(
        ("Difficulty", "gpt-4o-mini, context", "haiku, context"),
        [
            (
                str(d),
                per_arm["no_defense-context"][d].format(),
                per_arm["no_defense-context-haiku"][d].format(),
            )
            for d in levels
        ],
    )


def generalization_gap() -> None:
    """H5: does canonicalization narrow the gap on unseen families?"""
    header("H5 -- detection on known versus unseen transformation families")
    rows = []
    for arm in ("input_classifier_v2-context", "tip_guard_no_canon-context", "tip_guard-context"):
        records = [r for r in load(arm) if r.case_type.value in ATTACKS]
        known = wilson_rate(
            [r.decision.value == "block" for r in records if r.split.value != HELDOUT_FAMILIES]
        )
        unseen = wilson_rate(
            [r.decision.value == "block" for r in records if r.split.value == HELDOUT_FAMILIES]
        )
        rows.append(
            (
                f"`{arm.removesuffix('-context')}`",
                known.format(),
                unseen.format(),
                f"{known.value - unseen.value:+.3f}",
            )
        )
    table(("Arm", "Known families", "Unseen families", "Gap"), rows)


def threshold_sweep() -> None:
    """The operating-point curve, recovered from stored classifier scores.

    Only detection and false positives can be swept. A case that was blocked
    has no stored model response, so its violation outcome at a lower
    threshold is unknown and the violation rate cannot be recomputed.
    """
    for arm in ("input_classifier_v2-context", "input_classifier-context"):
        records = load(arm)
        scored = [
            (s, r.case_type.value)
            for r in records
            if (s := _classifier_score(r, "input_classifier")) is not None
        ]
        failures = sum(
            1
            for r in records
            if any(t.component == "input_classifier" and t.detail for t in r.components)
        ) - len(scored)
        attacks = [s for s, t in scored if t in ATTACKS]
        benign = [s for s, t in scored if t in BENIGN]
        header(
            f"Threshold sweep -- `{arm.removesuffix('-context')}`, context "
            f"(attacks={len(attacks)}, benign={len(benign)}, parser failures={failures})"
        )
        table(
            ("Threshold", "Attacks blocked", "False positives"),
            [
                (
                    f"{t:.2f}",
                    wilson_rate([s >= t for s in attacks]).format(),
                    wilson_rate([s >= t for s in benign]).format(),
                )
                for t in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
            ],
        )


def blocking_stage() -> None:
    """Which pipeline stage first blocked each attack.

    The table that shows canonicalization's one measurable effect: without it,
    some attacks are caught only after the main model has already answered.
    """
    header("Blocking stage attribution (context, attacks only)")
    arms = ("tip_guard-context", "tip_guard_no_canon-context")
    counts: dict[str, dict[str, int]] = {}
    totals: dict[str, tuple[int, int]] = {}
    for arm in arms:
        records = [r for r in load(arm) if r.case_type.value in ATTACKS]
        blocked = [r for r in records if r.decision.value == "block"]
        tally: dict[str, int] = defaultdict(int)
        for r in blocked:
            tally[r.reasons[0].split(":")[0] if r.reasons else "(none)"] += 1
        counts[arm] = dict(tally)
        totals[arm] = (len(blocked), len(records))
    stages = ("original_classifier", "canonical_classifier", "output_guard")
    rows = []
    for stage in stages:
        cells = []
        for arm in arms:
            n = counts[arm].get(stage, 0)
            cells.append(f"{n} ({n / totals[arm][1]:.1%})")
        rows.append((f"`{stage}`", *cells))
    rows.append(("**Total blocked**", *(f"**{totals[a][0]} / {totals[a][1]}**" for a in arms)))
    table(("Blocking stage", "`tip_guard`", "`tip_guard_no_canon`"), rows)


def capability_confound() -> None:
    """Violation rate beside the model's ability to do the same transformation.

    Benign cases carry gold answers, so benign accuracy per family measures
    whether the model can perform the transformation at all -- which is what
    makes the confound measurable rather than speculative.
    """
    header("Violation rate versus benign accuracy, by family (undefended, context)")
    data: dict[str, dict[str, dict[str, Rate]]] = {}
    for arm in ("no_defense-context", "no_defense-context-haiku"):
        leak: dict[str, list[bool]] = defaultdict(list)
        skill: dict[str, list[bool]] = defaultdict(list)
        for r in load(arm):
            if r.case_type.value in ATTACKS:
                leak[r.transformation].append(r.leaked)
            if r.case_type.value == "benign_transformation" and r.answer_correct is not None:
                skill[r.transformation].append(bool(r.answer_correct))
        shared = sorted(set(leak) & set(skill))
        data[arm] = {
            "leak": {f: wilson_rate(leak[f]) for f in shared},
            "skill": {f: wilson_rate(skill[f]) for f in shared},
        }
    families = sorted(
        data["no_defense-context"]["leak"],
        key=lambda f: -data["no_defense-context"]["leak"][f].value,
    )
    table(
        (
            "Family",
            "gpt-4o-mini violation",
            "gpt-4o-mini benign accuracy",
            "haiku violation",
            "haiku benign accuracy",
        ),
        [
            (
                f,
                data["no_defense-context"]["leak"][f].format(),
                data["no_defense-context"]["skill"][f].format(),
                data["no_defense-context-haiku"]["leak"][f].format(),
                data["no_defense-context-haiku"]["skill"][f].format(),
            )
            for f in families
        ],
    )
    # The correlation the report quotes, with its own caveat attached.
    import statistics

    for arm, label in (
        ("no_defense-context", "gpt-4o-mini"),
        ("no_defense-context-haiku", "haiku"),
    ):
        fams = sorted(data[arm]["leak"])
        xs = [data[arm]["skill"][f].value for f in fams]
        ys = [data[arm]["leak"][f].value for f in fams]
        print(
            f"\n{label}: Pearson r between benign accuracy and violation rate = "
            f"{statistics.correlation(xs, ys):+.2f} over {len(fams)} families"
        )


def undefended_by_family() -> None:
    """Undefended violation rate per family, for every family.

    Distinct from the capability table below, which can only cover families
    that have benign counterparts to measure skill on. `riddle` and `indirect`
    are attack-only, and they are exactly the families where haiku is most
    vulnerable -- so leaving them out would drop the evidence for the claim
    that the two models fail on different attack classes.
    """
    header("Undefended violation rate by family (context)")
    per_arm: dict[str, dict[str, Rate]] = {}
    for arm in ("no_defense-context", "no_defense-context-haiku"):
        buckets: dict[str, list[bool]] = defaultdict(list)
        for r in load(arm):
            if r.case_type.value in ATTACKS:
                buckets[r.transformation].append(r.leaked)
        per_arm[arm] = {f: wilson_rate(buckets[f]) for f in buckets}
    families = sorted(
        per_arm["no_defense-context"],
        key=lambda f: -per_arm["no_defense-context"][f].value,
    )
    table(
        ("Family", "gpt-4o-mini", "haiku"),
        [
            (
                f,
                per_arm["no_defense-context"][f].format(),
                per_arm["no_defense-context-haiku"][f].format(),
            )
            for f in families
        ],
    )


def per_family_detection() -> None:
    """Where each defence fails, by transformation family."""
    header("Attacks blocked by family (context)")
    arms = (
        "pattern_detector-context",
        "input_classifier_v2-context",
        "tip_guard_no_canon-context",
        "tip_guard-context",
    )
    per_arm: dict[str, dict[str, list[bool]]] = {}
    families: set[str] = set()
    for arm in arms:
        buckets: dict[str, list[bool]] = defaultdict(list)
        for r in load(arm):
            if r.case_type.value in ATTACKS:
                buckets[r.transformation].append(r.decision.value == "block")
        per_arm[arm] = buckets
        families |= set(buckets)
    rows = []
    for family in sorted(families):
        cells = []
        for arm in arms:
            sample = per_arm[arm].get(family, [])
            cells.append(f"{sum(sample) / len(sample):.2f}" if sample else "-")
        n = len(per_arm[arms[0]].get(family, []))
        rows.append((family, *cells, str(n)))
    table(
        ("Family", *(f"`{a.removesuffix('-context')}`" for a in arms), "n"),
        rows,
    )


def main() -> int:
    if not MARKERS.is_dir():
        print(f"no marker directory at {MARKERS}", file=sys.stderr)
        return 1
    print("# Report tables regenerated from run artifacts")
    print(f"\nSource: `{MARKERS}`")
    tip_versus_direct()
    by_difficulty()
    generalization_gap()
    threshold_sweep()
    blocking_stage()
    undefended_by_family()
    capability_confound()
    per_family_detection()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
