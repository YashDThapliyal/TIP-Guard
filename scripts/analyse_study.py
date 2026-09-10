"""Turn the study's run directories into the tables the report quotes.

Reads `reports/study/*.json` markers, loads each run's records, and prints
Markdown. Every rate carries a 95% bootstrap interval and its sample size,
and a gap is only called a result when the intervals do not overlap.
"""

import json
from collections import defaultdict
from pathlib import Path

from tipguard.evaluation.metrics import DefenceMetrics, compute_metrics, separates
from tipguard.evaluation.summary import CaseRecord

STUDY = Path("reports/study")

#: The order arms appear in every table: cheapest first, so a reader sees
#: what each added component bought over the one above it.
ARM_ORDER = (
    "no_defense",
    "keyword_filter",
    "pattern_detector",
    "input_classifier",
    "input_classifier_v2",
    "input_output_classifier",
    "tip_guard_no_canon",
    "tip_guard",
)

#: The only pair that isolates canonicalization. `tip_guard` differs from
#: `input_classifier_v2` in three ways at once -- canonicalization, a second
#: input-side classifier, and an output guard -- so a difference between
#: those two cannot be attributed to any one of them. `tip_guard_no_canon` is
#: `tip_guard` with the decoders and the LLM canonicalizer switched off and
#: nothing else changed, so this pair differs in canonicalization alone and
#: is what the study's primary question is answered from.
CANON_ABLATION = ("tip_guard", "tip_guard_no_canon")

#: The strongest conventional-filtering arm, for the separate and weaker
#: claim of whole-system comparison. Both run the calibrated risk prompt, so
#: this is not confounded by the v1 prompt defect -- but it remains a
#: comparison of two different systems, not of one component.
CONVENTIONAL_BASELINE = "input_output_classifier"


def load_records(run_dir: Path) -> list[CaseRecord]:
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return [CaseRecord.model_validate_json(line) for line in lines if line.strip()]


def collect() -> dict[tuple[str, str], DefenceMetrics]:
    """Metrics keyed by (condition, arm), for every marker that has a run."""
    found: dict[tuple[str, str], DefenceMetrics] = {}
    for marker in sorted(STUDY.glob("*.json")):
        payload = json.loads(marker.read_text(encoding="utf-8"))
        run_dir = Path(payload["run_dir"])
        if not (run_dir / "results.jsonl").exists():
            continue
        stem = marker.stem
        condition = "context" if "-context" in stem else "forbidden"
        arm = stem.split("-context")[0].split("-forbidden")[0]
        if stem.endswith("-haiku"):
            arm = f"{arm} (haiku)"
        found[(condition, arm)] = compute_metrics(arm, load_records(run_dir))
    return found


def table(metrics: dict[tuple[str, str], DefenceMetrics], condition: str) -> str:
    rows = [
        "| Defence | Violation rate | Attacks blocked | False positives | Benign accuracy | Cost |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    arms = [a for a in ARM_ORDER if (condition, a) in metrics]
    arms += sorted(a for (c, a) in metrics if c == condition and a not in ARM_ORDER)
    for arm in arms:
        m = metrics[(condition, arm)]
        rows.append(
            f"| {arm} | {m.violation_rate.format()} | {m.detection_rate.format()} "
            f"| {m.false_positive_rate.format()} | {m.benign_accuracy.format()} "
            f"| ${m.cost_usd:.2f} |"
        )
    return "\n".join(rows)


def findings(metrics: dict[tuple[str, str], DefenceMetrics]) -> list[str]:
    """Only claims the intervals support."""
    out: list[str] = []
    for condition in ("context", "forbidden"):
        base = metrics.get((condition, "no_defense"))
        if base is None:
            continue
        out.append(f"- Undefended, {condition}: violation rate {base.violation_rate.format()}.")
        for arm in ARM_ORDER[1:]:
            m = metrics.get((condition, arm))
            if m is None:
                continue
            verdict = (
                "reduces violations"
                if separates(m.violation_rate, base.violation_rate)
                and m.violation_rate.value < base.violation_rate.value
                else "no separable change in violations"
            )
            out.append(f"  - {arm}: {verdict}; false positives {m.false_positive_rate.format()}.")
    out.extend(_canonicalization_claim(metrics))
    tip = metrics.get(("context", "tip_guard"))
    best = metrics.get(("context", CONVENTIONAL_BASELINE))
    if tip and best:
        if separates(tip.false_positive_rate, best.false_positive_rate):
            direction = (
                "higher"
                if tip.false_positive_rate.value > best.false_positive_rate.value
                else "lower"
            )
            out.append(
                f"- tip_guard's false-positive rate is separably {direction} than "
                f"{CONVENTIONAL_BASELINE}'s."
            )
        else:
            out.append(
                f"- tip_guard and {CONVENTIONAL_BASELINE} do not separate on false positives."
            )
        if separates(tip.violation_rate, best.violation_rate):
            direction = (
                "higher" if tip.violation_rate.value > best.violation_rate.value else "lower"
            )
            out.append(
                f"- tip_guard's violation rate is separably {direction} than "
                f"{CONVENTIONAL_BASELINE}'s."
            )
        else:
            out.append(f"- tip_guard and {CONVENTIONAL_BASELINE} do not separate on violations.")
    return out


def _canonicalization_claim(metrics: dict[tuple[str, str], DefenceMetrics]) -> list[str]:
    """What the ablation pair supports about canonicalization itself.

    This is the study's primary question, and it is the only comparison
    entitled to answer it: the two arms differ in canonicalization and in
    nothing else.
    """
    out: list[str] = []
    full_name, ablated_name = CANON_ABLATION
    for condition in ("context", "forbidden"):
        full = metrics.get((condition, full_name))
        ablated = metrics.get((condition, ablated_name))
        if full is None or ablated is None:
            continue
        out.append(f"- Canonicalization, isolated ({condition}):")
        for label, a, b in (
            ("violations", full.violation_rate, ablated.violation_rate),
            ("attacks blocked", full.detection_rate, ablated.detection_rate),
            ("false positives", full.false_positive_rate, ablated.false_positive_rate),
            ("benign accuracy", full.benign_accuracy, ablated.benign_accuracy),
        ):
            if separates(a, b):
                direction = "higher" if a.value > b.value else "lower"
                verdict = f"separably {direction} with canonicalization"
            else:
                verdict = "no separable difference"
            out.append(f"    - {label}: {verdict} ({a.format()} vs {b.format()})")
    return out


def per_family(condition: str, arm: str) -> str:
    """Where a defence fails, by transformation family."""
    marker = STUDY / f"{arm}-{condition}.json"
    if not marker.exists():
        return ""
    run_dir = Path(json.loads(marker.read_text(encoding="utf-8"))["run_dir"])
    if not (run_dir / "results.jsonl").exists():
        return ""
    buckets: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in load_records(run_dir):
        if record.case_type == "tip":
            buckets[record.transformation].append(record)
    rows = ["| Family | Attacks blocked | n |", "| --- | --- | --- |"]
    for family in sorted(buckets):
        records = buckets[family]
        blocked = sum(1 for r in records if r.decision.value == "block")
        rows.append(f"| {family} | {blocked / len(records):.2f} | {len(records)} |")
    return "\n".join(rows)


def main() -> int:
    metrics = collect()
    if not metrics:
        print("no completed runs yet")
        return 1
    print(f"# Study results\n\n{len(metrics)} completed arms.\n")
    for condition in ("context", "forbidden"):
        if any(c == condition for c, _ in metrics):
            print(f"## Condition: {condition}\n")
            print(table(metrics, condition), "\n")
    print("## Findings the intervals support\n")
    for line in findings(metrics):
        print(line)
    families = per_family("context", "tip_guard")
    if families:
        print("\n## tip_guard, attacks blocked by family (context)\n")
        print(families)
    total = sum(m.cost_usd for m in metrics.values())
    print(f"\nTotal spend across completed arms: ${total:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
