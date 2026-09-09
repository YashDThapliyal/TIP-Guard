"""The gold subset: which cases a human reads, and what they ruled about them.

Phase 2 requires that at least a tenth of the dataset be read case by case and
labelled correct or not. The sample must be reproducible, or a later reviewer
cannot tell whether they are looking at the same cases, and it must reach into
every corner of the dataset, or a defect confined to one small stratum -- a
single family at a single difficulty -- can never be found. So the sample is
stratified by the tuple that decides how a case is built (its type, its
transformation and its difficulty), takes at least one case from every stratum,
and fills the rest in proportion to stratum size.
"""

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from tipguard.base import FrozenModel
from tipguard.benchmark.io import DatasetError
from tipguard.benchmark.schema import BenchmarkCase

#: The share of the dataset the completion criteria require to be reviewed.
DEFAULT_GOLD_FRACTION = 0.10

#: Fixed so the subset is the same set of cases on every machine.
DEFAULT_GOLD_SEED = 20260907

Verdict = Literal["correct", "label_error", "prompt_error"]

#: (case type, transformation, difficulty) -- what the generator varies.
Stratum = tuple[str, str, int]

VERDICTS: tuple[Verdict, ...] = ("correct", "label_error", "prompt_error")


class GoldReview(FrozenModel):
    """One reviewer's ruling on one sampled case.

    `verdict` separates the two ways a case can be wrong: `label_error` means
    the prompt is fine but `expected_decision` does not follow from the
    canonical intent and the policy, and `prompt_error` means the prompt itself
    does not ask what the case says it asks.
    """

    case_id: str
    reviewer: str
    verdict: Verdict
    note: str


def stratum_of(case: BenchmarkCase) -> Stratum:
    return (case.case_type.value, case.transformation, case.difficulty)


def _stratify(cases: Sequence[BenchmarkCase]) -> dict[Stratum, tuple[BenchmarkCase, ...]]:
    """Group the cases by stratum, in a fixed order and with fixed contents.

    Both the strata and the members within one are sorted, so the sample does
    not depend on the order the dataset happened to be written in.
    """
    groups: defaultdict[Stratum, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        groups[stratum_of(case)].append(case)
    return {key: tuple(sorted(groups[key], key=lambda c: c.case_id)) for key in sorted(groups)}


def _quotas(sizes: Mapping[Stratum, int], fraction: float) -> dict[Stratum, int]:
    """How many cases to draw from each stratum.

    One from every stratum first, so nothing is invisible, then the remainder
    of the overall target handed out by largest remaining proportional share.
    Taking `ceil(fraction * size)` per stratum would be simpler but overshoots
    badly when there are many small strata, and every extra case is one a human
    has to read.

    `fraction` is at most 1, so the target never exceeds the total and there is
    always a stratum with room left while the quotas fall short of it. That is
    why the loop can be a bounded `for` rather than a `while` guarded against
    running out of candidates.
    """
    total = sum(sizes.values())
    target = max(math.ceil(fraction * total), len(sizes))
    quotas = dict.fromkeys(sizes, 1)
    shares = {key: fraction * size for key, size in sizes.items()}
    for _ in range(target - len(sizes)):
        candidates = [key for key in sizes if quotas[key] < sizes[key]]
        quotas[max(candidates, key=lambda key: shares[key] - quotas[key])] += 1
    return quotas


def sample_gold(
    cases: Sequence[BenchmarkCase],
    fraction: float = DEFAULT_GOLD_FRACTION,
    seed: int = DEFAULT_GOLD_SEED,
) -> tuple[BenchmarkCase, ...]:
    """Draw a reproducible stratified subset covering every stratum.

    The input is only read: the returned tuple holds the same frozen cases.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    groups = _stratify(cases)
    if not groups:
        return ()
    quotas = _quotas({key: len(members) for key, members in groups.items()}, fraction)
    sampled: list[BenchmarkCase] = []
    for key, members in groups.items():
        # Seeded per stratum so adding a stratum cannot reshuffle the others.
        rng = random.Random(f"{seed}:{key}")
        sampled.extend(rng.sample(members, quotas[key]))
    return tuple(sorted(sampled, key=lambda case: case.case_id))


def load_reviews(path: Path) -> tuple[GoldReview, ...]:
    """Read a JSONL review file, failing the way the dataset loader does."""
    if not path.is_file():
        raise DatasetError(f"{path}: file not found")
    reviews: list[GoldReview] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    reviews.append(GoldReview.model_validate_json(line))
                except ValidationError as exc:
                    raise DatasetError(f"{path} line {number}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{path}: cannot read file: {exc}") from exc
    return tuple(reviews)


def write_reviews(path: Path, reviews: Sequence[GoldReview]) -> None:
    """Write the reviews as JSONL, one object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for review in reviews:
            handle.write(review.model_dump_json() + "\n")


def review_summary(cases: Sequence[BenchmarkCase], reviews: Sequence[GoldReview]) -> dict[str, int]:
    """Coverage of the sampled cases, and how the verdicts fell.

    Reviews naming a case outside `cases` are ignored rather than counted: the
    summary answers "was this sample reviewed", so a stale line left over from
    an earlier sample must not make the coverage look complete.
    """
    sampled = {case.case_id for case in cases}
    relevant = [review for review in reviews if review.case_id in sampled]
    summary = {"sampled": len(sampled), "reviewed": len({r.case_id for r in relevant})}
    summary.update(dict.fromkeys(VERDICTS, 0))
    for review in relevant:
        summary[review.verdict] += 1
    return summary
