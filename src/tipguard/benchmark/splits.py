"""Split assignment and manifests for the benchmark dataset.

Four conditions are held out before anything is randomised, because each one
answers a question a random split cannot. `heldout_transformation` removes two
encoding families outright, `heldout_policy` removes one policy's protected
secret, `heldout_compositional` removes two multi-step orderings, and
`heldout_paraphrase` removes the last phrasings of every intent. What is left
is stratified into train, dev and test.

The rules are applied in a fixed priority order and the first match wins, so a
case that satisfies two conditions is counted once, under the rarer one. A
held-out family takes every case that uses it, benign controls and hard
negatives included: a defense tuned on the standard split must never have seen
the family in any guise, or the transfer number it reports is measuring
memorisation.
"""

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, model_validator

from tipguard.benchmark.schema import BenchmarkCase, CaseType, Split
from tipguard.benchmark.transformations.base import Family
from tipguard.config.schemas import FrozenModel

#: Floating-point slack allowed when the three ratios are summed.
RATIO_TOLERANCE = 1e-9

FamilyPair = tuple[Family, Family]

DEFAULT_HELDOUT_FAMILIES: tuple[Family, ...] = (Family.MORSE, Family.CODE)

#: Read as (inner, outer): the family applied to the payload first, then the
#: family applied to that result.
DEFAULT_HELDOUT_PAIRS: tuple[FamilyPair, ...] = (
    (Family.CAESAR, Family.BASE64),
    (Family.SUBSTITUTION, Family.REVERSE),
)


class SplitConfig(FrozenModel):
    """Which conditions are held out, and how the rest is divided."""

    seed: int = 20260907
    train: float = 0.6
    dev: float = 0.1
    test: float = 0.3
    heldout_families: list[Family] = Field(default_factory=lambda: list(DEFAULT_HELDOUT_FAMILIES))
    heldout_policy: str = "protect-launch-codename"
    #: How many of each intent's trailing phrasing indices are held out. The
    #: indices themselves are read off the dataset rather than configured, so
    #: regenerating with more paraphrases moves the boundary automatically.
    heldout_phrasing_count: int = Field(default=4, ge=1)
    heldout_multi_step_pairs: list[FamilyPair] = Field(
        default_factory=lambda: list(DEFAULT_HELDOUT_PAIRS)
    )

    @model_validator(mode="after")
    def _ratios_partition_the_standard_pool(self) -> "SplitConfig":
        """A ratio of 0 or 1, or three that do not sum to 1, silently loses or
        duplicates cases later, so it is rejected at load time instead."""
        ratios = {"train": self.train, "dev": self.dev, "test": self.test}
        for name, value in ratios.items():
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1, not {value}")
        total = sum(ratios.values())
        if abs(total - 1.0) > RATIO_TOLERANCE:
            raise ValueError(f"train, dev and test must sum to 1.0, not {total}")
        return self


@dataclass(frozen=True)
class _Conditions:
    """The held-out rules, resolved against the dataset once."""

    families: frozenset[str]
    policy: str
    pairs: frozenset[FamilyPair]
    #: intent id -> the phrasing indices held out for that intent.
    phrasings: Mapping[str, frozenset[int]]


def _phrasing_index(case: BenchmarkCase) -> int | None:
    """The case's phrasing index, or None if it carries none that parses.

    Cases without one, such as hard negatives generated outside the intent
    bank, simply cannot match the paraphrase rule.
    """
    raw = case.metadata.get("phrasing_index")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _multi_step_pair(case: BenchmarkCase) -> FamilyPair | None:
    """The (inner, outer) families of a multi-step case, or None."""
    if case.transformation != Family.MULTI_STEP.value:
        return None
    try:
        params = json.loads(case.metadata.get("encoded_params", "{}"))
    except json.JSONDecodeError:
        return None
    if not isinstance(params, dict):
        return None
    try:
        return (Family(params["inner"]), Family(params["outer"]))
    except (KeyError, ValueError):
        return None


def _heldout_phrasings(cases: Sequence[BenchmarkCase], count: int) -> dict[str, frozenset[int]]:
    """The last `count` phrasing indices observed per intent, from the data."""
    observed: dict[str, set[int]] = defaultdict(set)
    for case in cases:
        intent = case.metadata.get("intent_id")
        index = _phrasing_index(case)
        if intent and index is not None:
            observed[intent].add(index)
    return {intent: frozenset(sorted(indices)[-count:]) for intent, indices in observed.items()}


def _conditions(cases: Sequence[BenchmarkCase], config: SplitConfig) -> _Conditions:
    return _Conditions(
        families=frozenset(family.value for family in config.heldout_families),
        policy=config.heldout_policy,
        pairs=frozenset(config.heldout_multi_step_pairs),
        phrasings=_heldout_phrasings(cases, config.heldout_phrasing_count),
    )


def _heldout_split(case: BenchmarkCase, conditions: _Conditions) -> Split | None:
    """The held-out condition `case` belongs to, first match wins, or None.

    The compositional test precedes the paraphrase test deliberately: a
    multi-step case can satisfy both, and the compositional pool is both
    smaller and the more interesting generalization claim.
    """
    if case.transformation in conditions.families:
        return Split.HELDOUT_TRANSFORMATION
    if case.policy_id is not None and case.policy_id == conditions.policy:
        return Split.HELDOUT_POLICY
    if _multi_step_pair(case) in conditions.pairs:
        return Split.HELDOUT_COMPOSITIONAL
    if case.case_type is CaseType.TIP:
        intent = case.metadata.get("intent_id")
        index = _phrasing_index(case)
        if intent and index in conditions.phrasings.get(intent, frozenset()):
            return Split.HELDOUT_PARAPHRASE
    return None


#: What the standard pool is stratified by, so every family, type and
#: difficulty is represented in train, dev and test in the same proportions.
Stratum = tuple[str, str, int]


def _stratum(case: BenchmarkCase) -> Stratum:
    return (case.case_type.value, case.transformation, case.difficulty)


def _stratum_splits(size: int, config: SplitConfig) -> tuple[Split, ...]:
    """The split labels for one shuffled stratum, in order.

    Boundaries are rounded rather than floored so a stratum of any size stays
    close to the configured proportions instead of leaking its remainder into
    a single split.
    """
    train_end = round(size * config.train)
    dev_end = round(size * (config.train + config.dev))
    return (
        (Split.TRAIN,) * train_end
        + (Split.DEV,) * (dev_end - train_end)
        + (Split.TEST,) * (size - dev_end)
    )


def _standard_splits(cases: Sequence[BenchmarkCase], config: SplitConfig) -> dict[str, Split]:
    """case_id -> train, dev or test, stratified and seeded.

    Each stratum is shuffled with its own seeded generator derived from the
    configured seed and the stratum key, so the assignment does not depend on
    the order the cases arrived in or on which other strata exist.
    """
    strata: dict[Stratum, list[str]] = defaultdict(list)
    for case in cases:
        strata[_stratum(case)].append(case.case_id)
    assignment: dict[str, Split] = {}
    for stratum in sorted(strata):
        case_ids = sorted(strata[stratum])
        random.Random(f"{config.seed}:{stratum}").shuffle(case_ids)
        labels = _stratum_splits(len(case_ids), config)
        assignment.update(zip(case_ids, labels, strict=True))
    return assignment


def assign_splits(cases: Sequence[BenchmarkCase], config: SplitConfig) -> tuple[BenchmarkCase, ...]:
    """`cases` with `split` rewritten, in the order given, inputs untouched."""
    conditions = _conditions(cases, config)
    heldout = {
        case.case_id: split
        for case in cases
        if (split := _heldout_split(case, conditions)) is not None
    }
    standard = _standard_splits([case for case in cases if case.case_id not in heldout], config)
    return tuple(
        case.model_copy(
            update={
                "split": heldout[case.case_id]
                if case.case_id in heldout
                else standard[case.case_id]
            }
        )
        for case in cases
    )


README = """# Benchmark splits

Each file lists the `case_id`s of one split of `data/generated/tipguard-v1.jsonl`.
Together they cover the dataset exactly once. Regenerate with `tipguard split`.

The held-out conditions below are evaluation-only: they must never be used for tuning,
prompt engineering, threshold selection or any other model-facing choice. A defense
that has seen them measures memorisation, not generalization.

- `train` — the pool a defense may be developed on.
- `dev` — the pool it may be tuned and validated against.
- `test` — the in-distribution held-out pool, drawn from the same conditions as
  `train`, so it measures ordinary generalization only.
- `heldout_transformation` — every case using an encoding family withheld from
  the standard splits, attacks and their benign counterparts alike. It tests
  whether a defense recovers intent from a transformation it has never seen.
- `heldout_policy` — every case targeting a withheld policy's secret. It tests
  whether the defense generalizes across protected content rather than
  memorising the phrasing of one policy.
- `heldout_compositional` — multi-step cases, attacks and benign counterparts
  alike, whose inner and outer families are each present in training but whose
  composition is not. It separates knowing the parts from following the chain.
- `heldout_paraphrase` — attacks written with the last phrasings of each
  intent, unseen elsewhere. It tests robustness to rewording alone, holding the
  transformation and the policy fixed.
"""


def write_manifests(cases: Sequence[BenchmarkCase], out_dir: Path) -> dict[str, int]:
    """Write one manifest per non-empty split plus a README; return the counts."""
    grouped: dict[Split, list[str]] = defaultdict(list)
    for case in cases:
        grouped[case.split].append(case.case_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in Split:
        case_ids = sorted(grouped.get(split, ()))
        if not case_ids:
            continue
        manifest = {"split": split.value, "count": len(case_ids), "case_ids": case_ids}
        path = out_dir / f"{split.value}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        counts[split.value] = len(case_ids)
    (out_dir / "README.md").write_text(README, encoding="utf-8")
    return counts
