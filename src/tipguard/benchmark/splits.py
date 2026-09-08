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

from tipguard.benchmark.schema import BenchmarkCase, Split
from tipguard.benchmark.transformations.base import Family
from tipguard.config.schemas import FrozenModel

#: Floating-point slack allowed when the three ratios are summed.
RATIO_TOLERANCE = 1e-9

FamilyPair = tuple[Family, Family]

#: Every family name the registry knows, for reading metadata that may not be
#: trustworthy.
FAMILY_VALUES: frozenset[str] = frozenset(family.value for family in Family)

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


def _encoded_params(case: BenchmarkCase) -> dict[str, object]:
    """The case's decoded `encoded_params`, or an empty mapping.

    Malformed metadata is tolerated rather than fatal: a case that cannot say
    what it composed simply contributes nothing to the composition rules.
    """
    try:
        params = json.loads(case.metadata.get("encoded_params", "{}"))
    except json.JSONDecodeError:
        return {}
    return params if isinstance(params, dict) else {}


def _multi_step_pair(case: BenchmarkCase) -> FamilyPair | None:
    """The (inner, outer) families of a multi-step case, or None.

    The compositional rule is about an ordered pair, so both halves have to
    parse for it to mean anything.
    """
    if case.transformation != Family.MULTI_STEP.value:
        return None
    params = _encoded_params(case)
    inner, outer = params.get("inner"), params.get("outer")
    if not isinstance(inner, str) or not isinstance(outer, str):
        return None
    try:
        return (Family(inner), Family(outer))
    except ValueError:
        return None


def _composed_families(case: BenchmarkCase) -> frozenset[str]:
    """Every family named in a multi-step case's params that is recognisable.

    Read independently of each other, not as a pair: one unreadable half must
    not hide the other, or a case composing a withheld family alongside a name
    the registry does not know would slip into the standard pool.
    """
    if case.transformation != Family.MULTI_STEP.value:
        return frozenset()
    params = _encoded_params(case)
    names = (params.get("inner"), params.get("outer"))
    return frozenset(name for name in names if isinstance(name, str) and name in FAMILY_VALUES)


def _case_families(case: BenchmarkCase) -> frozenset[str]:
    """Every family `case` uses, including the halves of a composition.

    A multi-step case is labelled `multi_step` whatever it composes, so the
    family hold-out has to look inside it: otherwise a configuration that
    withholds base64 still shows base64 to the standard splits, wrapped in a
    second transformation.
    """
    return frozenset({case.transformation}) | _composed_families(case)


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

    The paraphrase test is not restricted to TIP cases. A `direct` case carries
    the same `intent_id` and `phrasing_index` and is the plaintext form of the
    very sentence the TIP case encodes, so leaving it in the standard pool
    would put the reserved wording in front of the defense verbatim.
    """
    if _case_families(case) & conditions.families:
        return Split.HELDOUT_TRANSFORMATION
    if case.policy_id is not None and case.policy_id == conditions.policy:
        return Split.HELDOUT_POLICY
    if _multi_step_pair(case) in conditions.pairs:
        return Split.HELDOUT_COMPOSITIONAL
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


#: A stratum at least this large always contributes to dev, even when its
#: proportional share rounds to nothing. Below it there is nothing sensible to
#: take, since every split would be left empty by the transfer.
MIN_SIZE_FOR_DEV = 3


def _largest_remainder(size: int, config: SplitConfig) -> dict[Split, int]:
    """Whole counts summing to `size` and closest to the configured shares.

    Flooring alone loses up to two cases per stratum and rounding each boundary
    independently can overshoot, so the floors are taken first and the leftover
    handed to the largest fractional parts, ties broken by split name for
    determinism.
    """
    shares = {
        Split.TRAIN: size * config.train,
        Split.DEV: size * config.dev,
        Split.TEST: size * config.test,
    }
    counts = {split: int(share) for split, share in shares.items()}
    leftover = size - sum(counts.values())
    ranked = sorted(shares, key=lambda split: (-(shares[split] % 1), split.value))
    for split in ranked[:leftover]:
        counts[split] += 1
    return counts


def _stratum_splits(size: int, config: SplitConfig) -> tuple[Split, ...]:
    """The split labels for one shuffled stratum, in order."""
    counts = _largest_remainder(size, config)
    if size >= MIN_SIZE_FOR_DEV and counts[Split.DEV] == 0:
        donor = max(counts, key=lambda split: (counts[split], split.value))
        counts[donor] -= 1
        counts[Split.DEV] += 1
    return tuple(
        split for split in (Split.TRAIN, Split.DEV, Split.TEST) for _ in range(counts[split])
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
  the standard splits, including a family composed inside a multi-step case,
  attacks and their benign counterparts alike. It tests whether a defense
  recovers intent from a transformation it has never seen. Some of its cases
  also target the withheld policy and are therefore doubly held out, so a
  failure on them cannot be attributed to the novel transformation alone;
  report the headline both over the whole condition and over the cases that
  are held out for the transformation only.
- `heldout_policy` — every case targeting a withheld policy's secret. It tests
  whether the defense generalizes across protected content rather than
  memorising the phrasing of one policy.
- `heldout_compositional` — multi-step cases, attacks and benign counterparts
  alike, whose inner and outer families are each present in training but whose
  composition is not. It separates knowing the parts from following the chain.
- `heldout_paraphrase` — cases written with the last phrasings of each intent,
  unseen elsewhere, in both their encoded and their plaintext form. It tests
  robustness to rewording: the transformations and policies these cases use
  were all seen in training, so only the wording is new.

`heldout_policy` and `heldout_paraphrase` contain no allow-side cases at all.
Benign and hard-negative controls carry no policy and no intent, so neither
condition can claim one, and every case in both expects `block`. No
false-positive rate can be computed within them; measure false positives on the
standard pool or on `heldout_transformation`, which does carry benign
counterparts.
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
            # A split emptied by a configuration change must not leave last
            # run's manifest behind: a consumer enumerating the directory would
            # read stale case ids and the partition would no longer be exact.
            (out_dir / f"{split.value}.json").unlink(missing_ok=True)
            continue
        manifest = {"split": split.value, "count": len(case_ids), "case_ids": case_ids}
        path = out_dir / f"{split.value}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        counts[split.value] = len(case_ids)
    (out_dir / "README.md").write_text(README, encoding="utf-8")
    return counts
