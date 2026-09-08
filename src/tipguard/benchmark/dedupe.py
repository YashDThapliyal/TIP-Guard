"""Exact and near-duplicate removal for generated benchmark cases.

Two cases whose prompts differ only in a word or two teach a defense nothing
extra and inflate every score computed over the set, so they are dropped. An
exact repeat is caught by the normalised prompt; a near repeat by five-gram
word shingles compared with Jaccard similarity.

Near-duplicates are only sought *within* a case type. A hard negative that
quotes an attack is meant to look like that attack: it is the whole point of
the control, and removing it would delete exactly the cases that separate a
guardrail from a keyword filter.

Comparison is driven by a shingle index rather than by every earlier case, so
a prompt sharing no five-gram with anything kept is never compared at all.
"""

from collections.abc import Sequence

from tipguard.benchmark.schema import BenchmarkCase, CaseType
from tipguard.benchmark.validate import normalise_prompt

SHINGLE_SIZE = 5

#: Similarity at or above which two prompts of the same type count as the
#: same case. Chosen high: the generator's paraphrases are meant to differ by
#: more than one trailing word.
JACCARD_THRESHOLD = 0.9

Shingle = tuple[str, ...]


def shingles(prompt: str) -> frozenset[Shingle]:
    """The five-gram word shingles of a prompt, normalised first.

    A prompt shorter than the window has no five-gram, so it is represented
    by the single shingle of its whole word sequence; otherwise every short
    prompt would have an empty shingle set and compare as similar to nothing,
    including to its own repeats.
    """
    words = normalise_prompt(prompt).split()
    if len(words) < SHINGLE_SIZE:
        return frozenset({tuple(words)})
    return frozenset(
        tuple(words[start : start + SHINGLE_SIZE]) for start in range(len(words) - SHINGLE_SIZE + 1)
    )


def jaccard(left: frozenset[Shingle], right: frozenset[Shingle]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


class _ShingleIndex:
    """Kept prompts, searchable by the shingles they contain."""

    def __init__(self) -> None:
        self._sets: list[frozenset[Shingle]] = []
        self._by_shingle: dict[tuple[CaseType, Shingle], list[int]] = {}

    def has_near_duplicate(self, case_type: CaseType, candidate: frozenset[Shingle]) -> bool:
        seen: set[int] = set()
        for shingle in candidate:
            seen.update(self._by_shingle.get((case_type, shingle), ()))
        return any(
            jaccard(candidate, self._sets[position]) >= JACCARD_THRESHOLD for position in seen
        )

    def add(self, case_type: CaseType, candidate: frozenset[Shingle]) -> None:
        position = len(self._sets)
        self._sets.append(candidate)
        for shingle in candidate:
            self._by_shingle.setdefault((case_type, shingle), []).append(position)


def dedupe_cases(cases: Sequence[BenchmarkCase]) -> tuple[BenchmarkCase, ...]:
    """`cases` with exact and near repeats dropped, keeping the first of each."""
    index = _ShingleIndex()
    seen_prompts: set[str] = set()
    kept: list[BenchmarkCase] = []
    for case in cases:
        normalised = normalise_prompt(case.prompt)
        if normalised in seen_prompts:
            continue
        candidate = shingles(case.prompt)
        if index.has_near_duplicate(case.case_type, candidate):
            continue
        seen_prompts.add(normalised)
        index.add(case.case_type, candidate)
        kept.append(case)
    return tuple(kept)
