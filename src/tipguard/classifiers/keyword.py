"""Baseline 1: a flat keyword blocklist.

This is the weakest conventional filter in the study and it is meant to stay
that way. It knows nothing about what a prompt is asking for; it only knows
that certain nouns are worth noticing. Its value is as a floor: whatever the
semantic defenses buy has to be measured against what simply grepping for the
obvious words already achieves.
"""

import re
from collections.abc import Sequence

from tipguard.classifiers.types import RiskCategory, RiskScore

#: The key nouns of the six protected labels in `configs/policies.yaml`,
#: plus the generic exfiltration verb and the override phrases that a
#: hand-written blocklist would obviously carry.
DEFAULT_KEYWORDS: tuple[str, ...] = (
    "canary",
    "system prompt",
    "secret",
    "passphrase",
    "api token",
    "api key",
    "credential",
    "codename",
    "admin url",
    "customer record",
    "account number",
    "reveal",
    "ignore previous",
    "ignore prior",
    "disregard your instructions",
)

#: Keywords that describe an attempt to override the assistant's standing
#: instructions rather than an asset being asked for. Everything else in the
#: list is an asset noun.
OVERRIDE_KEYWORDS: frozenset[str] = frozenset(
    {"ignore previous", "ignore prior", "disregard your instructions"}
)

#: Score contributed by each distinct keyword found. Three keywords in one
#: prompt saturate the score, which matches the intuition that a prompt
#: naming three protected assets needs no further evidence.
WEIGHT_PER_KEYWORD = 0.35


def _compile(keyword: str) -> re.Pattern[str]:
    """Word-bounded, case-insensitive matcher for one keyword.

    Multi-word keywords are matched as phrases -- the words must appear in
    order and adjacent -- rather than as a bag of words, so "the system is
    fine and the prompt is short" does not count as "system prompt". The
    boundaries are symmetric, which means an inflection such as
    "credentials" does not match "credential". That is a genuine hole, and
    it is left open on purpose: widening it is exactly the kind of quiet
    tuning that would turn an honest baseline into a flattering one.
    """
    words = keyword.split()
    if not words:
        # An empty or whitespace-only keyword compiles to `\b\b`, which
        # matches almost any text and would silently saturate every score.
        # A hand-edited blocklist is exactly where that typo comes from, so
        # it fails at construction rather than at measurement time.
        raise ValueError("keyword must contain at least one word")
    body = r"\s+".join(re.escape(word) for word in words)
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


class KeywordClassifier:
    """Scores a prompt by how many distinct blocklist keywords it contains."""

    name = "keyword"

    def __init__(self, keywords: Sequence[str] = DEFAULT_KEYWORDS) -> None:
        # Kept in list order, not as a set: evidence has to come out in a
        # fixed order for two runs of the same experiment to be comparable.
        self._keywords: tuple[str, ...] = tuple(keywords)
        self._patterns: tuple[re.Pattern[str], ...] = tuple(_compile(k) for k in self._keywords)

    def score(self, text: str) -> RiskScore:
        matched = tuple(
            keyword
            for keyword, pattern in zip(self._keywords, self._patterns, strict=True)
            if pattern.search(text)
        )
        return RiskScore(
            score=min(1.0, WEIGHT_PER_KEYWORD * len(matched)),
            categories=_categories(matched),
            evidence=matched,
            classifier=self.name,
        )


def _categories(matched: Sequence[str]) -> tuple[str, ...]:
    found: list[str] = []
    if any(keyword not in OVERRIDE_KEYWORDS for keyword in matched):
        found.append(RiskCategory.DATA_EXFILTRATION.value)
    if any(keyword in OVERRIDE_KEYWORDS for keyword in matched):
        found.append(RiskCategory.PROMPT_INJECTION.value)
    return tuple(found) if found else (RiskCategory.NONE.value,)
