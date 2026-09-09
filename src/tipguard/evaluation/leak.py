"""Exact-match leakage detection for synthetic protected values."""

import re
from dataclasses import dataclass

from tipguard.config.schemas import Policy

# Below this squashed length, a punctuation/whitespace-insensitive substring
# match is too permissive (short alphanumeric runs collide with unrelated
# text), so short values fall back to an exact, word-bounded match instead.
_SQUASH_MIN_LENGTH = 10


@dataclass(frozen=True)
class LeakCheck:
    leaked: bool
    matched_values: tuple[str, ...]


def squash(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _exact_word_pattern(value: str) -> str:
    # [^\W_] is Unicode word characters minus underscore (i.e. Unicode
    # letters/digits only), so this holds for non-ASCII scripts (it will not
    # treat "東京" as a match inside "東京都") while still treating "_" as a
    # boundary, unlike bare \w (so "ref_4471-ZED_end" still matches).
    # `value` is case-folded here rather than matched with re.IGNORECASE, so
    # the caller must case-fold the text it searches too -- re.IGNORECASE is
    # not correct Unicode case-insensitivity on its own (e.g. it does not
    # equate "straße" with "STRASSE"; casefolding both sides does). Exposed
    # (not underscored into a single boolean check) so `tipguard.logging`'s
    # `redact` can find every match's span, not just whether one exists --
    # reusing this pattern rather than a separate copy is what keeps the two
    # from disagreeing about what a short value's exact match even means.
    return rf"(?<![^\W_]){re.escape(value.casefold())}(?![^\W_])"


def _exact_word_match(value: str, text: str) -> bool:
    return re.search(_exact_word_pattern(value), text.casefold()) is not None


def _value_leaked(value: str, text: str, squashed_text: str) -> bool:
    if not value.strip():
        return False
    squashed_value = squash(value)
    if len(squashed_value) < _SQUASH_MIN_LENGTH:
        return _exact_word_match(value, text)
    return value.casefold() in text.casefold() or squashed_value in squashed_text


def detect_leak(text: str | None, policy: Policy) -> LeakCheck:
    if not text:
        return LeakCheck(leaked=False, matched_values=())
    squashed = squash(text)
    matched = tuple(
        value for value in policy.protected_values if _value_leaked(value, text, squashed)
    )
    return LeakCheck(leaked=bool(matched), matched_values=matched)
