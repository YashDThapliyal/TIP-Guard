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


def _squash(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _exact_word_match(value: str, text: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _value_leaked(value: str, text: str, squashed_text: str) -> bool:
    if not value.strip():
        return False
    squashed_value = _squash(value)
    if len(squashed_value) < _SQUASH_MIN_LENGTH:
        return _exact_word_match(value, text)
    return value.casefold() in text.casefold() or squashed_value in squashed_text


def detect_leak(text: str | None, policy: Policy) -> LeakCheck:
    if not text:
        return LeakCheck(leaked=False, matched_values=())
    squashed = _squash(text)
    matched = tuple(
        value for value in policy.protected_values if _value_leaked(value, text, squashed)
    )
    return LeakCheck(leaked=bool(matched), matched_values=matched)
