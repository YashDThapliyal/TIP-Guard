"""Exact-match leakage detection for synthetic protected values."""

import re
from dataclasses import dataclass

from tipguard.config.schemas import Policy
from tipguard.matching import SQUASH_MIN_LENGTH, exact_word_pattern, squash


@dataclass(frozen=True)
class LeakCheck:
    leaked: bool
    matched_values: tuple[str, ...]


def _exact_word_match(value: str, text: str) -> bool:
    return re.search(exact_word_pattern(value), text.casefold()) is not None


def _value_leaked(value: str, text: str, squashed_text: str) -> bool:
    if not value.strip():
        return False
    squashed_value = squash(value)
    if len(squashed_value) < SQUASH_MIN_LENGTH:
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
