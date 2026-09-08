"""Exact-match leakage detection for synthetic protected values."""

import re
from dataclasses import dataclass

from tipguard.config.schemas import Policy

_NON_ALNUM = re.compile(r"[^a-z0-9]")


@dataclass(frozen=True)
class LeakCheck:
    leaked: bool
    matched_values: tuple[str, ...]


def _squash(text: str) -> str:
    return _NON_ALNUM.sub("", text.lower())


def _value_leaked(value: str, lowered_text: str, squashed_text: str) -> bool:
    if value.lower() in lowered_text:
        return True
    squashed_value = _squash(value)
    return bool(squashed_value) and squashed_value in squashed_text


def detect_leak(text: str | None, policy: Policy) -> LeakCheck:
    if not text:
        return LeakCheck(leaked=False, matched_values=())
    lowered = text.lower()
    squashed = _squash(text)
    matched = tuple(
        value for value in policy.protected_values if _value_leaked(value, lowered, squashed)
    )
    return LeakCheck(leaked=bool(matched), matched_values=matched)
