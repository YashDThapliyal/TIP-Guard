"""Shared `DefenseConfig.params` parsing, used by every defense builder.

Mirrors the validation `guardrail.baselines` already applies to its own
params -- unknown keys rejected, a threshold parsed and range-checked, a
boolean spelled out explicitly rather than trusted to Python's `bool()`
(`bool("false")` is `True`) -- so a new defense gets the same guarantees
without repeating the logic.
"""

from collections.abc import Mapping
from typing import Any

from tipguard.config.loader import ConfigError

_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"false", "no", "off", "0"})


def check_known_params(params: Mapping[str, Any], known: frozenset[str], defense: str) -> None:
    """Reject any key in `params` that `known` does not name.

    `DefenseConfig.params` is an open mapping, so a mistyped key would
    otherwise fall back to its default in silence -- and a threshold sweep
    that mistyped one key would report the default arm as though it had
    swept.
    """
    unknown = sorted(set(params) - known)
    if unknown:
        raise ConfigError(
            f"defense {defense!r} got unknown params {unknown}; known params: {sorted(known)}"
        )


def parse_threshold(params: Mapping[str, Any], key: str, default: float, defense: str) -> float:
    raw = params.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"defense {defense!r}: {key} must be a number, got {raw!r}") from exc
    if not 0.0 <= value <= 1.01:
        # 1.01 rather than 1.0 so a config can express "never block": a score
        # is capped at 1.0 and the comparison is `>=`, so 1.0 would still
        # fire on a maximum-risk verdict.
        raise ConfigError(f"defense {defense!r}: {key} must be within [0, 1.01], got {value}")
    return value


def parse_bool(params: Mapping[str, Any], key: str, default: bool, defense: str) -> bool:
    """`params[key]` as a boolean, accepting common string spellings.

    Spelled out rather than passed through `bool()`, because `bool("false")`
    is `True`: a config saying an ablation switch is `"false"` would leave
    the component running while the run reported it as ablated, which
    corrupts the measurement the ablation exists to produce rather than
    merely failing.
    """
    raw = params.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        if raw.strip().lower() in _TRUE:
            return True
        if raw.strip().lower() in _FALSE:
            return False
    raise ConfigError(f"defense {defense!r}: {key} must be a boolean, got {raw!r}")
