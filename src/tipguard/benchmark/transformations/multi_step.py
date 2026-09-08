"""Two-step composition of the character-level deterministic transformations.

Morse is excluded from composition entirely: it lower-cases its output and
drops every character outside `a-z`, `0-9` and space, so an outer decode
cannot restore what an inner Morse step removed, and an outer Morse step
would destroy an inner step's punctuation.

Substitution is composable but only as the *inner* step. Its decode maps
every character to lower case, so wrapping a case-sensitive payload in it
(base64 above all) would be irreversible. Drawing it as the outer family
therefore swaps the two steps rather than producing a lossy pair; the two
families drawn are still distinct and still drawn from the same set.
"""

import random

from tipguard.benchmark.transformations.base import Encoded, Family, Transformation
from tipguard.benchmark.transformations.base64_t import Base64Transformation
from tipguard.benchmark.transformations.caesar import CaesarTransformation
from tipguard.benchmark.transformations.reverse import ReverseTransformation
from tipguard.benchmark.transformations.substitution import SubstitutionTransformation

COMPOSABLE: dict[Family, Transformation] = {
    Family.BASE64: Base64Transformation(),
    Family.CAESAR: CaesarTransformation(),
    Family.REVERSE: ReverseTransformation(),
    Family.SUBSTITUTION: SubstitutionTransformation(),
}
COMPOSABLE_FAMILIES: tuple[Family, ...] = tuple(sorted(COMPOSABLE, key=lambda f: f.value))
# Families whose decode cannot restore the case of the text they wrapped.
CASE_DESTROYING = frozenset({Family.SUBSTITUTION})
STEP_HINT = "one step of a composed transformation"


def _draw_steps(rng: random.Random) -> tuple[Family, Family]:
    """Draw two distinct families as (inner, outer), keeping decode exact."""
    first, second = rng.sample(COMPOSABLE_FAMILIES, 2)
    if second in CASE_DESTROYING and first not in CASE_DESTROYING:
        return second, first
    return first, second


def _prefixed(params: dict[str, str], prefix: str) -> dict[str, str]:
    return {f"{prefix}{key}": value for key, value in params.items()}


def _unprefixed(params: dict[str, str], prefix: str) -> dict[str, str]:
    return {
        key.removeprefix(prefix): value for key, value in params.items() if key.startswith(prefix)
    }


def _decode_step(params: dict[str, str], role: str, payload: str) -> str:
    family = Family(params[role])
    step = COMPOSABLE[family]
    encoded = Encoded(
        payload=payload,
        family=family,
        params=_unprefixed(params, f"{role}_"),
        hint=STEP_HINT,
    )
    return step.decode(encoded)


class MultiStepTransformation:
    family = Family.MULTI_STEP
    deterministic = True

    def encode(self, text: str, rng: random.Random) -> Encoded:
        inner_family, outer_family = _draw_steps(rng)
        inner = COMPOSABLE[inner_family].encode(text, rng)
        outer = COMPOSABLE[outer_family].encode(inner.payload, rng)
        params = (
            {"inner": inner_family.value, "outer": outer_family.value}
            | _prefixed(inner.params, "inner_")
            | _prefixed(outer.params, "outer_")
        )
        return Encoded(
            payload=outer.payload,
            family=self.family,
            params=params,
            hint=f"{outer.hint} wrapping {inner.hint}",
        )

    def decode(self, encoded: Encoded) -> str:
        inner_payload = _decode_step(encoded.params, "outer", encoded.payload)
        return _decode_step(encoded.params, "inner", inner_payload)
