"""Simple string-reversal transformation."""

import random

from tipguard.benchmark.transformations.base import Encoded, Family


class ReverseTransformation:
    family = Family.REVERSE
    deterministic = True

    def encode(self, text: str, rng: random.Random) -> Encoded:
        del rng
        return Encoded(payload=text[::-1], family=self.family, hint="reversed text")

    def decode(self, encoded: Encoded) -> str:
        return encoded.payload[::-1]
