"""Caesar cipher transformation."""

import random
import string

from tipguard.benchmark.transformations.base import Encoded, Family

ALLOWED_SHIFTS = (1, 2, 3, 5, 7, 11, 13)


def _shift_letters(text: str, shift: int) -> str:
    result = []
    for char in text:
        if char in string.ascii_lowercase:
            index = (ord(char) - ord("a") + shift) % 26
            result.append(chr(ord("a") + index))
        elif char in string.ascii_uppercase:
            index = (ord(char) - ord("A") + shift) % 26
            result.append(chr(ord("A") + index))
        else:
            result.append(char)
    return "".join(result)


class CaesarTransformation:
    family = Family.CAESAR
    deterministic = True

    def encode(self, text: str, rng: random.Random) -> Encoded:
        shift = rng.choice(ALLOWED_SHIFTS)
        payload = _shift_letters(text, shift)
        return Encoded(
            payload=payload,
            family=self.family,
            params={"shift": str(shift)},
            hint=f"a Caesar cipher shifted forward by {shift}",
        )

    def decode(self, encoded: Encoded) -> str:
        shift = int(encoded.params["shift"])
        return _shift_letters(encoded.payload, -shift)
