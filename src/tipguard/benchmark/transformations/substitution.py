"""Character-substitution transformations (leetspeak and symbol swaps)."""

import random

from tipguard.benchmark.transformations.base import Encoded, Family

SUBSTITUTION_MAPS: dict[str, dict[str, str]] = {
    "leet": {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"},
    "symbols": {"a": "@", "e": "€", "i": "!", "o": "°", "s": "$", "t": "+"},
}


class SubstitutionTransformation:
    family = Family.SUBSTITUTION
    deterministic = True

    def encode(self, text: str, rng: random.Random) -> Encoded:
        name = rng.choice(list(SUBSTITUTION_MAPS))
        char_map = SUBSTITUTION_MAPS[name]
        payload = "".join(char_map.get(char.lower(), char) for char in text)
        return Encoded(
            payload=payload,
            family=self.family,
            params={"map": name},
            hint=f"text with letters replaced by {name} symbols",
        )

    def decode(self, encoded: Encoded) -> str:
        name = encoded.params["map"]
        reverse_map = {symbol: letter for letter, symbol in SUBSTITUTION_MAPS[name].items()}
        return "".join(reverse_map.get(char, char.lower()) for char in encoded.payload)
