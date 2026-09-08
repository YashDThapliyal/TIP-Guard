"""Character-substitution transformations (leetspeak and symbol swaps)."""

import random

from tipguard.benchmark.transformations.base import Encoded, Family

SUBSTITUTION_MAPS: dict[str, dict[str, str]] = {
    "leet": {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"},
    "symbols": {"a": "@", "e": "€", "i": "!", "o": "°", "s": "$", "t": "+"},
}


def _literal_indices(text: str, char_map: dict[str, str]) -> list[int]:
    """Indices whose source character is already a map glyph, not a map key.

    Substitution is one character to one position, so these positions must
    be excluded from reverse-mapping on decode, or a source character that
    happens to already be a replacement glyph (e.g. a literal digit under
    the leet map) would be corrupted into the letter it stands for.
    """
    keys = char_map.keys()
    values = set(char_map.values())
    return [index for index, char in enumerate(text) if char in values and char not in keys]


def _choose_map(text: str, rng: random.Random) -> str:
    """Pick the map with the fewest literal collisions against `text`.

    Ties (including the common case of zero collisions for both maps) are
    broken with `rng.choice` over the tied names sorted alphabetically, so
    the draw stays deterministic for a given seed.
    """
    counts = {name: len(_literal_indices(text, m)) for name, m in SUBSTITUTION_MAPS.items()}
    min_count = min(counts.values())
    candidates = sorted(name for name, count in counts.items() if count == min_count)
    return str(rng.choice(candidates))


def _format_literals(indices: list[int]) -> str:
    return ",".join(str(index) for index in indices)


def _parse_literals(literals: str) -> set[int]:
    return {int(index) for index in literals.split(",") if index}


def decode_without_params(payload: str, map_name: str) -> str:
    """Best-effort reverse-substitution with no positional ground truth.

    This is what a canonicalizer sees in practice: raw prompt text with no
    `Encoded.params` to consult. It reverse-maps every glyph of `map_name`
    it finds, with no way to tell a substituted glyph apart from a literal
    character that happens to already look like one (e.g. a real digit
    under the leet map, or a literal "$" under the symbols map). Callers
    must treat its output as low confidence whenever the source text isn't
    known to be collision-free; `Encoded.params["ambiguous"]` records that
    at generation time, but this function has no access to it.
    """
    reverse_map = {symbol: letter for letter, symbol in SUBSTITUTION_MAPS[map_name].items()}
    return "".join(reverse_map.get(char, char.lower()) for char in payload)


class SubstitutionTransformation:
    family = Family.SUBSTITUTION
    deterministic = True

    def encode(self, text: str, rng: random.Random) -> Encoded:
        name = _choose_map(text, rng)
        char_map = SUBSTITUTION_MAPS[name]
        literals = _literal_indices(text, char_map)
        payload = "".join(char_map.get(char.lower(), char) for char in text)
        return Encoded(
            payload=payload,
            family=self.family,
            params={
                "map": name,
                "literals": _format_literals(literals),
                "ambiguous": "true" if literals else "false",
            },
            hint=f"text with letters replaced by {name} symbols",
        )

    def decode(self, encoded: Encoded) -> str:
        name = encoded.params["map"]
        reverse_map = {symbol: letter for letter, symbol in SUBSTITUTION_MAPS[name].items()}
        literal_indices = _parse_literals(encoded.params.get("literals", ""))
        result = []
        for index, char in enumerate(encoded.payload):
            if index in literal_indices:
                result.append(char)
            else:
                result.append(reverse_map.get(char, char.lower()))
        return "".join(result)
