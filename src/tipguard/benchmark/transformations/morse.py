"""International Morse code transformation for letters and digits."""

import random

from tipguard.benchmark.transformations.base import Encoded, Family

MORSE_TABLE: dict[str, str] = {
    "a": ".-",
    "b": "-...",
    "c": "-.-.",
    "d": "-..",
    "e": ".",
    "f": "..-.",
    "g": "--.",
    "h": "....",
    "i": "..",
    "j": ".---",
    "k": "-.-",
    "l": ".-..",
    "m": "--",
    "n": "-.",
    "o": "---",
    "p": ".--.",
    "q": "--.-",
    "r": ".-.",
    "s": "...",
    "t": "-",
    "u": "..-",
    "v": "...-",
    "w": ".--",
    "x": "-..-",
    "y": "-.--",
    "z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
}

SPACE_TOKEN = "/"
REVERSE_MORSE_TABLE = {code: char for char, code in MORSE_TABLE.items()}


class MorseTransformation:
    family = Family.MORSE
    deterministic = True

    def encode(self, text: str, rng: random.Random) -> Encoded:
        del rng
        tokens = []
        for char in text.lower():
            if char == " ":
                tokens.append(SPACE_TOKEN)
            elif char in MORSE_TABLE:
                tokens.append(MORSE_TABLE[char])
        payload = " ".join(tokens)
        return Encoded(payload=payload, family=self.family, hint="Morse code")

    def decode(self, encoded: Encoded) -> str:
        chars = []
        for token in encoded.payload.split(" "):
            if not token:
                continue
            if token == SPACE_TOKEN:
                chars.append(" ")
            else:
                chars.append(REVERSE_MORSE_TABLE[token])
        return "".join(chars)
