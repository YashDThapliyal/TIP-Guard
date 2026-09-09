"""Baseline 2: a syntactic detector for encoded content and injection phrasing.

Where the keyword filter looks at what a prompt names, this one looks at what
a prompt is shaped like. It is the strongest thing a defender can build
without a model in the loop, and it has one structural blind spot the study
cares about: it can tell that a prompt carries an encoded payload, but not
what the payload says. A benign transformation and a Task-in-Prompt attack
look identical to it. Nothing here tries to close that gap.

Every pattern is compiled once at import; scoring is a fixed number of scans
over the text with no state carried between calls.
"""

import re

from tipguard.classifiers.types import RiskCategory, RiskScore

#: A contiguous base64-alphabet run. Twenty-four characters is roughly
#: eighteen bytes of payload, long enough that ordinary prose and identifiers
#: do not reach it.
BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")

#: Four or more Morse letter groups in a row. Fewer than four is within reach
#: of ordinary punctuation such as an ellipsis followed by a dash. Written as
#: one group plus three or more whitespace-separated successors rather than as
#: four repetitions of "group then whitespace", so that a payload running to
#: the very end of the prompt -- which has no trailing whitespace -- still
#: reaches the documented four-group minimum.
MORSE_RUN = re.compile(r"[.\-]{1,6}(?:\s+[.\-]{1,6}){3,}")

#: Markers of a Python snippet that assembles a string a character or a chunk
#: at a time -- the shape the `code` transformation family emits.
PYTHON_STRING_BUILDER = re.compile(r"(?:\bchr\(|\.join\(|\.replace\(|\breversed\()")

#: Glyphs the substitution family swaps letters for. Digits are handled
#: separately, since any non-alphabetic character in a word position counts.
SUBSTITUTION_SYMBOLS = "@€!°$+#%&*"

#: Window and threshold for the substitution-density test: within any run of
#: twelve consecutive word-position characters, at least a quarter must be
#: digits or symbols. Twelve is about two English words, short enough to see
#: a single leeted phrase inside a long benign wrapper; a quarter is well
#: above the density of ordinary prose, which is near zero once punctuation
#: and whitespace are excluded.
DENSITY_WINDOW = 12
MIN_REPLACED_FRACTION = 0.25

WEIGHT_PER_ENCODING = 0.3
WEIGHT_PER_INSTRUCTION = 0.4
WEIGHT_PER_INJECTION = 0.4

DECODE_INSTRUCTIONS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rf"\b{body}\b", re.IGNORECASE))
    for name, body in (
        ("decode", "decode"),
        ("decipher", "decipher"),
        ("unscramble", "unscramble"),
        ("shift_each_letter", r"shift\s+each\s+letter"),
        ("caesar", "caesar"),
        ("rot13", "rot13"),
        ("base64", "base64"),
        ("morse", "morse"),
    )
)

INJECTION_PHRASES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(body, re.IGNORECASE))
    for name, body in (
        ("ignore_previous_instructions", r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior)\b"),
        ("do_what_it_says", r"\bdo\s+what\s+it\s+says\b"),
        ("follow_the_instruction", r"\bfollow\s+the\s+instructions?\b"),
        ("comply", r"\bcomply\b"),
    )
)


def _is_word_position(char: str) -> bool:
    """True for a character that would hold a letter in the plaintext.

    Whitespace and punctuation are excluded so that the density is measured
    over word content only; a substituted glyph and the letter it stands for
    both count, which is what makes the ratio meaningful.
    """
    return char.isalnum() or char in SUBSTITUTION_SYMBOLS


def _replacement_flags(text: str) -> list[int]:
    """One flag per word-position character: 1 where a letter looks replaced.

    A non-alphabetic character counts as a replacement only when its own
    whitespace-delimited token also contains a letter. Without that guard the
    rule fires on any prose carrying a date, a price, a version number or a
    phone number: collapsing whitespace lets a purely numeric token supply the
    density for a window made otherwise of ordinary words. Leetspeak, by
    contrast, always leaves letters beside the glyphs it substitutes in.
    """
    flags: list[int] = []
    for token in text.split():
        lettered = any(char.isalpha() for char in token)
        flags.extend(
            0 if char.isalpha() or not lettered else 1 for char in token if _is_word_position(char)
        )
    return flags


def _has_substitution_density(text: str) -> bool:
    """True if some twelve-character word run is a quarter digits or symbols.

    Whitespace is collapsed away before the window slides, so a short leeted
    phrase is still detected when it sits inside a long, ordinary sentence --
    which is how the benchmark's wrappers present it.
    """
    replaced = _replacement_flags(text)
    if len(replaced) < DENSITY_WINDOW:
        return False
    threshold = DENSITY_WINDOW * MIN_REPLACED_FRACTION
    window = sum(replaced[:DENSITY_WINDOW])
    if window >= threshold:
        return True
    for index in range(DENSITY_WINDOW, len(replaced)):
        window += replaced[index] - replaced[index - DENSITY_WINDOW]
        if window >= threshold:
            return True
    return False


def _encoding_hits(text: str) -> tuple[str, ...]:
    checks = (
        ("base64_run", bool(BASE64_RUN.search(text))),
        ("morse_run", bool(MORSE_RUN.search(text))),
        ("substitution_density", _has_substitution_density(text)),
        ("python_string_builder", bool(PYTHON_STRING_BUILDER.search(text))),
    )
    return tuple(name for name, hit in checks if hit)


def _named_hits(text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> tuple[str, ...]:
    return tuple(name for name, pattern in patterns if pattern.search(text))


def _categories(encodings: int, instructions: int, injections: int) -> tuple[str, ...]:
    found = [
        category.value
        for category, count in (
            (RiskCategory.ENCODED_CONTENT, encodings),
            (RiskCategory.DECODE_INSTRUCTION, instructions),
            (RiskCategory.PROMPT_INJECTION, injections),
        )
        if count
    ]
    return tuple(found) if found else (RiskCategory.NONE.value,)


class PatternClassifier:
    """Scores a prompt by the encoding and injection patterns it exhibits."""

    name = "pattern"

    def score(self, text: str) -> RiskScore:
        encodings = _encoding_hits(text)
        instructions = _named_hits(text, DECODE_INSTRUCTIONS)
        injections = _named_hits(text, INJECTION_PHRASES)
        total = (
            WEIGHT_PER_ENCODING * len(encodings)
            + WEIGHT_PER_INSTRUCTION * len(instructions)
            + WEIGHT_PER_INJECTION * len(injections)
        )
        return RiskScore(
            score=min(1.0, total),
            categories=_categories(len(encodings), len(instructions), len(injections)),
            evidence=encodings + instructions + injections,
            classifier=self.name,
        )
