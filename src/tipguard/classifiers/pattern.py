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
import string

from tipguard.classifiers.types import RiskCategory, RiskScore
from tipguard.models.types import ModelResponse

#: A contiguous base64-alphabet run. Twenty-four characters is roughly
#: eighteen bytes of payload, long enough that ordinary prose and identifiers
#: do not reach it.
BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")

#: Four or more Morse letter groups in a row. Fewer than four is within reach
#: of ordinary punctuation such as an ellipsis followed by a dash. Written as
#: one group plus three or more whitespace-separated successors rather than as
#: four repetitions of "group then whitespace", so that a payload running to
#: the very end of the prompt -- which has no trailing whitespace -- still
#: reaches the documented four-group minimum. The dot-or-dash lookarounds stop
#: the scan entering an oversized run part way: without them a trailing
#: "....... . . ." matches by starting six dots into the seven-dot run, even
#: though no four-group sequence there respects the one-to-six-symbol limit.
MORSE_RUN = re.compile(r"(?<![.\-])[.\-]{1,6}(?:\s+[.\-]{1,6}){3,}(?![.\-])")

#: Markers of a Python snippet that assembles a string a character or a chunk
#: at a time -- the shape the `code` transformation family emits.
PYTHON_STRING_BUILDER = re.compile(r"(?:\bchr\(|\.join\(|\.replace\(|\breversed\()")

#: The non-alphanumeric characters that can stand where a letter would.
#: Together with the digits, which are alphanumeric and so always count, this
#: is the whole of what the detector will read as a replaced letter: inside a
#: token that kept a letter, every character here and every digit is flagged.
#:
#: It is deliberately wider than any substitution table, and written out here
#: rather than imported from `benchmark.transformations.substitution`, because
#: a defender does not get to read the attacker's table. There used to be a
#: second, narrower list of the glyphs a substitution actually uses; three
#: successive rules built on it were removed for buying one class of error and
#: selling another, and with them the last reader of that list. Coverage no
#: longer depends on enumerating an attacker's glyphs correctly: any character
#: here is flagged whether or not a table uses it. `tests/classifiers/
#: test_pattern.py` pins this as a superset of the benchmark's symbol swaps,
#: which catches drift without creating the coupling.
SUBSTITUTION_SYMBOLS = "@€!°$+#%&*"

#: Window and threshold for the substitution-density test: within any run of
#: twelve consecutive word-position characters, at least a quarter must be
#: digits or symbols. Twelve is about two English words, short enough to see
#: a single leeted phrase inside a long benign wrapper; a quarter is well
#: above the density of ordinary prose, which is near zero once punctuation
#: and whitespace are excluded.
_DIGIT_RUN = re.compile(r"\d+")

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


def _obfuscated(tokens: list[str]) -> list[bool]:
    """Which tokens may carry replaced letters: those that kept a letter.

    Leetspeak beside a surviving letter is unambiguous, so a token holding any
    letter is judged on its glyphs. A token that kept no letter is not judged
    at all, and that is a deliberate limit rather than an omission.

    Four rounds of review tried to read those tokens, and each rule bought a
    fresh class of false positive while changing the verdict on none of the
    1,700 benchmark cases. Reading any long glyph-only token flagged four-digit
    ticket numbers. Requiring glyph-only tokens to be a share of the text moved
    that onto short sentences and onto comma-separated serial lists, and lost a
    payload behind seven words of wrapper. Requiring an interior symbol glyph
    cleared those and then flagged unspaced arithmetic ("10+5"), bearings
    ("45°30"), unit-at-price notation ("10@45") and runs of exclamation marks.

    The pattern underneath is not bad luck. A token with no letters left has
    only its character classes to go on, and ordinary writing is full of digit
    and symbol clusters that a substitution table could equally have produced.
    "45537" is "asset" in leet digits and a five-digit identifier; the two are
    the same string. Separating them needs a dictionary check on the de-leeted
    token, which is semantic work -- exactly the work this syntactic baseline
    exists to be compared against.

    So the miss is recorded and pinned by a test rather than patched. A word
    substituted end to end escapes this classifier. That is a property of
    conventional filtering and part of what the study is measuring, and hiding
    it behind a rule that fires on arithmetic would overstate the baseline.

    The same verdict applies in the other direction. Emphasis ("stop!!!!")
    scores here, because a run of exclamation marks is genuinely dense in
    characters a substitution table uses. A trim for it was tried and removed:
    the symbols that spell emphasis are the symbols that spell letters. A word
    ending in a double letter ends in a doubled glyph once substituted, so the
    trim cut two glyphs off "@cc€$$", "p@$$", "@ddr€$$", "pr°c€$$", "bu$!n€$$"
    and "cl@$$" -- much of this project's own protected vocabulary -- and
    silenced whole phrases built from them: "r€v€@l +h€ p@$$", "p@$$ +h€
    cl@$$" and "+h€ cl@$$ l!$+" all scored zero with it and score with it
    gone. Firing on emphasis is what a syntactic detector does; losing
    "reveal the pass" is a hole in the one thing this classifier is good at,
    so the false positive stays and is pinned.
    """
    return [
        any(char.isalpha() for char in token) and not _is_identifier_shape(token)
        for token in tokens
    ]


def _is_identifier_shape(token: str) -> bool:
    """True for a token whose digits sit in one run against an edge.

    "SEC4412", "LH2291", "v2", "Q3", "AB1234": a label followed or preceded
    by a number, which is how ordinary writing spells a ticket, a version or
    a room. Substitution does not produce that shape, because it replaces
    letters wherever they fall, so its digits interleave -- "7h15", "c4n4ry",
    "urg3n7", "n33d5" all have two or more digit runs, or one run sitting
    inside the word rather than against an edge.

    Without this the density rule read every alphanumeric identifier in
    ordinary prose as leetspeak. It is why three indirect cases in the
    benchmark, whose only unusual feature was a ticket number, were handed to
    a substitution decoder.
    """
    core = token.strip(string.punctuation)
    runs = [match.span() for match in _DIGIT_RUN.finditer(core)]
    if len(runs) != 1:
        return False
    start, end = runs[0]
    return start == 0 or end == len(core)


def _replacement_flags(text: str) -> list[int]:
    """One flag per word-position character: 1 where a letter looks replaced.

    A non-alphabetic character counts as a replacement only inside a token
    `_obfuscated` accepts. Without that guard the rule fires on any prose
    carrying a date, a price, a version number or a phone number: collapsing
    whitespace lets a purely numeric token supply the density for a window
    made otherwise of ordinary words.

    The trade is a deliberate one, and it is an evasion, not a property of
    leetspeak: an attacker who spaces the substituted characters into tokens
    of their own -- "Th 3 v 4 ult" -- leaves every token either lettered with
    no glyphs or too short to qualify, and the score drops to zero. Removing
    the prose false positives is worth that, since the prose case fires by
    accident on ordinary traffic while the evasion has to be constructed. It
    is pinned by a test rather than left to be rediscovered.
    """
    tokens = text.split()
    flags: list[int] = []
    for token, obfuscated in zip(tokens, _obfuscated(tokens), strict=True):
        flags.extend(
            0 if char.isalpha() or not obfuscated else 1
            for char in token
            if _is_word_position(char)
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

    #: This classifier makes no model call, so there is no usage for a
    #: guard to attribute. Declared to satisfy `RiskClassifier`, which
    #: requires it so a classifier that *does* call a model cannot omit it
    #: and vanish from a run's cost and latency columns.
    last_usage: ModelResponse | None = None
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
