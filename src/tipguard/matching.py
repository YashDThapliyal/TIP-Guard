"""Shared text-normalisation rules for finding a protected value in text.

Both `tipguard.logging.redact` and `tipguard.evaluation.leak.detect_leak`
have to decide where, or whether, a protected value appears in a piece of
text, and they have to agree: a value the detector calls a leak but the
redactor leaves in the clear is a value that reaches a trace or a report.
Three review rounds went into two implementations of the same rule drifting
apart, so the rule lives here once and both import it.

This module deliberately depends on nothing but `re`. `logging` sits below
`evaluation` in the project's dependency order -- `evaluation.runner`
imports the logger -- so the rule cannot live in `evaluation.leak` without
inverting that, and a leaf module is what lets both sides share it without
either importing the other.
"""

import re

#: Below this squashed length, a punctuation- and whitespace-insensitive
#: match is too permissive: a short alphanumeric run collides with unrelated
#: text often enough to matter, so short values are found only as whole
#: words. One constant, read by both callers, so the threshold cannot drift.
SQUASH_MIN_LENGTH = 10


def squash(text: str) -> str:
    """`text` casefolded, with everything but alphanumerics discarded.

    Casefolding rather than lowercasing, because only casefolding equates
    "straße" with "strasse"; note it can lengthen a string, which is why a
    caller mapping positions back to the original text must account for one
    source character becoming several.
    """
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def exact_word_pattern(value: str) -> str:
    """A pattern matching `value` as a whole word, for casefolded text.

    `[^\\W_]` is Unicode word characters minus underscore -- letters and
    digits only -- so this holds for non-ASCII scripts (it will not treat
    "東京" as a match inside "東京都") while still treating "_" as a
    boundary, unlike bare `\\w`, so "ref_4471-ZED_end" still matches.

    The value is casefolded here rather than matched with `re.IGNORECASE`,
    so a caller must casefold the text it searches too: `re.IGNORECASE` is
    not correct Unicode case-insensitivity on its own.
    """
    return rf"(?<![^\W_]){re.escape(value.casefold())}(?![^\W_])"
