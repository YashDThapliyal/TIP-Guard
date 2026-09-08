"""Registry of secret-encoding transformations used to build benchmark cases.

Seven families need no arguments and live in `TRANSFORMATIONS`. The two
bank-backed families (`riddle`, `indirect`) are built by
`build_riddle_transformations` instead, so that importing this module never
reads a data file.
"""

from tipguard.benchmark.transformations.base import Encoded, Family, Transformation
from tipguard.benchmark.transformations.base64_t import Base64Transformation
from tipguard.benchmark.transformations.caesar import CaesarTransformation
from tipguard.benchmark.transformations.code_snippet import CodeSnippetTransformation
from tipguard.benchmark.transformations.morse import MorseTransformation
from tipguard.benchmark.transformations.multi_step import MultiStepTransformation
from tipguard.benchmark.transformations.reverse import ReverseTransformation
from tipguard.benchmark.transformations.riddle import (
    IndirectTransformation,
    RiddleBank,
    RiddleTransformation,
    load_riddle_bank,
)
from tipguard.benchmark.transformations.substitution import SubstitutionTransformation

TRANSFORMATIONS: dict[Family, Transformation] = {
    Family.BASE64: Base64Transformation(),
    Family.CAESAR: CaesarTransformation(),
    Family.REVERSE: ReverseTransformation(),
    Family.MORSE: MorseTransformation(),
    Family.SUBSTITUTION: SubstitutionTransformation(),
    Family.CODE: CodeSnippetTransformation(),
    Family.MULTI_STEP: MultiStepTransformation(),
}


def get_transformation(family: Family) -> Transformation:
    """The argument-free transformation for `family`.

    Raises `KeyError` for `riddle`, `indirect` and `none`; the first two
    come from `build_riddle_transformations`.
    """
    return TRANSFORMATIONS[family]


def build_riddle_transformations(bank: RiddleBank) -> dict[Family, Transformation]:
    """The two bank-backed transformations, built against `bank`."""
    return {
        Family.RIDDLE: RiddleTransformation(bank),
        Family.INDIRECT: IndirectTransformation(bank),
    }


__all__ = [
    "TRANSFORMATIONS",
    "Encoded",
    "Family",
    "RiddleBank",
    "Transformation",
    "build_riddle_transformations",
    "get_transformation",
    "load_riddle_bank",
]
