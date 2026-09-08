"""Registry of secret-encoding transformations used to build benchmark cases.

Seven families need no arguments and live in `TRANSFORMATIONS`. The two
bank-backed families (`riddle`, `indirect`) are built from a `RiddleBank`,
so that importing this module never reads a data file. `get_transformation`
resolves all nine attack families through one lookup: pass the bank and it
serves the bank-backed pair as well.
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


BANK_BACKED_FAMILIES = (Family.RIDDLE, Family.INDIRECT)


def get_transformation(family: Family, bank: RiddleBank | None = None) -> Transformation:
    """The transformation for `family`, the single lookup a generator needs.

    `riddle` and `indirect` are built from `bank`; requesting either without
    one raises `ValueError` naming the bank, since loading it here would put
    a file read behind an import. `KeyError` is raised for `none`.
    """
    if family in BANK_BACKED_FAMILIES:
        if bank is None:
            raise ValueError(
                f"the {family.value} transformation needs a RiddleBank: "
                f"call get_transformation({family.value!r}, load_riddle_bank(path))"
            )
        return build_riddle_transformations(bank)[family]
    return TRANSFORMATIONS[family]


def build_riddle_transformations(bank: RiddleBank) -> dict[Family, Transformation]:
    """The two bank-backed transformations, built against `bank`."""
    return {
        Family.RIDDLE: RiddleTransformation(bank),
        Family.INDIRECT: IndirectTransformation(bank),
    }


__all__ = [
    "BANK_BACKED_FAMILIES",
    "TRANSFORMATIONS",
    "Encoded",
    "Family",
    "RiddleBank",
    "Transformation",
    "build_riddle_transformations",
    "get_transformation",
    "load_riddle_bank",
]
