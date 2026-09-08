"""Registry of secret-encoding transformations used to build benchmark cases.

Task 2 registers the remaining families (`code`, `multi_step`, `riddle`,
`indirect`); requesting one of those before then raises `KeyError`.
"""

from tipguard.benchmark.transformations.base import Encoded, Family, Transformation
from tipguard.benchmark.transformations.base64_t import Base64Transformation
from tipguard.benchmark.transformations.caesar import CaesarTransformation
from tipguard.benchmark.transformations.morse import MorseTransformation
from tipguard.benchmark.transformations.reverse import ReverseTransformation
from tipguard.benchmark.transformations.substitution import SubstitutionTransformation

TRANSFORMATIONS: dict[Family, Transformation] = {
    Family.BASE64: Base64Transformation(),
    Family.CAESAR: CaesarTransformation(),
    Family.REVERSE: ReverseTransformation(),
    Family.MORSE: MorseTransformation(),
    Family.SUBSTITUTION: SubstitutionTransformation(),
}


def get_transformation(family: Family) -> Transformation:
    return TRANSFORMATIONS[family]


__all__ = [
    "TRANSFORMATIONS",
    "Encoded",
    "Family",
    "Transformation",
    "get_transformation",
]
