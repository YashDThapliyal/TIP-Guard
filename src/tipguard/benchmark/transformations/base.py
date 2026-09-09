"""Base types shared by every transformation encoder."""

import random
from enum import StrEnum
from typing import Protocol

from pydantic import Field

from tipguard.base import FrozenModel


class Family(StrEnum):
    BASE64 = "base64"
    CAESAR = "caesar"
    REVERSE = "reverse"
    MORSE = "morse"
    SUBSTITUTION = "substitution"
    CODE = "code"
    MULTI_STEP = "multi_step"
    RIDDLE = "riddle"
    INDIRECT = "indirect"
    NONE = "none"


class Encoded(FrozenModel):
    """The result of applying a transformation to a secret payload.

    `hint` is a short human phrase naming the transformation, used by
    difficulty-level wrappers, e.g. "base64" or
    "a Caesar cipher shifted forward by 3".
    """

    payload: str
    family: Family
    params: dict[str, str] = Field(default_factory=dict)
    hint: str


class Transformation(Protocol):
    """A deterministic or non-deterministic secret-encoding transformation.

    A non-deterministic transformation's `decode` raises `NotImplementedError`.
    """

    family: Family
    deterministic: bool

    def encode(self, text: str, rng: random.Random) -> Encoded: ...

    def decode(self, encoded: Encoded) -> str: ...
