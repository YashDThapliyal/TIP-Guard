"""Configuration for generating the TIP-Guard benchmark dataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

from pydantic import Field, ValidationInfo, field_validator

from tipguard.benchmark.templates import FramingId
from tipguard.benchmark.transformations.base import Family
from tipguard.config.schemas import FrozenModel

#: The families used for attack cases. `reverse` is excluded: reversing a
#: sentence hides nothing from a competent reader, so it is a benign control
#: rather than an attack.
ATTACK_FAMILIES: tuple[Family, ...] = (
    Family.BASE64,
    Family.CAESAR,
    Family.MORSE,
    Family.SUBSTITUTION,
    Family.CODE,
    Family.MULTI_STEP,
    Family.RIDDLE,
    Family.INDIRECT,
)

#: The families used for benign-transformation controls: every mechanically
#: decodable attack family, plus `reverse`. `riddle` and `indirect` are absent
#: because they encode a protected label, not an arbitrary benign payload.
BENIGN_FAMILIES: tuple[Family, ...] = (
    Family.BASE64,
    Family.CAESAR,
    Family.MORSE,
    Family.SUBSTITUTION,
    Family.CODE,
    Family.MULTI_STEP,
    Family.REVERSE,
)

DifficultyLevel = Literal[1, 2, 3, 4]

DIFFICULTY_LEVELS: tuple[DifficultyLevel, ...] = get_args(DifficultyLevel)
FRAMING_IDS: tuple[FramingId, ...] = get_args(FramingId)

#: Each list field, with the values it is allowed to draw on. Anything outside
#: these sets fails later with a bare `KeyError` from the transformation
#: registry or the template bank, so it is rejected at load time instead.
ALLOWED_VALUES: dict[str, tuple[object, ...]] = {
    "families": ATTACK_FAMILIES,
    "benign_families": BENIGN_FAMILIES,
    "difficulties": DIFFICULTY_LEVELS,
    "framings": FRAMING_IDS,
}


def _check_allowed(values: Sequence[object], field: str) -> None:
    """Non-empty, duplicate-free, and drawn only from the allowed set."""
    if not values:
        raise ValueError(f"{field} must not be empty")
    seen: set[object] = set()
    allowed = ALLOWED_VALUES[field]
    for value in values:
        if value in seen:
            raise ValueError(f"{field} lists {value!r} more than once")
        if value not in allowed:
            raise ValueError(
                f"{field} may not include {value!r}; allowed values are "
                f"{[str(item) for item in allowed]}"
            )
        seen.add(value)


class BenchmarkConfig(FrozenModel):
    """What to generate, from which templates, with which seed."""

    name: str = "tipguard-v1"
    seed: int = 20260907
    templates_dir: Path = Path("data/templates")
    policies_config: Path = Path("configs/policies.yaml")
    families: list[Family] = Field(default_factory=lambda: list(ATTACK_FAMILIES))
    benign_families: list[Family] = Field(default_factory=lambda: list(BENIGN_FAMILIES))
    difficulties: list[DifficultyLevel] = Field(default_factory=lambda: list(DIFFICULTY_LEVELS))
    framings: list[FramingId] = Field(default_factory=lambda: list(FRAMING_IDS))
    paraphrases_per_combination: int = Field(default=3, ge=1)
    benign_per_family_per_level: int = Field(default=12, ge=1)
    direct_per_policy: int = Field(default=8, ge=1)
    hard_negative_paraphrases: int = Field(default=3, ge=1)
    output: Path = Path("data/generated/tipguard-v1.jsonl")

    @field_validator("families", "benign_families", "difficulties", "framings", mode="before")
    @classmethod
    def _drawn_from_the_allowed_set(cls, value: object, info: ValidationInfo) -> object:
        """Checked before coercion, so the message can name the offending value
        rather than only the values that would have been acceptable."""
        if isinstance(value, list):
            _check_allowed(value, str(info.field_name))
        return value
