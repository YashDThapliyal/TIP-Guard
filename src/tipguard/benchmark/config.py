"""Configuration for generating the TIP-Guard benchmark dataset."""

from pathlib import Path

from pydantic import Field

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

DIFFICULTY_LEVELS: tuple[int, ...] = (1, 2, 3, 4)
FRAMING_IDS: tuple[str, ...] = ("first-person", "third-person")


class BenchmarkConfig(FrozenModel):
    """What to generate, from which templates, with which seed."""

    name: str = "tipguard-v1"
    seed: int = 20260907
    templates_dir: Path = Path("data/templates")
    policies_config: Path = Path("configs/policies.yaml")
    families: list[Family] = Field(default_factory=lambda: list(ATTACK_FAMILIES))
    benign_families: list[Family] = Field(default_factory=lambda: list(BENIGN_FAMILIES))
    difficulties: list[int] = Field(default_factory=lambda: list(DIFFICULTY_LEVELS))
    framings: list[str] = Field(default_factory=lambda: list(FRAMING_IDS))
    paraphrases_per_combination: int = Field(default=3, ge=1)
    benign_per_family_per_level: int = Field(default=12, ge=1)
    direct_per_policy: int = Field(default=8, ge=1)
    hard_negative_paraphrases: int = Field(default=3, ge=1)
    output: Path = Path("data/generated/tipguard-v1.jsonl")
