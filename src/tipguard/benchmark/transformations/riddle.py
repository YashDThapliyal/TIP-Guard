"""Riddle and indirect-description transformations backed by a YAML bank.

These two families are not encodings: nothing about the payload can be
mechanically decoded, so `deterministic` is False and `decode` raises.
Their `encode` receives a policy's `protected_label` rather than an
instruction, and draws one prewritten entry for that label.

Bank entries describe the protected *asset* and never its content: no
entry may contain a protected value or paraphrase one closely enough to
let a reader reconstruct it.
"""

import random
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from tipguard.base import FrozenModel
from tipguard.benchmark.transformations.base import Encoded, Family
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import NonEmptyStr

MINIMUM_ENTRIES = 6
DEFAULT_BANK_PATH = Path("data/templates/riddles.yaml")
RIDDLE_HINT = "a riddle"
INDIRECT_HINT = "an indirect description"


class LabelEntries(FrozenModel):
    """The prewritten riddles and indirect descriptions for one label."""

    riddles: list[NonEmptyStr] = Field(min_length=MINIMUM_ENTRIES)
    indirect: list[NonEmptyStr] = Field(min_length=MINIMUM_ENTRIES)


class RiddleBank(FrozenModel):
    labels: dict[str, LabelEntries] = Field(min_length=1)

    def entries_for(self, label: str) -> LabelEntries:
        try:
            return self.labels[label]
        except KeyError as exc:
            raise ConfigError(f"no riddle bank entries for label {label!r}") from exc


def load_riddle_bank(
    path: Path = DEFAULT_BANK_PATH,
    required_labels: Sequence[str] = (),
) -> RiddleBank:
    """Load the bank, requiring at least `MINIMUM_ENTRIES` of each kind per label."""
    bank = load_yaml_model(path, RiddleBank)
    missing = sorted(set(required_labels) - set(bank.labels))
    if missing:
        listed = ", ".join(repr(label) for label in missing)
        raise ConfigError(f"{path}: missing entries for {listed}")
    return bank


def _select(
    entries: Sequence[str], label: str, family: Family, hint: str, rng: random.Random
) -> Encoded:
    payload = str(rng.choice(sorted(entries)))
    return Encoded(payload=payload, family=family, params={"label": label}, hint=hint)


class RiddleTransformation:
    family = Family.RIDDLE
    deterministic = False

    def __init__(self, bank: RiddleBank) -> None:
        self._bank = bank

    def encode(self, text: str, rng: random.Random) -> Encoded:
        """`text` is a policy's protected label, not an instruction."""
        entries = self._bank.entries_for(text).riddles
        return _select(entries, text, self.family, RIDDLE_HINT, rng)

    def decode(self, encoded: Encoded) -> str:
        raise NotImplementedError("riddles have no mechanical decoding")


class IndirectTransformation:
    family = Family.INDIRECT
    deterministic = False

    def __init__(self, bank: RiddleBank) -> None:
        self._bank = bank

    def encode(self, text: str, rng: random.Random) -> Encoded:
        """`text` is a policy's protected label, not an instruction."""
        entries = self._bank.entries_for(text).indirect
        return _select(entries, text, self.family, INDIRECT_HINT, rng)

    def decode(self, encoded: Encoded) -> str:
        raise NotImplementedError("indirect descriptions have no mechanical decoding")
