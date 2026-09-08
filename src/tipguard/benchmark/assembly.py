"""The pieces a generated prompt is assembled from, and the case ids it gets.

`generator.py` decides *which* case to build; this module knows *how* one is
put together: which article a protected label takes, how a framing variant is
drawn and applied, how a difficulty wrapper is filled, and how the running
per-group counter turns into a stable case id.
"""

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass

from tipguard.benchmark.schema import CaseType
from tipguard.benchmark.templates import (
    DifficultyWrappers,
    Framing,
    TemplateBank,
    TemplateKind,
    placeholders,
)

#: The determiner `intents.yaml` deliberately leaves off before `{label}`.
LABEL_ARTICLE = "the"

#: An instruction longer than this is a template bug, not a case worth
#: encoding: the code-snippet decoder refuses a reconstruction past 10,000
#: characters, and every handwritten phrasing is under 200.
MAX_INSTRUCTION_LENGTH = 1000

ID_COUNTER_WIDTH = 4


def check_instruction_length(text: str, source_id: str) -> None:
    """Refuse an instruction long enough to threaten a transformation's limits."""
    if len(text) > MAX_INSTRUCTION_LENGTH:
        raise ValueError(
            f"{source_id}: instruction is {len(text)} characters, over the "
            f"{MAX_INSTRUCTION_LENGTH}-character limit a benchmark instruction may take; "
            f"fix the template rather than the generator"
        )


def label_with_article(protected_label: str) -> str:
    """A policy's label as a phrasing expects to receive it, article included."""
    return f"{LABEL_ARTICLE} {protected_label}"


@dataclass(frozen=True)
class FramingChoice:
    """One drawn framing variant, and where it came from."""

    framing_id: str
    variant_index: int
    template: str

    def apply(self, body: str) -> str:
        return self.template.format(body=body)


def _choices_for(framings: Sequence[Framing], framing_id: str) -> list[FramingChoice]:
    return [
        FramingChoice(framing_id=framing_id, variant_index=index, template=framing.template)
        for index, framing in enumerate(framings)
    ]


def framing_choices(bank: TemplateBank, framing_id: str) -> list[FramingChoice]:
    """Every variant of one framing, in file order."""
    return _choices_for(bank.framings[framing_id], framing_id)


def all_framing_choices(bank: TemplateBank) -> list[FramingChoice]:
    """Every variant of every framing, ordered by framing id then file order."""
    return [
        choice
        for framing_id in sorted(bank.framings)
        for choice in framing_choices(bank, framing_id)
    ]


def draw_framing(bank: TemplateBank, framing_id: str, rng: random.Random) -> FramingChoice:
    return rng.choice(framing_choices(bank, framing_id))


def draw_any_framing(bank: TemplateBank, rng: random.Random) -> FramingChoice:
    return rng.choice(all_framing_choices(bank))


def wrap_payload(
    wrappers: DifficultyWrappers,
    level: int,
    kind: TemplateKind,
    rng: random.Random,
    *,
    payload: str,
    hint: str,
    intent_hint: str,
) -> str:
    """Fill a difficulty template for `level` that suits `kind`.

    Only the placeholders the level actually declares are consumed; the rest
    of the keyword arguments are ignored by `str.format`, which keeps the
    caller from having to branch on the level.
    """
    template = rng.choice(wrappers.templates_for(level, kind)).template
    distractor = rng.choice(wrappers.distractors) if "distractor" in placeholders(template) else ""
    return template.format(
        payload=payload, hint=hint, intent_hint=intent_hint, distractor=distractor
    )


def encoded_params_json(params: dict[str, str]) -> str:
    """Encoding parameters as the single string `metadata` can hold."""
    return json.dumps(params, sort_keys=True)


class CaseIdAllocator:
    """Four-digit counters, one per (type, transformation, difficulty) group."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str, int], int] = {}

    def next_id(self, case_type: CaseType, transformation: str, difficulty: int) -> str:
        group = (case_type.value, transformation, difficulty)
        number = self._counts.get(group, 0) + 1
        self._counts[group] = number
        return f"{case_type.value}-{transformation}-l{difficulty}-{number:0{ID_COUNTER_WIDTH}d}"
