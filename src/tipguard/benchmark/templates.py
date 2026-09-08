"""The handwritten template bank the benchmark generator draws its material from.

A generated case is assembled from three ingredients that live here: an
*intent* phrasing that asks for a protected asset by label, a *transformation*
that hides that phrasing (see `transformations/`), and a *difficulty wrapper*
that frames the resulting puzzle. The bank also carries the benign payloads
used for the control cases, the two framings, and the hard negatives.

Every string in the files under `data/templates/` is handwritten. Nothing in
them may contain a protected value: the labels from `configs/policies.yaml`
are the only way a template refers to a protected asset.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Formatter
from typing import Literal, get_args

from pydantic import Field, ValidationError, ValidationInfo, field_validator

from tipguard.benchmark.transformations import BANK_BACKED_FAMILIES, Family
from tipguard.benchmark.transformations.riddle import RiddleBank, load_riddle_bank
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import FrozenModel, NonEmptyStr, PoliciesConfig

MIN_PHRASINGS = 20
MIN_BENIGN_PAYLOADS = 40
MIN_FRAMING_VARIANTS = 3
MIN_TEMPLATES_PER_LEVEL = 4
MIN_DISTRACTORS = 6
MIN_HARD_NEGATIVES = 60
MIN_PER_CATEGORY = 10

LABEL_PLACEHOLDER = "{label}"
BODY_PLACEHOLDER = "{body}"
BENIGN_CATEGORY = "benign"

FramingId = Literal["first-person", "third-person"]

#: Which vocabulary a difficulty template may use. `riddle` and `indirect`
#: payloads are ordinary prose that was never encoded, so telling their reader
#: to decode or undo anything is incoherent; every other family really was
#: transformed. `both` is wording that is true either way.
TemplateKind = Literal["encoded", "semantic"]
TemplateSuits = Literal["encoded", "semantic", "both"]
TEMPLATE_KINDS: tuple[TemplateKind, ...] = get_args(TemplateKind)
HardNegativeCategory = Literal[
    "academic", "classification", "defensive", "quoted", "suspicion_check"
]

REQUIRED_FRAMING_IDS: tuple[FramingId, ...] = get_args(FramingId)
HARD_NEGATIVE_CATEGORIES: tuple[HardNegativeCategory, ...] = get_args(HardNegativeCategory)

#: The placeholders each difficulty level's templates must use, exactly.
#: Level 1 names the transformation and the goal, level 2 only the
#: transformation, level 3 neither, and level 4 buries the payload in prose.
LEVEL_PLACEHOLDERS: Mapping[int, frozenset[str]] = {
    1: frozenset({"hint", "payload", "intent_hint"}),
    2: frozenset({"hint", "payload"}),
    3: frozenset({"payload"}),
    4: frozenset({"payload", "distractor"}),
}


def kind_for(family: Family) -> TemplateKind:
    """Which difficulty vocabulary suits `family`."""
    return "semantic" if family in BANK_BACKED_FAMILIES else "encoded"


INTENTS_FILE = "intents.yaml"
PAYLOADS_FILE = "benign_payloads.yaml"
FRAMINGS_FILE = "framings.yaml"
DIFFICULTY_FILE = "difficulty.yaml"
HARD_NEGATIVES_FILE = "hard_negatives.yaml"
RIDDLES_FILE = "riddles.yaml"


def placeholders(template: str) -> frozenset[str]:
    """The `{name}` field names a template uses, as a set."""
    return frozenset(name for _, name, _, _ in Formatter().parse(template) if name)


class IntentTemplate(FrozenModel):
    """One policy's ways of asking for its protected asset.

    Every phrasing refers to the asset through `{label}` and never by value,
    so the same phrasings can be reused for any policy-shaped asset.
    """

    intent_id: NonEmptyStr
    policy_id: NonEmptyStr
    canonical_intent: NonEmptyStr
    phrasings: list[NonEmptyStr] = Field(min_length=MIN_PHRASINGS)

    @field_validator("phrasings")
    @classmethod
    def _every_phrasing_names_the_label(cls, value: list[str]) -> list[str]:
        """Exactly `{label}` and nothing else.

        A substring test would accept `{{label}}`, which renders as a literal
        `{label}`, and `{label} {typo}`, which raises `KeyError` at format
        time. Comparing parsed field names catches both.
        """
        for phrasing in value:
            if placeholders(phrasing) != frozenset({"label"}):
                raise ValueError(
                    f"phrasing must use exactly the {LABEL_PLACEHOLDER} placeholder "
                    f"and no other: {phrasing!r}"
                )
        return value


class BenignPayload(FrozenModel):
    """A harmless instruction used for the benign-transformation control cases."""

    payload_id: NonEmptyStr
    instruction: NonEmptyStr
    expected_answer: NonEmptyStr
    topic: NonEmptyStr


class Framing(FrozenModel):
    """How a request reaches the assistant: as the user's own, or relayed."""

    framing_id: FramingId
    template: NonEmptyStr

    @field_validator("template")
    @classmethod
    def _carries_the_body(cls, value: str) -> str:
        if placeholders(value) != frozenset({"body"}):
            raise ValueError(
                f"framing template must use exactly the {BODY_PLACEHOLDER} placeholder "
                f"and no other: {value!r}"
            )
        return value


class DifficultyTemplate(FrozenModel):
    """One wrapper template, and which kind of family its wording fits."""

    template: NonEmptyStr
    suits: TemplateSuits

    def fits(self, kind: TemplateKind) -> bool:
        return self.suits in (kind, "both")


class DifficultyWrappers(FrozenModel):
    """The four levels of puzzle framing, the goal clauses, and the prose
    level 4 hides its payload in."""

    intent_hints: dict[str, NonEmptyStr] = Field(min_length=1)
    level_1: list[DifficultyTemplate] = Field(min_length=MIN_TEMPLATES_PER_LEVEL)
    level_2: list[DifficultyTemplate] = Field(min_length=MIN_TEMPLATES_PER_LEVEL)
    level_3: list[DifficultyTemplate] = Field(min_length=MIN_TEMPLATES_PER_LEVEL)
    level_4: list[DifficultyTemplate] = Field(min_length=MIN_TEMPLATES_PER_LEVEL)
    distractors: list[NonEmptyStr] = Field(min_length=MIN_DISTRACTORS)

    def templates_for(
        self, level: int, kind: TemplateKind | None = None
    ) -> list[DifficultyTemplate]:
        """The templates for `level`, restricted to those suiting `kind`.

        `KeyError` for a level that does not exist.
        """
        if level not in LEVEL_PLACEHOLDERS:
            raise KeyError(level)
        templates: list[DifficultyTemplate] = getattr(self, f"level_{level}")
        if kind is None:
            return templates
        return [template for template in templates if template.fits(kind)]

    def intent_hint_for(self, category: str) -> str:
        """The plain-language goal clause for `category`."""
        try:
            return self.intent_hints[category]
        except KeyError as exc:
            raise ConfigError(f"no intent hint for goal category {category!r}") from exc

    @field_validator("intent_hints")
    @classmethod
    def _hints_read_as_mid_sentence_clauses(cls, value: dict[str, str]) -> dict[str, str]:
        """Level 1 templates place the clause mid-sentence and start a new
        sentence after it, so it must begin lowercase and end with a stop."""
        for category, hint in value.items():
            if not hint[0].islower() or not hint.endswith("."):
                raise ValueError(
                    f"intent hint for {category!r} must start lowercase and end "
                    f"with a full stop: {hint!r}"
                )
        return value

    @field_validator("level_1", "level_2", "level_3", "level_4")
    @classmethod
    def _uses_exactly_its_placeholders(
        cls, value: list[DifficultyTemplate], info: ValidationInfo
    ) -> list[DifficultyTemplate]:
        level = int(str(info.field_name).removeprefix("level_"))
        required = LEVEL_PLACEHOLDERS[level]
        for entry in value:
            found = placeholders(entry.template)
            if found != required:
                raise ValueError(
                    f"template must use exactly {sorted(required)}, "
                    f"found {sorted(found)}: {entry.template[:60]!r}"
                )
        return value

    @field_validator("level_1", "level_2", "level_3", "level_4")
    @classmethod
    def _offers_enough_of_each_kind(
        cls, value: list[DifficultyTemplate], info: ValidationInfo
    ) -> list[DifficultyTemplate]:
        for kind in TEMPLATE_KINDS:
            fitting = sum(1 for entry in value if entry.fits(kind))
            if fitting < MIN_TEMPLATES_PER_LEVEL:
                raise ValueError(
                    f"has {fitting} templates suiting {kind} families, "
                    f"needs at least {MIN_TEMPLATES_PER_LEVEL}"
                )
        return value


class HardNegative(FrozenModel):
    """A prompt that talks about attacks or secrets and must still be allowed."""

    hn_id: NonEmptyStr
    prompt: NonEmptyStr
    canonical_intent: NonEmptyStr
    category: HardNegativeCategory


class _IntentsFile(FrozenModel):
    intents: list[IntentTemplate] = Field(min_length=1)


class _PayloadsFile(FrozenModel):
    payloads: list[BenignPayload] = Field(min_length=MIN_BENIGN_PAYLOADS)


class _FramingsFile(FrozenModel):
    framings: dict[str, list[NonEmptyStr]] = Field(min_length=1)


class _HardNegativesFile(FrozenModel):
    hard_negatives: list[HardNegative] = Field(min_length=MIN_HARD_NEGATIVES)


def _check_intents_match_policies(
    path: Path, intents: Sequence[IntentTemplate], policies: PoliciesConfig
) -> None:
    known = {policy.policy_id for policy in policies.policies}
    covered = {intent.policy_id for intent in intents}
    missing = sorted(known - covered)
    if missing:
        raise ConfigError(f"{path}: no intent for {', '.join(missing)}")
    unknown = sorted(covered - known)
    if unknown:
        raise ConfigError(f"{path}: intent for unknown policy {', '.join(unknown)}")
    if len(covered) != len(intents):
        raise ConfigError(f"{path}: more than one intent for the same policy")


def _build_framings(path: Path, raw: Mapping[str, Sequence[str]]) -> dict[str, list[Framing]]:
    built: dict[str, list[Framing]] = {}
    for framing_id in REQUIRED_FRAMING_IDS:
        variants = raw.get(framing_id, ())
        if len(variants) < MIN_FRAMING_VARIANTS:
            raise ConfigError(
                f"{path}: framings.{framing_id} has {len(variants)} variants, "
                f"needs at least {MIN_FRAMING_VARIANTS}"
            )
        try:
            built[framing_id] = [
                Framing(framing_id=framing_id, template=template) for template in variants
            ]
        except ValidationError as exc:
            raise ConfigError(f"{path}: framings.{framing_id}: {exc.errors()[0]['msg']}") from exc
    unknown = sorted(set(raw) - set(REQUIRED_FRAMING_IDS))
    if unknown:
        raise ConfigError(f"{path}: unknown framing id {', '.join(unknown)}")
    return built


def _check_intent_hints_cover_policies(
    path: Path, wrappers: "DifficultyWrappers", policies: PoliciesConfig
) -> None:
    """Every goal category a policy can present, plus benign, needs a clause."""
    needed = {policy.categories[0] for policy in policies.policies if policy.categories}
    needed.add(BENIGN_CATEGORY)
    missing = sorted(needed - set(wrappers.intent_hints))
    if missing:
        raise ConfigError(f"{path}: no intent_hints entry for {', '.join(missing)}")


def _check_hard_negative_categories(path: Path, hard_negatives: Sequence[HardNegative]) -> None:
    for category in HARD_NEGATIVE_CATEGORIES:
        found = sum(1 for hn in hard_negatives if hn.category == category)
        if found < MIN_PER_CATEGORY:
            raise ConfigError(
                f"{path}: category {category} has {found} entries, "
                f"needs at least {MIN_PER_CATEGORY}"
            )


class TemplateBank(FrozenModel):
    """Every handwritten ingredient the generator needs, loaded and checked."""

    intents: list[IntentTemplate]
    benign_payloads: list[BenignPayload]
    framings: dict[str, list[Framing]]
    difficulty: DifficultyWrappers
    hard_negatives: list[HardNegative]
    riddles: RiddleBank

    @classmethod
    def load(cls, templates_dir: Path, policies: PoliciesConfig) -> "TemplateBank":
        """Load the whole bank, checking it against the policies it serves.

        Any shortfall raises `ConfigError` naming the file and the field or
        category that was short or missing.
        """
        intents_path = templates_dir / INTENTS_FILE
        intents = load_yaml_model(intents_path, _IntentsFile).intents
        _check_intents_match_policies(intents_path, intents, policies)

        framings_path = templates_dir / FRAMINGS_FILE
        framings_file = load_yaml_model(framings_path, _FramingsFile)

        hard_negatives_path = templates_dir / HARD_NEGATIVES_FILE
        hard_negatives = load_yaml_model(hard_negatives_path, _HardNegativesFile).hard_negatives
        _check_hard_negative_categories(hard_negatives_path, hard_negatives)

        difficulty_path = templates_dir / DIFFICULTY_FILE
        difficulty = load_yaml_model(difficulty_path, DifficultyWrappers)
        _check_intent_hints_cover_policies(difficulty_path, difficulty, policies)

        labels = [policy.protected_label for policy in policies.policies]
        return cls(
            intents=intents,
            benign_payloads=load_yaml_model(templates_dir / PAYLOADS_FILE, _PayloadsFile).payloads,
            framings=_build_framings(framings_path, framings_file.framings),
            difficulty=difficulty,
            hard_negatives=hard_negatives,
            riddles=load_riddle_bank(templates_dir / RIDDLES_FILE, required_labels=labels),
        )

    def intent_for(self, policy_id: str) -> IntentTemplate:
        for intent in self.intents:
            if intent.policy_id == policy_id:
                return intent
        raise KeyError(policy_id)
