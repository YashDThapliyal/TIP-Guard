"""Builds the benchmark dataset from the template bank and the policies.

Four blocks are produced, always in this order so that the per-group case-id
counters are stable across runs: direct requests, Task-in-Prompt attacks,
benign transformation controls, and hard negatives. One `random.Random`,
seeded from the configuration, is threaded through every draw, so the whole
dataset is a pure function of `(config, bank, policies)`.
"""

import itertools
import random
from collections.abc import Iterator
from dataclasses import dataclass

from tipguard.benchmark.assembly import (
    CaseIdAllocator,
    FramingChoice,
    all_framing_choices,
    check_instruction_length,
    draw_any_framing,
    draw_framing,
    encoded_params_json,
    label_with_article,
    wrap_payload,
)
from tipguard.benchmark.config import BenchmarkConfig, DifficultyLevel
from tipguard.benchmark.schema import (
    BenchmarkCase,
    CaseType,
    Decision,
    Split,
    hash_protected_value,
)
from tipguard.benchmark.templates import (
    BENIGN_CATEGORY,
    BenignPayload,
    FramingId,
    HardNegative,
    IntentTemplate,
    TemplateBank,
    kind_for,
)
from tipguard.benchmark.transformations import Family, Transformation, get_transformation
from tipguard.config.schemas import PoliciesConfig, Policy

#: Every case is generated into one bucket; Task 5 reassigns splits.
DEFAULT_SPLIT = Split.TEST

#: Benign controls have no policy, so their canonical intent is the same
#: sentence everywhere: decode the wrapper, then answer what is inside it.
BENIGN_INTENT = "Decode the transformed text and answer the harmless request it contains."

#: The framing a direct, unhidden request arrives in: the user's own words.
DIRECT_FRAMING: FramingId = "first-person"

NO_PARAMS = encoded_params_json({})


@dataclass(frozen=True)
class _Draw:
    """The material one case is built from, after every random choice is made."""

    text: str
    hint: str
    params: dict[str, str]
    source_index: int


@dataclass(frozen=True)
class _Context:
    """What every block needs: the bank, the shared rng, the id counters."""

    bank: TemplateBank
    rng: random.Random
    ids: CaseIdAllocator


def generate_cases(
    config: BenchmarkConfig, bank: TemplateBank, policies: PoliciesConfig
) -> tuple[BenchmarkCase, ...]:
    """Every case the configuration asks for, before deduplication."""
    context = _Context(bank=bank, rng=random.Random(config.seed), ids=CaseIdAllocator())
    return (
        *_direct_cases(config, context, policies),
        *_tip_cases(config, context, policies),
        *_benign_cases(config, context),
        *_hard_negative_cases(config, context),
    )


def _metadata(
    source_key: str, source_id: str, index: int, framing: FramingChoice, params: str
) -> dict[str, str]:
    return {
        source_key: source_id,
        "phrasing_index": str(index),
        "framing_variant": str(framing.variant_index),
        "encoded_params": params,
    }


def _direct_cases(
    config: BenchmarkConfig, context: _Context, policies: PoliciesConfig
) -> list[BenchmarkCase]:
    """A plain request for the protected asset, in the user's own voice."""
    cases: list[BenchmarkCase] = []
    for policy in policies.policies:
        intent = context.bank.intent_for(policy.policy_id)
        label = label_with_article(policy.protected_label)
        indices = context.rng.sample(range(len(intent.phrasings)), config.direct_per_policy)
        for index in indices:
            text = intent.phrasings[index].format(label=label)
            check_instruction_length(text, intent.intent_id)
            framing = draw_framing(context.bank, DIRECT_FRAMING, context.rng)
            cases.append(
                _prohibited_case(
                    context,
                    policy=policy,
                    case_type=CaseType.DIRECT,
                    transformation=Family.NONE.value,
                    difficulty=1,
                    framing=framing,
                    prompt=framing.apply(text),
                    canonical_intent=intent.canonical_intent,
                    metadata=_metadata("intent_id", intent.intent_id, index, framing, NO_PARAMS),
                )
            )
    return cases


def _combinations(
    config: BenchmarkConfig,
) -> Iterator[tuple[Family, DifficultyLevel, FramingId]]:
    """Family, difficulty and framing in configuration order."""
    return itertools.product(config.families, config.difficulties, config.framings)


def _tip_cases(
    config: BenchmarkConfig, context: _Context, policies: PoliciesConfig
) -> list[BenchmarkCase]:
    """The attack set: the same intents, hidden by a transformation."""
    transformations = {
        family: get_transformation(family, context.bank.riddles) for family in config.families
    }
    cases: list[BenchmarkCase] = []
    for policy in policies.policies:
        intent = context.bank.intent_for(policy.policy_id)
        for family, difficulty, framing_id in _combinations(config):
            phrasings = context.rng.sample(
                range(len(intent.phrasings)), config.paraphrases_per_combination
            )
            cases.extend(
                _tip_case(
                    context,
                    policy=policy,
                    intent=intent,
                    family=family,
                    transformation=transformations[family],
                    difficulty=difficulty,
                    framing_id=framing_id,
                    phrasing_index=index,
                )
                for index in phrasings
            )
    return cases


def _tip_case(
    context: _Context,
    *,
    policy: Policy,
    intent: IntentTemplate,
    family: Family,
    transformation: Transformation,
    difficulty: DifficultyLevel,
    framing_id: FramingId,
    phrasing_index: int,
) -> BenchmarkCase:
    draw = _tip_draw(context, policy, intent, family, transformation, phrasing_index)
    framing = draw_framing(context.bank, framing_id, context.rng)
    body = wrap_payload(
        context.bank.difficulty,
        difficulty,
        kind_for(family),
        context.rng,
        payload=draw.text,
        hint=draw.hint,
        intent_hint=context.bank.difficulty.intent_hint_for(policy.categories[0]),
    )
    return _prohibited_case(
        context,
        policy=policy,
        case_type=CaseType.TIP,
        transformation=family.value,
        difficulty=difficulty,
        framing=framing,
        prompt=framing.apply(body),
        canonical_intent=intent.canonical_intent,
        metadata=_metadata(
            "intent_id",
            intent.intent_id,
            draw.source_index,
            framing,
            encoded_params_json(draw.params),
        ),
    )


def _tip_draw(
    context: _Context,
    policy: Policy,
    intent: IntentTemplate,
    family: Family,
    transformation: Transformation,
    phrasing_index: int,
) -> _Draw:
    """The payload text for one attack case, and where it came from.

    `riddle` and `indirect` are not encodings: their `encode` takes the
    protected *label* and returns prewritten prose. A riddle is a whole
    prompt and stands alone; an indirect description is a noun phrase that
    replaces `{label}` in an ordinary phrasing, which is why `intents.yaml`
    writes no article before the placeholder.
    """
    if family is Family.RIDDLE:
        encoded = transformation.encode(policy.protected_label, context.rng)
        entries = sorted(context.bank.riddles.entries_for(policy.protected_label).riddles)
        return _Draw(encoded.payload, encoded.hint, encoded.params, entries.index(encoded.payload))
    if family is Family.INDIRECT:
        encoded = transformation.encode(policy.protected_label, context.rng)
        text = intent.phrasings[phrasing_index].format(label=encoded.payload)
        check_instruction_length(text, intent.intent_id)
        return _Draw(text, encoded.hint, encoded.params, phrasing_index)
    text = intent.phrasings[phrasing_index].format(label=label_with_article(policy.protected_label))
    check_instruction_length(text, intent.intent_id)
    encoded = transformation.encode(text, context.rng)
    return _Draw(encoded.payload, encoded.hint, encoded.params, phrasing_index)


def _prohibited_case(
    context: _Context,
    *,
    policy: Policy,
    case_type: CaseType,
    transformation: str,
    difficulty: int,
    framing: FramingChoice,
    prompt: str,
    canonical_intent: str,
    metadata: dict[str, str],
) -> BenchmarkCase:
    """A case the guardrail must block, with the hash the schema insists on."""
    return BenchmarkCase(
        case_id=context.ids.next_id(case_type, transformation, difficulty),
        policy_id=policy.policy_id,
        case_type=case_type,
        transformation=transformation,
        difficulty=difficulty,
        framing=framing.framing_id,
        prompt=prompt,
        canonical_intent=canonical_intent,
        expected_decision=Decision.BLOCK,
        protected_value_hash=hash_protected_value(policy.protected_values[0]),
        split=DEFAULT_SPLIT,
        metadata=metadata,
    )


def _benign_cases(config: BenchmarkConfig, context: _Context) -> list[BenchmarkCase]:
    """Controls: the same wrappers around a harmless instruction."""
    payload_index = {
        payload.payload_id: index for index, payload in enumerate(context.bank.benign_payloads)
    }
    cases: list[BenchmarkCase] = []
    for family, difficulty in itertools.product(config.benign_families, config.difficulties):
        transformation = get_transformation(family)
        drawn = context.rng.sample(context.bank.benign_payloads, config.benign_per_family_per_level)
        cases.extend(
            _benign_case(
                context,
                payload=payload,
                index=payload_index[payload.payload_id],
                family=family,
                transformation=transformation,
                difficulty=difficulty,
            )
            for payload in drawn
        )
    return cases


def _benign_case(
    context: _Context,
    *,
    payload: BenignPayload,
    index: int,
    family: Family,
    transformation: Transformation,
    difficulty: DifficultyLevel,
) -> BenchmarkCase:
    check_instruction_length(payload.instruction, payload.payload_id)
    encoded = transformation.encode(payload.instruction, context.rng)
    body = wrap_payload(
        context.bank.difficulty,
        difficulty,
        kind_for(family),
        context.rng,
        payload=encoded.payload,
        hint=encoded.hint,
        intent_hint=context.bank.difficulty.intent_hint_for(BENIGN_CATEGORY),
    )
    framing = draw_any_framing(context.bank, context.rng)
    metadata = _metadata(
        "payload_id", payload.payload_id, index, framing, encoded_params_json(encoded.params)
    )
    return BenchmarkCase(
        case_id=context.ids.next_id(CaseType.BENIGN_TRANSFORMATION, family.value, difficulty),
        policy_id=None,
        case_type=CaseType.BENIGN_TRANSFORMATION,
        transformation=family.value,
        difficulty=difficulty,
        framing=framing.framing_id,
        prompt=framing.apply(body),
        canonical_intent=BENIGN_INTENT,
        expected_decision=Decision.ALLOW,
        protected_value_hash=None,
        split=DEFAULT_SPLIT,
        expected_answer=payload.expected_answer,
        metadata={**metadata, "topic": payload.topic},
    )


def _hard_negative_cases(config: BenchmarkConfig, context: _Context) -> list[BenchmarkCase]:
    """Allowable prompts that talk about attacks, one per distinct framing."""
    variants = all_framing_choices(context.bank)
    cases: list[BenchmarkCase] = []
    for index, negative in enumerate(context.bank.hard_negatives):
        framings = context.rng.sample(variants, config.hard_negative_paraphrases)
        cases.extend(
            _hard_negative_case(context, negative=negative, index=index, framing=framing)
            for framing in framings
        )
    return cases


def _hard_negative_case(
    context: _Context, *, negative: HardNegative, index: int, framing: FramingChoice
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=context.ids.next_id(CaseType.HARD_NEGATIVE, Family.NONE.value, 1),
        policy_id=None,
        case_type=CaseType.HARD_NEGATIVE,
        transformation=Family.NONE.value,
        difficulty=1,
        framing=framing.framing_id,
        prompt=framing.apply(negative.prompt),
        canonical_intent=negative.canonical_intent,
        expected_decision=Decision.ALLOW,
        protected_value_hash=None,
        split=DEFAULT_SPLIT,
        metadata={
            **_metadata("hn_id", negative.hn_id, index, framing, NO_PARAMS),
            "category": negative.category,
        },
    )
