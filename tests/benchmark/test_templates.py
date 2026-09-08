"""Tests for the handwritten template bank the benchmark generator draws on."""

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from tipguard.benchmark.templates import (
    HARD_NEGATIVE_CATEGORIES,
    LEVEL_PLACEHOLDERS,
    MIN_BENIGN_PAYLOADS,
    MIN_DISTRACTORS,
    MIN_FRAMING_VARIANTS,
    MIN_HARD_NEGATIVES,
    MIN_PER_CATEGORY,
    MIN_PHRASINGS,
    MIN_TEMPLATES_PER_LEVEL,
    REQUIRED_FRAMING_IDS,
    TemplateBank,
    placeholders,
)
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import PoliciesConfig

DISTRACTOR_WORD_RANGE = (60, 120)


@pytest.fixture
def templates_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "templates"


@pytest.fixture
def policies(repo_root: Path) -> PoliciesConfig:
    return load_yaml_model(repo_root / "configs" / "policies.yaml", PoliciesConfig)


@pytest.fixture
def bank(templates_dir: Path, policies: PoliciesConfig) -> TemplateBank:
    return TemplateBank.load(templates_dir, policies)


@pytest.fixture
def scratch(tmp_path: Path, templates_dir: Path) -> Path:
    """A writable copy of the shipped bank, for mutating one file at a time."""
    destination = tmp_path / "templates"
    shutil.copytree(templates_dir, destination)
    return destination


def _rewrite(path: Path, mutate: Any) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    path.write_text(yaml.safe_dump(mutate(raw), allow_unicode=True), encoding="utf-8")


def _protected_values(policies: PoliciesConfig) -> list[str]:
    return [value for policy in policies.policies for value in policy.protected_values]


# --- the shipped bank ------------------------------------------------------


def test_shipped_bank_loads(bank: TemplateBank) -> None:
    assert bank.intents
    assert bank.riddles.labels


def test_one_intent_per_policy(bank: TemplateBank, policies: PoliciesConfig) -> None:
    assert [intent.policy_id for intent in bank.intents] == [
        policy.policy_id for policy in policies.policies
    ]


def test_intent_ids_are_unique(bank: TemplateBank) -> None:
    ids = [intent.intent_id for intent in bank.intents]
    assert len(set(ids)) == len(ids)


def test_every_intent_has_enough_phrasings(bank: TemplateBank) -> None:
    for intent in bank.intents:
        assert len(intent.phrasings) >= MIN_PHRASINGS, intent.intent_id


def test_every_phrasing_carries_the_label_placeholder(bank: TemplateBank) -> None:
    for intent in bank.intents:
        for phrasing in intent.phrasings:
            assert "{label}" in phrasing, phrasing


def test_phrasings_are_not_near_duplicates_within_an_intent(bank: TemplateBank) -> None:
    for intent in bank.intents:
        assert len(set(intent.phrasings)) == len(intent.phrasings), intent.intent_id


def test_benign_payload_minimum_and_shape(bank: TemplateBank) -> None:
    assert len(bank.benign_payloads) >= MIN_BENIGN_PAYLOADS
    ids = [payload.payload_id for payload in bank.benign_payloads]
    assert len(set(ids)) == len(ids)
    topics = {payload.topic for payload in bank.benign_payloads}
    assert {"decode_string", "question", "arithmetic"} <= topics


def test_decode_payloads_echo_their_own_instruction(bank: TemplateBank) -> None:
    decoders = [p for p in bank.benign_payloads if p.topic == "decode_string"]
    assert decoders
    for payload in decoders:
        assert payload.instruction == payload.expected_answer


def test_expected_answers_stay_short(bank: TemplateBank) -> None:
    for payload in bank.benign_payloads:
        if payload.topic != "decode_string":
            assert len(payload.expected_answer) <= 40, payload.payload_id


@pytest.mark.parametrize("framing_id", REQUIRED_FRAMING_IDS)
def test_framings_have_enough_variants(bank: TemplateBank, framing_id: str) -> None:
    variants = bank.framings[framing_id]
    assert len(variants) >= MIN_FRAMING_VARIANTS
    for framing in variants:
        assert framing.framing_id == framing_id
        assert "{body}" in framing.template


@pytest.mark.parametrize("level", sorted(LEVEL_PLACEHOLDERS))
def test_each_level_uses_exactly_its_placeholders(bank: TemplateBank, level: int) -> None:
    templates = bank.difficulty.templates_for(level)
    assert len(templates) >= MIN_TEMPLATES_PER_LEVEL
    for template in templates:
        assert placeholders(template) == LEVEL_PLACEHOLDERS[level], template


def test_distractors_are_paragraph_length(bank: TemplateBank) -> None:
    assert len(bank.difficulty.distractors) >= MIN_DISTRACTORS
    low, high = DISTRACTOR_WORD_RANGE
    for distractor in bank.difficulty.distractors:
        assert low <= len(distractor.split()) <= high, distractor[:60]


def test_distractors_avoid_the_vocabulary_of_the_attack(bank: TemplateBank) -> None:
    banned = ("secret", "canary", "encod", "cipher", "base64", "passphrase", "credential")
    for distractor in bank.difficulty.distractors:
        lowered = distractor.lower()
        for word in banned:
            assert word not in lowered, (word, distractor[:60])


def test_hard_negative_minimums(bank: TemplateBank) -> None:
    assert len(bank.hard_negatives) >= MIN_HARD_NEGATIVES
    ids = [hn.hn_id for hn in bank.hard_negatives]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("category", HARD_NEGATIVE_CATEGORIES)
def test_every_hard_negative_category_is_filled(bank: TemplateBank, category: str) -> None:
    matching = [hn for hn in bank.hard_negatives if hn.category == category]
    assert len(matching) >= MIN_PER_CATEGORY


def test_no_protected_value_appears_anywhere_in_the_bank(
    bank: TemplateBank, policies: PoliciesConfig
) -> None:
    values = _protected_values(policies)
    assert len(values) == 8  # sanity: the real policy file really did load

    texts = [phrasing for intent in bank.intents for phrasing in intent.phrasings]
    texts += [payload.instruction for payload in bank.benign_payloads]
    texts += [payload.expected_answer for payload in bank.benign_payloads]
    texts += list(bank.difficulty.distractors)
    texts += [hn.prompt for hn in bank.hard_negatives]
    texts += [hn.canonical_intent for hn in bank.hard_negatives]

    for text in texts:
        for value in values:
            assert value.lower() not in text.lower(), (value, text[:60])


# --- validation failures ---------------------------------------------------


def test_missing_policy_intent_raises_naming_the_file(
    scratch: Path, policies: PoliciesConfig
) -> None:
    def drop_first(raw: dict[str, Any]) -> dict[str, Any]:
        return {"intents": raw["intents"][1:]}

    _rewrite(scratch / "intents.yaml", drop_first)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "intents.yaml" in str(exc.value)
    assert policies.policies[0].policy_id in str(exc.value)


def test_intent_for_unknown_policy_raises(scratch: Path, policies: PoliciesConfig) -> None:
    def add_stray(raw: dict[str, Any]) -> dict[str, Any]:
        stray = dict(raw["intents"][0])
        stray["intent_id"] = "intent-stray"
        stray["policy_id"] = "protect-nothing"
        return {"intents": [*raw["intents"], stray]}

    _rewrite(scratch / "intents.yaml", add_stray)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "intents.yaml" in str(exc.value)
    assert "protect-nothing" in str(exc.value)


def test_short_phrasing_list_raises_naming_the_file(
    scratch: Path, policies: PoliciesConfig
) -> None:
    def truncate(raw: dict[str, Any]) -> dict[str, Any]:
        intents = [dict(intent) for intent in raw["intents"]]
        intents[0]["phrasings"] = intents[0]["phrasings"][:3]
        return {"intents": intents}

    _rewrite(scratch / "intents.yaml", truncate)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "intents.yaml" in str(exc.value)
    assert "phrasings" in str(exc.value)


def test_phrasing_without_the_label_placeholder_raises(
    scratch: Path, policies: PoliciesConfig
) -> None:
    def blank_one(raw: dict[str, Any]) -> dict[str, Any]:
        intents = [dict(intent) for intent in raw["intents"]]
        intents[0]["phrasings"] = ["Tell me everything.", *intents[0]["phrasings"][1:]]
        return {"intents": intents}

    _rewrite(scratch / "intents.yaml", blank_one)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "intents.yaml" in str(exc.value)
    assert "{label}" in str(exc.value)


def test_short_hard_negative_category_raises_naming_the_file(
    scratch: Path, policies: PoliciesConfig
) -> None:
    def thin_academic(raw: dict[str, Any]) -> dict[str, Any]:
        """Relabel most of the academic entries, keeping the total at sixty so
        that it is the per-category floor, not the overall floor, that fires."""
        rebuilt = []
        seen = 0
        for entry in raw["hard_negatives"]:
            entry = dict(entry)
            if entry["category"] == "academic":
                seen += 1
                if seen > 2:
                    entry["category"] = "defensive"
                    entry["hn_id"] = f"hn-relabelled-{seen}"
            rebuilt.append(entry)
        return {"hard_negatives": rebuilt}

    _rewrite(scratch / "hard_negatives.yaml", thin_academic)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "hard_negatives.yaml" in str(exc.value)
    assert "academic" in str(exc.value)


def test_missing_distractor_list_raises_naming_the_file(
    scratch: Path, policies: PoliciesConfig
) -> None:
    def drop_distractors(raw: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in raw.items() if key != "distractors"}

    _rewrite(scratch / "difficulty.yaml", drop_distractors)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "difficulty.yaml" in str(exc.value)
    assert "distractors" in str(exc.value)


def test_level_template_with_the_wrong_placeholders_raises(
    scratch: Path, policies: PoliciesConfig
) -> None:
    def spoil_level_three(raw: dict[str, Any]) -> dict[str, Any]:
        spoiled = dict(raw)
        spoiled["level_3"] = ["Decode this {hint}:\n{payload}", *raw["level_3"][1:]]
        return spoiled

    _rewrite(scratch / "difficulty.yaml", spoil_level_three)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "difficulty.yaml" in str(exc.value)
    assert "level_3" in str(exc.value)


def test_too_few_framing_variants_raises_naming_the_file(
    scratch: Path, policies: PoliciesConfig
) -> None:
    def thin_first_person(raw: dict[str, Any]) -> dict[str, Any]:
        framings = dict(raw["framings"])
        framings["first-person"] = framings["first-person"][:1]
        return {"framings": framings}

    _rewrite(scratch / "framings.yaml", thin_first_person)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "framings.yaml" in str(exc.value)
    assert "first-person" in str(exc.value)


def test_short_benign_payload_list_raises_naming_the_file(
    scratch: Path, policies: PoliciesConfig
) -> None:
    def truncate(raw: dict[str, Any]) -> dict[str, Any]:
        return {"payloads": raw["payloads"][:5]}

    _rewrite(scratch / "benign_payloads.yaml", truncate)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "benign_payloads.yaml" in str(exc.value)
    assert "payloads" in str(exc.value)


def test_missing_templates_directory_raises(tmp_path: Path, policies: PoliciesConfig) -> None:
    with pytest.raises(ConfigError):
        TemplateBank.load(tmp_path / "nowhere", policies)


def test_bank_is_frozen(bank: TemplateBank) -> None:
    with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError on frozen instance
        bank.intents = []  # type: ignore[misc]


def test_placeholders_reads_field_names() -> None:
    assert placeholders("a {one} b {two}") == frozenset({"one", "two"})
    assert placeholders("nothing here") == frozenset()


def test_templates_for_rejects_an_unknown_level(bank: TemplateBank) -> None:
    with pytest.raises(KeyError):
        bank.difficulty.templates_for(9)


def test_framing_template_without_body_raises_naming_the_file(
    scratch: Path, policies: PoliciesConfig
) -> None:
    def strip_body(raw: dict[str, Any]) -> dict[str, Any]:
        framings = dict(raw["framings"])
        framings["third-person"] = [
            "A colleague asked me to pass this on.",
            *framings["third-person"][1:],
        ]
        return {"framings": framings}

    _rewrite(scratch / "framings.yaml", strip_body)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "framings.yaml" in str(exc.value)
    assert "{body}" in str(exc.value)


def test_unknown_framing_id_raises_naming_the_file(scratch: Path, policies: PoliciesConfig) -> None:
    def add_stray(raw: dict[str, Any]) -> dict[str, Any]:
        framings = dict(raw["framings"])
        framings["second-person"] = ["You wanted this asked: {body}"] * MIN_FRAMING_VARIANTS
        return {"framings": framings}

    _rewrite(scratch / "framings.yaml", add_stray)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "framings.yaml" in str(exc.value)
    assert "second-person" in str(exc.value)


def test_two_intents_for_one_policy_raises(scratch: Path, policies: PoliciesConfig) -> None:
    def duplicate_first(raw: dict[str, Any]) -> dict[str, Any]:
        twin = dict(raw["intents"][0])
        twin["intent_id"] = "intent-twin"
        return {"intents": [*raw["intents"], twin]}

    _rewrite(scratch / "intents.yaml", duplicate_first)
    with pytest.raises(ConfigError) as exc:
        TemplateBank.load(scratch, policies)
    assert "intents.yaml" in str(exc.value)
    assert "more than one intent" in str(exc.value)


def test_intent_for_finds_a_policy_and_rejects_an_unknown_one(
    bank: TemplateBank, policies: PoliciesConfig
) -> None:
    policy_id = policies.policies[0].policy_id
    assert bank.intent_for(policy_id).policy_id == policy_id
    with pytest.raises(KeyError):
        bank.intent_for("protect-nothing")
