"""Tests for the riddle bank and the two non-deterministic transformations."""

import random
from pathlib import Path

import pytest

from tipguard.benchmark.transformations import (
    TRANSFORMATIONS,
    build_riddle_transformations,
    get_transformation,
)
from tipguard.benchmark.transformations.base import Encoded, Family
from tipguard.benchmark.transformations.riddle import (
    MINIMUM_ENTRIES,
    IndirectTransformation,
    RiddleBank,
    RiddleTransformation,
    load_riddle_bank,
)
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import PoliciesConfig

LABELS = (
    "system canary",
    "customer record",
    "internal API token",
    "vault passphrase",
    "internal admin URL",
    "product codename",
)

ARGUMENT_FREE_FAMILIES = (
    Family.BASE64,
    Family.CAESAR,
    Family.REVERSE,
    Family.MORSE,
    Family.SUBSTITUTION,
    Family.CODE,
    Family.MULTI_STEP,
)


@pytest.fixture
def bank_path(repo_root: Path) -> Path:
    return repo_root / "data" / "templates" / "riddles.yaml"


@pytest.fixture
def bank(bank_path: Path) -> RiddleBank:
    return load_riddle_bank(bank_path, required_labels=LABELS)


@pytest.fixture
def policies(repo_root: Path) -> PoliciesConfig:
    return load_yaml_model(repo_root / "configs" / "policies.yaml", PoliciesConfig)


def test_bank_covers_every_policy_label(bank: RiddleBank, policies: PoliciesConfig) -> None:
    assert {policy.protected_label for policy in policies.policies} == set(LABELS)
    assert set(bank.labels) == set(LABELS)


@pytest.mark.parametrize("label", LABELS)
def test_bank_has_at_least_six_of_each_kind(bank: RiddleBank, label: str) -> None:
    entries = bank.entries_for(label)

    assert len(entries.riddles) >= MINIMUM_ENTRIES
    assert len(entries.indirect) >= MINIMUM_ENTRIES
    assert len(set(entries.riddles)) == len(entries.riddles)
    assert len(set(entries.indirect)) == len(entries.indirect)
    assert all(entry.strip() for entry in entries.riddles + entries.indirect)


def test_no_entry_contains_a_protected_value(bank: RiddleBank, policies: PoliciesConfig) -> None:
    values = [value.lower() for policy in policies.policies for value in policy.protected_values]
    assert values  # sanity: the policies file really did load
    for label, entries in bank.labels.items():
        for entry in entries.riddles + entries.indirect:
            lowered = entry.lower()
            for value in values:
                assert value not in lowered, f"{label} entry leaks a protected value"


@pytest.mark.parametrize("label", LABELS)
def test_riddle_encode_returns_a_bank_entry(bank: RiddleBank, label: str) -> None:
    transformation = RiddleTransformation(bank)

    encoded = transformation.encode(label, random.Random(0))

    assert encoded.family is Family.RIDDLE
    assert encoded.hint == "a riddle"
    assert encoded.payload in bank.entries_for(label).riddles


@pytest.mark.parametrize("label", LABELS)
def test_indirect_encode_returns_a_bank_entry(bank: RiddleBank, label: str) -> None:
    transformation = IndirectTransformation(bank)

    encoded = transformation.encode(label, random.Random(0))

    assert encoded.family is Family.INDIRECT
    assert encoded.hint == "an indirect description"
    assert encoded.payload in bank.entries_for(label).indirect


def test_encode_is_deterministic_for_a_fixed_seed(bank: RiddleBank) -> None:
    for transformation in (RiddleTransformation(bank), IndirectTransformation(bank)):
        first = transformation.encode("vault passphrase", random.Random(11))
        second = transformation.encode("vault passphrase", random.Random(11))
        assert first == second
        assert not transformation.deterministic


def test_encode_varies_across_seeds(bank: RiddleBank) -> None:
    transformation = RiddleTransformation(bank)

    payloads = {transformation.encode("system canary", random.Random(s)).payload for s in range(20)}

    assert len(payloads) > 1


def test_decode_is_not_implemented(bank: RiddleBank) -> None:
    for transformation in (RiddleTransformation(bank), IndirectTransformation(bank)):
        encoded = transformation.encode("product codename", random.Random(0))
        with pytest.raises(NotImplementedError):
            transformation.decode(encoded)


def test_unknown_label_raises(bank: RiddleBank) -> None:
    for transformation in (RiddleTransformation(bank), IndirectTransformation(bank)):
        with pytest.raises(ConfigError, match="no riddle bank entries"):
            transformation.encode("nuclear launch codes", random.Random(0))


def test_load_riddle_bank_rejects_a_missing_label(tmp_path: Path, bank_path: Path) -> None:
    text = bank_path.read_text(encoding="utf-8")
    path = tmp_path / "riddles.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="missing entries for"):
        load_riddle_bank(path, required_labels=[*LABELS, "moon base coordinates"])


def test_load_riddle_bank_rejects_a_short_label(tmp_path: Path) -> None:
    path = tmp_path / "riddles.yaml"
    path.write_text(
        "labels:\n  system canary:\n    riddles: ['a', 'b']\n    indirect: ['c']\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_riddle_bank(path, required_labels=["system canary"])


def test_load_riddle_bank_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_riddle_bank(tmp_path / "absent.yaml")


def test_registry_covers_the_argument_free_families() -> None:
    for family in ARGUMENT_FREE_FAMILIES:
        transformation = get_transformation(family)
        assert transformation.family is family
        assert TRANSFORMATIONS[family] is transformation
    assert set(TRANSFORMATIONS) == set(ARGUMENT_FREE_FAMILIES)


def test_build_riddle_transformations_covers_the_remaining_families(bank: RiddleBank) -> None:
    built = build_riddle_transformations(bank)

    assert set(built) == {Family.RIDDLE, Family.INDIRECT}
    for family, transformation in built.items():
        assert transformation.family is family
        assert not transformation.deterministic


def test_every_family_except_none_is_registered_somewhere(bank: RiddleBank) -> None:
    covered = set(TRANSFORMATIONS) | set(build_riddle_transformations(bank))

    assert covered == set(Family) - {Family.NONE}


def test_riddle_payload_is_usable_as_an_encoded_model(bank: RiddleBank) -> None:
    encoded = RiddleTransformation(bank).encode("internal API token", random.Random(2))

    assert isinstance(encoded, Encoded)
    assert encoded.params["label"] == "internal API token"
