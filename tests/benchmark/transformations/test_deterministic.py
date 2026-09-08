"""Round-trip and determinism tests for the deterministic transformation encoders."""

import random

import pytest

from tipguard.benchmark.transformations import TRANSFORMATIONS, get_transformation
from tipguard.benchmark.transformations.base import Family

# Mixed case, digits, a space and punctuation. The digits (2, 6) and the
# punctuation (?) are chosen to avoid colliding with the substitution
# transformation's own output alphabet (leet digits 0/1/3/4/5/7 and symbol
# glyphs @/€/!/°/$/+), which would otherwise make that round trip ambiguous.
GENERAL_TEXT = "Hello World 26?"
# Morse drops anything outside a-z, 0-9 and space, so its round-trip input
# must be alphanumeric plus spaces only (no punctuation).
MORSE_TEXT = "Hello World 26"

ROUND_TRIP_INPUT: dict[Family, str] = {
    Family.BASE64: GENERAL_TEXT,
    Family.CAESAR: GENERAL_TEXT,
    Family.REVERSE: GENERAL_TEXT,
    Family.MORSE: MORSE_TEXT,
    Family.SUBSTITUTION: GENERAL_TEXT,
}

EXACT_ROUND_TRIP_FAMILIES = {Family.BASE64, Family.CAESAR, Family.REVERSE}
LOWERCASED_ROUND_TRIP_FAMILIES = {Family.MORSE, Family.SUBSTITUTION}


@pytest.mark.parametrize("family", list(ROUND_TRIP_INPUT))
def test_round_trip(family: Family) -> None:
    text = ROUND_TRIP_INPUT[family]
    transformation = get_transformation(family)
    rng = random.Random(0)

    encoded = transformation.encode(text, rng)
    decoded = transformation.decode(encoded)

    if family in EXACT_ROUND_TRIP_FAMILIES:
        assert decoded == text
    else:
        assert family in LOWERCASED_ROUND_TRIP_FAMILIES
        assert decoded == text.lower()


@pytest.mark.parametrize("family", list(ROUND_TRIP_INPUT))
def test_determinism(family: Family) -> None:
    text = ROUND_TRIP_INPUT[family]
    transformation = get_transformation(family)

    encoded_a = transformation.encode(text, random.Random(0))
    encoded_b = transformation.encode(text, random.Random(0))

    assert encoded_a == encoded_b


@pytest.mark.parametrize("family", list(ROUND_TRIP_INPUT))
def test_hint_and_family(family: Family) -> None:
    text = ROUND_TRIP_INPUT[family]
    transformation = get_transformation(family)

    encoded = transformation.encode(text, random.Random(0))

    assert encoded.hint
    assert encoded.family is transformation.family
    assert encoded.family is family


def test_caesar_shift_recorded_and_case_restored() -> None:
    transformation = get_transformation(Family.CAESAR)
    text = "Hello World"

    encoded = transformation.encode(text, random.Random(0))

    assert int(encoded.params["shift"]) in {1, 2, 3, 5, 7, 11, 13}
    assert transformation.decode(encoded) == text


def test_morse_sos_1() -> None:
    transformation = get_transformation(Family.MORSE)

    encoded = transformation.encode("SOS 1", random.Random(0))

    assert encoded.payload == "... --- ... / .----"


def test_substitution_map_recorded_and_leet_has_no_vowel_letters() -> None:
    transformation = get_transformation(Family.SUBSTITUTION)

    # Search deterministically for a seed that selects the "leet" map so we
    # can make a strong assertion on its output.
    encoded = None
    for seed in range(20):
        candidate = transformation.encode("case sensitive", random.Random(seed))
        if candidate.params["map"] == "leet":
            encoded = candidate
            break
    assert encoded is not None, "expected at least one seed to select the leet map"

    assert encoded.params["map"] in {"leet", "symbols"}
    assert "a" not in encoded.payload
    assert "e" not in encoded.payload


def test_get_transformation_returns_registered_families() -> None:
    for family in (
        Family.BASE64,
        Family.CAESAR,
        Family.REVERSE,
        Family.MORSE,
        Family.SUBSTITUTION,
    ):
        transformation = get_transformation(family)
        assert transformation.family is family
        assert TRANSFORMATIONS[family] is transformation


def test_get_transformation_raises_key_error_for_unimplemented_family() -> None:
    with pytest.raises(KeyError):
        get_transformation(Family.CODE)
