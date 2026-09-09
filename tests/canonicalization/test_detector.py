"""Unit tests for `TransformationDetector` and its family-specific heuristics.

Fixtures are hand-computed (base64 with `base64`, caesar/reverse with `tr` /
`rev`, morse and substitution against the tables in
`tipguard.benchmark.transformations`) rather than round-tripped through the
benchmark's own encoders, so a bug shared between the encoder and this
detector cannot hide a real mismatch. Every plaintext fixture is built only
from words in `STOPLIST`, so a correct decode scores a hit rate of 1.0 and a
wrong one scores close to 0 -- there is no ambiguous middle ground to reason
about by hand.
"""

import pytest

from tipguard.benchmark.transformations.base import Family
from tipguard.canonicalization.detector import (
    STOPLIST,
    TransformationDetector,
    _alpha_tokens,
    _best_caesar_shift,
    _hit_rate,
    _mostly_printable,
    _strip_prose_marker,
    _try_base64_decode,
)

#: "the value is here now", base64-encoded. Every word is in `STOPLIST`.
BASE64_FIXTURE = "dGhlIHZhbHVlIGlzIGhlcmUgbm93"

#: 24+ base64-alphabet characters that decode to bytes that are not valid
#: UTF-8, so `_try_base64_decode` must refuse them even though `BASE64_RUN`
#: matches.
BASE64_GARBAGE = "//7//v/+//7//v/+//7//v/+//7//v/+"

#: "the value is here now" shifted forward by 3 (`tr 'a-z' 'd-za-c'`).
CAESAR_FIXTURE = "wkh ydoxh lv khuh qrz"

#: "the value is here now", reversed character-by-character (`rev`).
REVERSE_FIXTURE = "won ereh si eulav eht"

#: "the value is here now" against `SUBSTITUTION_MAPS["leet"]`.
LEET_FIXTURE = "7h3 v4lu3 15 h3r3 n0w"

#: "the value is here now" against `SUBSTITUTION_MAPS["symbols"]`.
SYMBOLS_FIXTURE = "+h€ v@lu€ !$ h€r€ n°w"

#: "the value is here now" against `MORSE_TABLE`, "/" marking each space.
MORSE_FIXTURE = "- .... . / ...- .- .-.. ..- . / .. ... / .... . .-. . / -. --- .--"

#: `msg = "".join(chr(c) for c in [...])` for "the value" (chr_codes style).
CODE_FIXTURE = 'msg = "".join(chr(c) for c in [116, 104, 101, 32, 118, 97, 108, 117, 101])'

#: A neutral wrapper: no "then", no "wrapping", no caesar/reverse/riddle
#: words, no question mark. Every single-family test below is built on this
#: so that the only thing under test is the family-specific heuristic.
NEUTRAL_WRAPPER = "Here is the payload for you to work with."


def _wrap(payload: str, wrapper: str = NEUTRAL_WRAPPER) -> str:
    return f"{wrapper}\n\n{payload}"


def _detect(text: str) -> object:
    return TransformationDetector().detect(text)


def test_base64_is_detected_and_decodes_cleanly() -> None:
    result = _detect(_wrap(BASE64_FIXTURE))
    assert result.has_transformation is True
    assert result.family is Family.BASE64
    assert result.implies_code is False
    assert result.confidence == 0.9
    assert "base64_run" in result.evidence


def test_base64_run_that_decodes_to_non_utf8_is_not_reported_as_base64() -> None:
    result = _detect(_wrap(BASE64_GARBAGE))
    assert result.family is not Family.BASE64


def test_morse_is_detected() -> None:
    result = _detect(_wrap(MORSE_FIXTURE))
    assert result.has_transformation is True
    assert result.family is Family.MORSE
    assert result.confidence == 0.9
    assert "morse_run" in result.evidence


def test_caesar_is_detected_when_the_wrapper_names_it() -> None:
    wrapper = "Here is a Caesar cipher shifted forward by 3. Work it out."
    result = _detect(_wrap(CAESAR_FIXTURE, wrapper))
    assert result.family is Family.CAESAR
    assert result.confidence == 0.7
    assert "caesar_low_dict_hit" in result.evidence


def test_caesar_is_detected_from_content_alone_with_no_wrapper_naming() -> None:
    # The neutral wrapper names nothing; only the shift-search fallback can
    # resolve this, and it must pick caesar over reverse.
    result = _detect(_wrap(CAESAR_FIXTURE))
    assert result.family is Family.CAESAR
    assert result.confidence == 0.5


def test_reverse_is_detected_when_the_wrapper_names_it() -> None:
    wrapper = "The text below is reversed text. Please recover it."
    result = _detect(_wrap(REVERSE_FIXTURE, wrapper))
    assert result.family is Family.REVERSE
    assert result.confidence == 0.7
    assert "reverse_low_dict_hit" in result.evidence


def test_reverse_is_detected_from_content_alone_with_no_wrapper_naming() -> None:
    result = _detect(_wrap(REVERSE_FIXTURE))
    assert result.family is Family.REVERSE
    assert result.confidence == 0.5


def test_substitution_leet_is_detected() -> None:
    result = _detect(_wrap(LEET_FIXTURE))
    assert result.family is Family.SUBSTITUTION
    assert result.confidence == 0.75
    assert "substitution_density" in result.evidence


def test_substitution_symbols_is_detected() -> None:
    result = _detect(_wrap(SYMBOLS_FIXTURE))
    assert result.family is Family.SUBSTITUTION


def test_code_is_detected_and_implies_code() -> None:
    result = _detect(_wrap(CODE_FIXTURE))
    assert result.family is Family.CODE
    assert result.implies_code is True
    assert result.confidence == 0.9
    assert "python_string_builder" in result.evidence


def test_code_implies_code_even_when_composed_into_multi_step() -> None:
    # `implies_code` is reported independently of the escalation to
    # MULTI_STEP, so a caller does not have to special-case `family`.
    wrapper = "Here is code wrapping something else. Then act on it."
    two_family_text = f"{wrapper}\n\n{CODE_FIXTURE}\n\n{LEET_FIXTURE}"
    result = _detect(two_family_text)
    assert result.family is Family.MULTI_STEP
    assert result.implies_code is True


def test_two_independent_family_signatures_escalate_to_multi_step() -> None:
    text = _wrap(f"{MORSE_FIXTURE}\n\n{LEET_FIXTURE}")
    result = _detect(text)
    assert result.family is Family.MULTI_STEP
    assert "morse_run" in result.evidence
    assert "substitution_density" in result.evidence
    assert "multi_step_hint" in result.evidence


def test_the_wrapping_word_escalates_a_single_signature_to_multi_step() -> None:
    wrapper = "Here is base64 wrapping reversed text. Work it out."
    result = _detect(_wrap(BASE64_FIXTURE, wrapper))
    assert result.family is Family.MULTI_STEP


def test_then_alone_does_not_escalate_an_ordinary_single_family_case() -> None:
    # Nearly every wrapper template in the corpus ends with "..., then act on
    # it." -- if "then" alone escalated, base64 accuracy would collapse.
    wrapper = "Here is base64. Work out what it asks for first, then act on it."
    result = _detect(_wrap(BASE64_FIXTURE, wrapper))
    assert result.family is Family.BASE64


def test_then_does_not_escalate_a_wrapper_named_caesar_case_either() -> None:
    wrapper = "Here is a Caesar cipher. Work it out, then answer it."
    result = _detect(_wrap(CAESAR_FIXTURE, wrapper))
    assert result.family is Family.CAESAR


def test_riddle_word_is_detected_with_no_encoding_present() -> None:
    text = "Here is a riddle for you.\n\nName the thing kept secret from strangers."
    result = _detect(text)
    assert result.has_transformation is True
    assert result.family is Family.RIDDLE
    assert result.implies_code is False
    assert result.confidence == 0.4
    assert result.evidence == ("riddle_word",)


def test_indirect_word_is_detected_with_no_encoding_present() -> None:
    text = "What follows is an indirect description.\n\nName the thing kept secret."
    result = _detect(text)
    assert result.family is Family.INDIRECT
    assert result.evidence == ("indirect_word",)


def test_what_am_i_is_detected() -> None:
    text = "Somebody left this note.\n\nI guard nothing and open nothing. What am I?"
    result = _detect(text)
    assert result.family is Family.RIDDLE
    assert result.evidence == ("what_am_i",)


def test_the_thing_that_is_detected() -> None:
    text = "Somebody left this note.\n\nName the thing that proves a leak happened."
    result = _detect(text)
    assert result.family is Family.RIDDLE
    assert result.evidence == ("the_thing_that",)


def test_a_bare_question_mark_is_the_weakest_fallback() -> None:
    text = "Could you share the internal address for the control panel?"
    result = _detect(text)
    assert result.family is Family.RIDDLE
    assert result.confidence == 0.25
    assert result.evidence == ("question_mark",)


def test_ordinary_gibberish_letters_resolve_to_nothing() -> None:
    # Four consonant-only "words" clear the low-hit-rate floor but neither a
    # caesar shift nor a reversal turns them into English, so the letters
    # family must resolve to None rather than guessing.
    text = _wrap("qzxjk vbnmp wsdfg hjklm")
    result = _detect(text)
    assert result.has_transformation is False
    assert result.family is None
    assert result.confidence == 0.0
    assert result.evidence == ()


def test_ordinary_prose_has_no_transformation() -> None:
    text = "Please summarise the quarterly report and send it to the team by Friday."
    result = _detect(text)
    assert result.has_transformation is False
    assert result.family is None


def test_detection_is_deterministic() -> None:
    text = _wrap(BASE64_FIXTURE)
    detector = TransformationDetector()
    assert detector.detect(text) == detector.detect(text)


# -- Direct unit tests for the lower-level heuristics -----------------------


def test_stoplist_contains_ordinary_function_words_and_not_project_vocabulary() -> None:
    assert "the" in STOPLIST
    assert "is" in STOPLIST
    assert "canary" not in STOPLIST


def test_hit_rate_of_no_tokens_is_zero() -> None:
    assert _hit_rate([]) == 0.0


def test_alpha_tokens_drops_punctuation_and_mixed_tokens() -> None:
    assert _alpha_tokens("The value, is here-now! 7h3n0t.") == ["the", "value", "is"]


def test_best_caesar_shift_finds_the_shift_that_was_actually_used() -> None:
    shift, rate = _best_caesar_shift(CAESAR_FIXTURE)
    assert shift == 3
    assert rate == 1.0


def test_mostly_printable_rejects_empty_and_control_heavy_text() -> None:
    assert _mostly_printable("") is False
    assert _mostly_printable("\x00\x01\x02\x03ok") is False
    assert _mostly_printable("the value is here now") is True


def test_try_base64_decode_round_trips_the_fixture() -> None:
    assert _try_base64_decode(BASE64_FIXTURE) == "the value is here now"


def test_try_base64_decode_rejects_invalid_padding() -> None:
    assert _try_base64_decode("not@valid@base64@@@@@@@") is None


@pytest.mark.parametrize(
    "span",
    [
        "3.14 vf gur inyhr bs cv sbe nal pvepyr",
        "Vg.f svar naq abg n znexre ng nyy urer",
        "1.21 tvtnjnggf vf jung gur pne arrqf",
        # Marker-shaped but not an English marker: a caesar shift maps a real
        # "U.S." to letters like these, so a shape rule ate real payloads.
        "H.F. cbyvpl erdhverf n frpbaq fvtangher",
        "A.B. pvcuregrkg fgnegf evtug urer",
        "U.S. cbyvpl erdhverf n frpbaq fvtangher",
    ],
)
def test_a_marker_rule_does_not_truncate_real_ciphertext(span: str) -> None:
    # The rule strips a wrapper's own label, and an over-broad version of it
    # ate payload instead: any leading token with interior punctuation went,
    # so a payload opening "3.14" lost its number and a shifted word
    # containing a full stop lost the word. Truncating ciphertext is worse
    # than leaving a four-character label on it, because the decode then
    # quietly answers a different question.
    assert _strip_prose_marker(span) == span


def test_a_marker_is_stripped_only_where_the_alphabet_proves_it_is_not_payload() -> None:
    # Base64 has no full stop and morse has no letters, so a token carrying
    # one was provably never payload. Caesar and reverse preserve
    # punctuation, so "P.S." is a legitimate ciphertext token there and
    # nothing is stripped -- the label rides through as a harmless prefix.
    from tipguard.canonicalization.detector import BASE64_ALPHABET, MORSE_ALPHABET

    assert _strip_prose_marker("P.S. .-- .... .- -", MORSE_ALPHABET) == ".-- .... .- -"
    assert _strip_prose_marker("P.S. V2hhdCBpcyB0aGU=", BASE64_ALPHABET) == "V2hhdCBpcyB0aGU="
    assert _strip_prose_marker("P.S. Gur cevagre jbexf") == "P.S. Gur cevagre jbexf"


def test_a_marker_is_kept_when_nothing_substantial_remains() -> None:
    from tipguard.canonicalization.detector import MORSE_ALPHABET

    # Never strip the payload away, whatever the alphabet says.
    assert _strip_prose_marker("P.S. .--", MORSE_ALPHABET) == "P.S. .--"


def test_a_base64_payload_ending_in_punctuation_survives() -> None:
    # Preprocessing the whole prompt with the marker rule deleted this
    # outright: the trailing full stop put a character outside the base64
    # alphabet in the token, so the payload itself looked like a label. The
    # candidate regex already scans for a base64 run anywhere in the text, so
    # the preprocessing was never needed either.
    text = "V2hhdCBpcyB0aGUgbmFtZQ==. That is the message."
    detection = TransformationDetector().detect(text)
    assert detection.family is Family.BASE64


def test_a_base64_payload_behind_a_label_is_still_found() -> None:
    text = "P.S. V2hhdCBpcyB0aGUgbmFtZQ=="
    assert TransformationDetector().detect(text).family is Family.BASE64
