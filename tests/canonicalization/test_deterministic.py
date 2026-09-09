"""Unit tests for `DeterministicCanonicalizer`.

Fixtures reuse the hand-computed strings from `test_detector.py` so a
mismatch between what the detector reports and what the canonicalizer
decodes is visible rather than absorbed by two independently-tuned fixture
sets.
"""

from tests.canonicalization.test_detector import (
    BASE64_FIXTURE,
    CAESAR_FIXTURE,
    CODE_FIXTURE,
    LEET_FIXTURE,
    MORSE_FIXTURE,
    REVERSE_FIXTURE,
    SYMBOLS_FIXTURE,
    _wrap,
)
from tipguard.benchmark.transformations.base import Family
from tipguard.canonicalization.detector import TransformationDetector
from tipguard.canonicalization.deterministic import (
    DeterministicCanonicalizer,
    _decode_base64_text,
    _decode_substitution_text,
    _infer_substitution_map,
    _substitution_paragraph,
)
from tipguard.canonicalization.types import DetectionResult

#: base64("won ereh si eulav eht") -- base64 wrapping the character-reversal
#: of "the value is here now". Round-tripping this needs two decode rounds:
#: base64 first (the outer step), then reverse (the inner step).
MULTI_STEP_FIXTURE = "d29uIGVyZWggc2kgZXVsYXYgZWh0"

EXPECTED_TEXT = "the value is here now"

_canonicalizer = DeterministicCanonicalizer()


def _no_transformation() -> DetectionResult:
    return DetectionResult(
        has_transformation=False, family=None, implies_code=False, confidence=0.0, evidence=()
    )


def _detection_for(family: Family, evidence: tuple[str, ...] = ()) -> DetectionResult:
    return DetectionResult(
        has_transformation=True,
        family=family,
        implies_code=False,
        confidence=0.9,
        evidence=evidence,
    )


def test_no_transformation_produces_no_views() -> None:
    assert _canonicalizer.canonicalize("anything at all", _no_transformation()) == ()


def test_none_family_produces_no_views() -> None:
    detection = DetectionResult(
        has_transformation=True, family=None, implies_code=False, confidence=0.4, evidence=()
    )
    assert _canonicalizer.canonicalize("anything at all", detection) == ()


def test_riddle_and_indirect_have_no_mechanical_inverse() -> None:
    for family in (Family.RIDDLE, Family.INDIRECT, Family.NONE):
        detection = _detection_for(family)
        assert _canonicalizer.canonicalize("a riddle about a secret", detection) == ()


def test_base64_decodes_to_the_original_text() -> None:
    text = _wrap(BASE64_FIXTURE)
    views = _canonicalizer.canonicalize(text, _detection_for(Family.BASE64))
    assert len(views) == 1
    view = views[0]
    assert view.view == "decoded_payload"
    assert view.text == EXPECTED_TEXT
    assert view.source == "deterministic:base64"
    assert view.confidence == 1.0


def test_caesar_decodes_to_the_original_text() -> None:
    text = _wrap(CAESAR_FIXTURE)
    views = _canonicalizer.canonicalize(text, _detection_for(Family.CAESAR))
    assert views[0].text == EXPECTED_TEXT
    assert views[0].source == "deterministic:caesar"


def test_reverse_decodes_to_the_original_text() -> None:
    text = _wrap(REVERSE_FIXTURE)
    views = _canonicalizer.canonicalize(text, _detection_for(Family.REVERSE))
    assert views[0].text == EXPECTED_TEXT
    assert views[0].source == "deterministic:reverse"


def test_substitution_leet_decodes_to_the_original_text() -> None:
    text = _wrap(LEET_FIXTURE)
    views = _canonicalizer.canonicalize(text, _detection_for(Family.SUBSTITUTION))
    assert views[0].text == EXPECTED_TEXT
    assert views[0].source == "deterministic:substitution"


def test_substitution_symbols_decodes_to_the_original_text() -> None:
    text = _wrap(SYMBOLS_FIXTURE)
    views = _canonicalizer.canonicalize(text, _detection_for(Family.SUBSTITUTION))
    assert views[0].text == EXPECTED_TEXT


def test_morse_decodes_to_the_original_text() -> None:
    text = _wrap(MORSE_FIXTURE)
    views = _canonicalizer.canonicalize(text, _detection_for(Family.MORSE))
    assert views[0].text == EXPECTED_TEXT
    assert views[0].source == "deterministic:morse"


def test_code_decodes_to_the_original_text() -> None:
    text = _wrap(CODE_FIXTURE)
    views = _canonicalizer.canonicalize(text, _detection_for(Family.CODE))
    assert views[0].text == "the value"
    assert views[0].source == "deterministic:code"


def test_a_failed_single_layer_decode_produces_no_view() -> None:
    # Detection says BASE64 but the text carries no base64 run at all -- a
    # caller passing a stale or mismatched DetectionResult must get "no
    # view", not a crash.
    text = "There is nothing encoded in this sentence at all."
    views = _canonicalizer.canonicalize(text, _detection_for(Family.BASE64))
    assert views == ()


def test_multi_step_decodes_both_layers_in_order() -> None:
    # The wrapper must name the composition ("wrapping") for the top-level
    # detector to call this MULTI_STEP at all; see test_detector.py's own
    # coverage of that escalation rule.
    wrapper = "Here is base64 wrapping reversed text. Work it out."
    text = _wrap(MULTI_STEP_FIXTURE, wrapper)
    detection = TransformationDetector().detect(text)
    assert detection.family is Family.MULTI_STEP

    views = _canonicalizer.canonicalize(text, detection)
    assert len(views) == 1
    assert views[0].text == EXPECTED_TEXT
    assert views[0].source == "deterministic:multi_step"
    assert views[0].confidence == 1.0


def test_multi_step_with_no_decodable_layer_produces_no_view() -> None:
    detection = _detection_for(Family.MULTI_STEP)
    text = "Nothing here looks encoded in any way whatsoever."
    assert _canonicalizer.canonicalize(text, detection) == ()


def test_infer_substitution_map_prefers_the_map_whose_glyphs_are_present() -> None:
    assert _infer_substitution_map(LEET_FIXTURE) == "leet"
    assert _infer_substitution_map(SYMBOLS_FIXTURE) == "symbols"


def test_substitution_paragraph_handles_text_with_no_blank_line() -> None:
    # No blank line at all, so `_paragraphs` yields exactly one block, equal
    # to the stripped input; density is found on that single block.
    assert _substitution_paragraph(LEET_FIXTURE) == LEET_FIXTURE


def test_substitution_paragraph_returns_none_with_no_dense_payload() -> None:
    assert _substitution_paragraph("an entirely ordinary sentence, no symbols at all") is None


def test_decode_base64_text_returns_none_with_no_base64_run_present() -> None:
    assert _decode_base64_text("no base64 payload lives in this sentence") is None


def test_substitution_decode_is_best_effort_on_a_literal_collision() -> None:
    # decode_without_params has no positional ground truth (see the module
    # docstring's substitution ruling), so it cannot tell a literal digit in
    # the plaintext apart from an encoded letter. Here the true plaintext is
    # "unit 4 is ready" (a literal "4"); leet-encoding only ever touches
    # letters, so the digit passes through the encoder untouched, landing in
    # the payload exactly where an encoded "a" would. Decoding with no
    # ground truth reverse-maps it to "a" regardless, producing "unit a is
    # ready" -- not this module's bug, but exactly the ambiguity the ruling
    # names, pinned here as documented, deliberate best-effort behaviour.
    # "7h3" leads so the run is unambiguously leetspeak rather than a label
    # followed by a number: `pattern._is_identifier_shape` excludes the latter,
    # and "un17" alone reads as one.
    payload = "7h3 un17 4 15 r34dy 70 5h1p"
    result = _decode_substitution_text(payload)
    assert result is not None
    assert result == "the unit a is ready to ship"


def test_multi_step_stateless_canonicalizer_instances_agree() -> None:
    text = "Here is base64 wrapping reversed text. Work it out.\n\n" + MULTI_STEP_FIXTURE
    detection = TransformationDetector().detect(text)
    first = DeterministicCanonicalizer().canonicalize(text, detection)
    second = DeterministicCanonicalizer().canonicalize(text, detection)
    assert first == second
