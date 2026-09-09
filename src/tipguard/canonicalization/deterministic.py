"""Deterministic decoding: turning a detected transformation back into text.

`DeterministicCanonicalizer.canonicalize` takes a prompt and the
`DetectionResult` `TransformationDetector.detect` already produced for it,
and tries to mechanically invert whichever family was detected, by calling
straight into the benchmark's own `decode` for that family
(`tipguard.benchmark.transformations`) rather than re-implementing it. It
never executes anything derived from the prompt: the code family's decode is
`code_snippet.decode_snippet`, which is `ast.parse`-and-inspect only, and
every other family's decode is a fixed arithmetic or table transform. Nothing
here ever calls `exec`, `eval` or `compile`.

**The substitution ruling.** In production this canonicalizer only ever has
the raw prompt text -- never the `Encoded.params` the benchmark generator
recorded, and in particular never whether the original mapping was
`ambiguous` (some source character already looked like a replacement glyph;
see `tipguard.benchmark.transformations.substitution`). So substitution is
always decoded with `decode_without_params`, the best-effort reverse mapping
that has no positional ground truth to consult, and its result is always
reported at whatever confidence the reconstruction earns -- there is no
separate "ambiguous" code path in this module. The ruling this module is
built to satisfy lives one level up, in the evaluation that compares a
decode against a dataset case's known ground truth: when `case.metadata`
records `params["ambiguous"] == "true"` for a substitution case, a decode
that does not match `canonical_intent` is that case being genuinely
unrecoverable, not this canonicalizer being wrong, and must be scored as
`ErrorCategory.AMBIGUOUS` rather than `ErrorCategory.INCORRECT`. See
`tests/canonicalization/test_on_corpus.py::_is_ambiguous_substitution`.

**What a caller gets back.** `canonicalize` returns a `tuple[CanonicalView,
...]` -- never a full `Canonicalization` -- because this is one canonicalizer
among several a later phase's pipeline will run and merge; assembling the
`errors`/`uncertainties`/`reconstructed_intent` envelope is that later
phase's job, not this one's. On success it is always exactly one
`"decoded_payload"` view; on any failure to invert -- detection said there
was nothing to decode, the extracted payload did not survive its family's
own decoder, morse hit a code point outside the table, a substitution
literal collision could not be resolved further than best-effort -- it is
the empty tuple. An empty tuple is this module's whole error channel: there
is no side output naming *why* a case failed, because the "why" is
`ErrorCategory.UNSUPPORTED` by construction (nothing else this module could
attempt succeeded) and a caller that wants a different category for its own
reporting -- `MISSED`, `INCORRECT`, `AMBIGUOUS` -- always has more context
than this module does to assign it.

**Never logged.** A `decoded_payload` view's `text` can be a protected
value in the clear. This module never writes it to a log, an exception
message, or anywhere but the view it returns; every exception caught below
is caught and discarded without including the payload that raised it.
"""

from collections.abc import Callable

from tipguard.benchmark.transformations.base import Encoded, Family
from tipguard.benchmark.transformations.base64_t import Base64Transformation
from tipguard.benchmark.transformations.caesar import CaesarTransformation
from tipguard.benchmark.transformations.code_snippet import decode_snippet
from tipguard.benchmark.transformations.morse import MorseTransformation
from tipguard.benchmark.transformations.reverse import ReverseTransformation
from tipguard.benchmark.transformations.substitution import (
    SUBSTITUTION_MAPS,
    decode_without_params,
)
from tipguard.canonicalization.detector import (
    CONFIDENT_HIT_THRESHOLD,
    LOW_HIT_THRESHOLD,
    _alpha_tokens,
    _best_caesar_shift,
    _extract_base64,
    _extract_code,
    _extract_morse,
    _hit_rate,
    _paragraphs,
    first_layer_family,
    letters_payload,
)
from tipguard.canonicalization.types import CanonicalView, DetectionResult
from tipguard.classifiers.pattern import _has_substitution_density as has_substitution_density

#: A multi-step decode never composes more than the benchmark itself ever
#: composes (`MultiStepTransformation` chains exactly two), but the third
#: round is kept as headroom against a detector guess that consumed a round
#: without actually removing a layer.
MAX_MULTI_STEP_ROUNDS = 3

#: The confidence — the decoded text's own stoplist hit rate — at or above
#: which a multi-step round is judged to have reached plain English, so
#: further rounds are skipped rather than degrading a good decode by trying
#: to peel a layer that is not there.
#: Fewest alphabetic tokens a decode must leave before its stoplist hit rate
#: is worth reading. Matches the detector's own floor for the same reason.
MIN_CONFIDENCE_TOKENS = 4

MULTI_STEP_DONE_THRESHOLD = CONFIDENT_HIT_THRESHOLD + 0.05


def _substitution_paragraph(text: str) -> str | None:
    """The paragraph carrying the substitution payload, or None.

    Tried per-paragraph first, so a dense leeted phrase inside an otherwise
    ordinary wrapper is not diluted by the wrapper's own low density; falls
    back to the whole text for a prompt with no blank-line-separated payload
    at all (every unit test in this module, and any prompt shorter than the
    corpus's own two-paragraph shape).
    """
    for paragraph in _paragraphs(text):
        if has_substitution_density(paragraph):
            return paragraph
    if has_substitution_density(text):
        return text
    return None


def _infer_substitution_map(payload: str) -> str:
    """Which of the two substitution maps `payload`'s glyphs belong to.

    Production has no `Encoded.params["map"]` to read, so the map is
    inferred from which map's own replacement glyphs actually appear in the
    payload. Ties (including "neither map's glyphs appear at all", for a
    payload this function is never called on in practice) fall to "leet" by
    the ordering `SUBSTITUTION_MAPS` was written in.
    """
    best_name, best_count = "leet", -1
    for name, char_map in SUBSTITUTION_MAPS.items():
        count = sum(payload.count(glyph) for glyph in char_map.values())
        if count > best_count:
            best_name, best_count = name, count
    return best_name


def _decode_base64_text(text: str) -> str | None:
    payload = _extract_base64(text)
    if payload is None:
        return None
    try:
        encoded = Encoded(payload=payload, family=Family.BASE64, params={}, hint="")
        return Base64Transformation().decode(encoded)
    except (ValueError, UnicodeDecodeError):
        # `_extract_base64` already confirmed this exact run decodes
        # cleanly, so this only ever fires if that guarantee is ever
        # loosened; kept so a decode failure here still returns "no view"
        # rather than propagating.
        return None


def _decode_morse_text(text: str) -> str | None:
    payload = _extract_morse(text)
    if payload is None:
        return None
    try:
        encoded = Encoded(payload=payload, family=Family.MORSE, params={}, hint="")
        return MorseTransformation().decode(encoded)
    except KeyError:
        # A morse-shaped run whose groups do not all correspond to a table
        # entry (e.g. six dashes) -- a false positive of the regex, not a
        # payload with a hole in it.
        return None


def _decode_caesar_text(text: str) -> str | None:
    """Undo the caesar shift that maximises the decoded stoplist hit rate.

    No confidence gate here: `detection.family is CAESAR` already means
    detection resolved this confidently (a wrapper naming it, or the
    content fallback already clearing `LOW_HIT_THRESHOLD` in
    `detector._resolve_letters_family`), so the shift search is trusted to
    have found the right one rather than re-questioned at decode time.
    """
    paragraph = letters_payload(text)
    if paragraph is None:
        return None
    shift, _ = _best_caesar_shift(paragraph)
    encoded = Encoded(
        payload=paragraph, family=Family.CAESAR, params={"shift": str(shift)}, hint=""
    )
    return CaesarTransformation().decode(encoded)


def _decode_reverse_text(text: str) -> str | None:
    paragraph = letters_payload(text)
    if paragraph is None:
        return None
    encoded = Encoded(payload=paragraph, family=Family.REVERSE, params={}, hint="")
    return ReverseTransformation().decode(encoded)


def _decode_substitution_text(text: str) -> str | None:
    payload = _substitution_paragraph(text)
    if payload is None:
        return None
    return decode_without_params(payload, _infer_substitution_map(payload))


def _decode_code_text(text: str) -> str | None:
    payload = _extract_code(text)
    if payload is None:
        return None
    try:
        return decode_snippet(payload)
    except ValueError:
        return None


#: One decoder per family this canonicalizer can invert directly: every
#: `Family` except `MULTI_STEP` (which iterates this table -- see
#: `_canonicalize_multi_step`) and `RIDDLE` / `INDIRECT` / `NONE` (which have
#: no mechanical inverse at all and so are simply absent from it).
_SINGLE_LAYER_DECODERS: dict[Family, Callable[[str], str | None]] = {
    Family.BASE64: _decode_base64_text,
    Family.MORSE: _decode_morse_text,
    Family.CAESAR: _decode_caesar_text,
    Family.REVERSE: _decode_reverse_text,
    Family.SUBSTITUTION: _decode_substitution_text,
    Family.CODE: _decode_code_text,
}


def _decode_single_layer(text: str, family: Family) -> str | None:
    decoder = _SINGLE_LAYER_DECODERS.get(family)
    return decoder(text) if decoder is not None else None


def _decoded_confidence(text: str) -> float:
    """The decode-success heuristic the brief specifies: stoplist hit rate.

    Applied to the decoded text itself, not the ciphertext: a correct decode
    reads as English and scores accordingly; a wrong shift, a wrong
    substitution map, or a snippet that "parsed" into nonsense all still
    score low, so this doubles as a soundness check a caller can threshold
    on without re-deriving one.
    """
    tokens = _alpha_tokens(text)
    # A hit rate over one or two tokens says nothing. Symbol-substituted text
    # leaves almost no purely-alphabetic tokens standing, and the few it does
    # are short fragments -- a stray "i" scored such a payload at 0.50, above
    # the multi-step "done" threshold, so the loop stopped a layer early and
    # returned still-encoded text as though it were the answer. Below the
    # sample floor the honest answer is no confidence at all.
    if len(tokens) < MIN_CONFIDENCE_TOKENS:
        return 0.0
    return _hit_rate(tokens)


#: The letter-only families, tried blind when the scan cannot name a layer.
_BLIND_LAYER_FAMILIES = (Family.REVERSE, Family.CAESAR)


def _best_letters_layer(text: str) -> str | None:
    """One speculative layer for text whose family the scan cannot name.

    A multi-step payload's outer layer usually leaves the inner one still
    encoded, so the round-one text is not English by any measure and the
    content test that decides between caesar and reverse has nothing to
    grade -- it returns None and the whole decode stops before it starts.
    That is why "reversed text wrapping a Caesar cipher" produced no view at
    all: both families are named in the wrapper, so naming cannot break the
    tie either, and 32 of the 141 easy multi-step cases were lost this way.

    Rather than guess an ordering from the wrapper's prose, each candidate is
    peeled and the one whose result looks most like the next layer is kept.

    A zero score is accepted, and ties keep the first candidate. That is
    deliberate, and an earlier version of this docstring claimed the
    opposite: an intermediate layer of a multi-step payload is still encoded,
    so its stoplist hit rate is legitimately zero, and refusing a zero cost a
    sixth of the multi-step cases when it was tried. The consequence is that
    genuinely undecodable text yields a view too -- carrying confidence 0.0,
    which is the signal a caller must read. Task 7's pipeline weighs a view's
    confidence, never its existence.
    """
    # Only speculate on text that does not already read as language. Every
    # text can be reversed, so without this guard ordinary English would be
    # handed back as a recovered payload. The test is on the *input*, not on
    # whether the result improved: an intermediate layer of a multi-step
    # payload is still encoded, so peeling it correctly produces something
    # that is no more English-like than what went in.
    # Judged on the payload span, not the whole text: the wrapper around a
    # payload is always ordinary English, so a whole-prompt reading is high
    # for every case and would block every speculative peel.
    span = letters_payload(text)
    if span is None or _decoded_confidence(span) >= LOW_HIT_THRESHOLD:
        return None
    best: tuple[str, float] | None = None
    for family in _BLIND_LAYER_FAMILIES:
        peeled = _decode_single_layer(text, family)
        if peeled is None or peeled == text:
            continue
        score = _decoded_confidence(peeled)
        if best is None or score > best[1]:
            best = (peeled, score)
    return best[0] if best is not None else None


def _peel_remaining_layers(text: str, rounds: int = MAX_MULTI_STEP_ROUNDS - 1) -> str:
    """Keep peeling `text` while it still looks encoded.

    Shared by the single-family path and the multi-step loop so the two
    cannot disagree about when a decode is finished. Returns the best text it
    reached, which is the input itself when nothing further could be peeled.
    """
    current = text
    for _ in range(rounds):
        if _decoded_confidence(current) >= MULTI_STEP_DONE_THRESHOLD:
            return current
        family = first_layer_family(current)
        peeled = (
            _decode_single_layer(current, family)
            if family is not None
            else _best_letters_layer(current)
        )
        if peeled is None or peeled == current:
            return current
        current = peeled
    return current


class DeterministicCanonicalizer:
    """Recovers the plaintext instruction a detected transformation wrapped.

    See the module docstring for the substitution ruling, the return-value
    contract, and the no-logging guarantee. Stateless: every method reads
    only its arguments, so one instance is safe to reuse or share across
    calls.
    """

    def canonicalize(self, text: str, detection: DetectionResult) -> tuple[CanonicalView, ...]:
        if not detection.has_transformation or detection.family is None:
            return ()
        if detection.family is Family.MULTI_STEP:
            return self._canonicalize_multi_step(text)
        if detection.family in (Family.RIDDLE, Family.INDIRECT, Family.NONE):
            # No mechanical inverse exists for these; a later, semantic
            # canonicalizer is the only thing that can produce a view here.
            return ()
        decoded = _decode_single_layer(text, detection.family)
        if decoded is None:
            return ()
        # Peeling one layer is not the same as finishing. A layered payload
        # whose wrapper does not announce itself shows only its outer
        # signature, so `detect` names a single family in good faith and the
        # inner layer survives -- every difficulty-3 multi-step case failed
        # this way, returning still-encoded text as though it were the
        # answer. Continue from what the first peel produced, on the same
        # terms the multi-step loop uses, and stop as soon as the text reads
        # as language.
        if _decoded_confidence(decoded) < MULTI_STEP_DONE_THRESHOLD:
            decoded = _peel_remaining_layers(decoded)
        return (self._view(decoded, detection.family.value),)

    def _canonicalize_multi_step(self, text: str) -> tuple[CanonicalView, ...]:
        """Peel one layer per round using `first_layer_family`, not `detect`.

        `first_layer_family` is the unescalated scan: on round one it reads
        the *outer* transformation's signature straight off the still-wrapped
        prompt (independent of whatever wrapper wording made the top-level
        `detect` call this multi-step in the first place), decodes it, and
        each further round repeats the same read on whatever text remains.
        Calling `TransformationDetector.detect` here instead would re-apply
        its own multi-step escalation to text that, on round one, is still
        the original prompt -- and escalate straight back to `MULTI_STEP`
        without ever decoding a layer.
        """
        current = text
        rounds_decoded = 0
        for _ in range(MAX_MULTI_STEP_ROUNDS):
            family = first_layer_family(current)
            peeled = (
                _decode_single_layer(current, family)
                if family is not None
                else _best_letters_layer(current)
            )
            if peeled is None or peeled == current:
                break
            current = peeled
            rounds_decoded += 1
            if _decoded_confidence(current) >= MULTI_STEP_DONE_THRESHOLD:
                break
        if rounds_decoded == 0:
            return ()
        return (self._view(current, "multi_step"),)

    @staticmethod
    def _view(text: str, source_suffix: str) -> CanonicalView:
        return CanonicalView(
            view="decoded_payload",
            text=text,
            source=f"deterministic:{source_suffix}",
            confidence=_decoded_confidence(text),
        )
