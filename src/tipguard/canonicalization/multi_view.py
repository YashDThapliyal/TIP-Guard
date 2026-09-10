"""Assembles one prompt's `Canonicalization` from every enabled canonicalizer.

`DeterministicCanonicalizer` and `LLMCanonicalizer` each return only their
own `CanonicalView` tuples (or, for the LLM side, a `CanonError`) -- neither
knows about the other, and neither builds the assembled
`tipguard.canonicalization.types.Canonicalization` a guard actually reads.
This module is where the pipeline runs `TransformationDetector` ->
`DeterministicCanonicalizer` -> `LLMCanonicalizer` in order and merges what
came back, exactly the ordering the deterministic side's own module
docstring anticipates: "a caller that wants a different category for its own
reporting ... always has more context than this module does to assign it".
"""

from tipguard.canonicalization.code_analysis import RestrictedCodeAnalyzer
from tipguard.canonicalization.detector import TransformationDetector
from tipguard.canonicalization.deterministic import DeterministicCanonicalizer
from tipguard.canonicalization.llm_canonicalizer import CanonError, CanonJudgement, LLMCanonicalizer
from tipguard.canonicalization.types import Canonicalization, CanonicalView, DetectionResult

#: How much a view's own `confidence` counts toward the assembled
#: `Canonicalization.confidence`, by which canonicalizer produced it (the
#: prefix of `CanonicalView.source` before its first `:`). A view from a
#: source this table does not name contributes nothing -- there is none
#: today, but a future canonicalizer that forgot to register itself here
#: should be silently underweighted, not crash a run.
_SOURCE_WEIGHT: dict[str, float] = {"llm": 1.0, "deterministic": 0.8}

#: `Canonicalization.uncertainties` markers naming an axis this run produced
#: no view for at all, as distinct from a view that was produced but
#: expressed its own doubt (which is instead carried in the view's own
#: `confidence`, and for the LLM side in `CanonJudgement.uncertainties`).
NO_LLM_VIEW = "no_llm_view"
NO_DECODED_VIEW = "no_decoded_view"


#: The category a model names when it judges a prompt free of policy risk.
#: Dropped from the merged result rather than passed through, because a
#: caller testing `if canon.policy_categories:` would read ("none",) as "some
#: category applies" -- the opposite of what the model said, and the same
#: shape the no-LLM path already reports as empty.
NO_CATEGORY = "none"


def _policy_categories(judgement: "CanonJudgement | None") -> tuple[str, ...]:
    """The policy categories the model named, with its no-risk marker removed.

    Emptiness therefore means "no category applies" from either path, so a
    caller cannot read the LLM's considered "none" and the absence of an LLM
    as opposite things.
    """
    if judgement is None:
        return ()
    return tuple(c for c in judgement.policy_categories if c != NO_CATEGORY)


def _weight_for(source: str) -> float:
    prefix = source.split(":", 1)[0]
    return _SOURCE_WEIGHT.get(prefix, 0.0)


def _reconstructed_intent(
    text: str, llm_views: tuple[CanonicalView, ...], decoded_views: tuple[CanonicalView, ...]
) -> str:
    """The LLM's reconstructed task if there is one, else the deterministic
    decode, else the prompt's own literal text -- in that order of trust,
    since each later fallback is a weaker reading of what the prompt asks."""
    for view in llm_views:
        if view.view == "reconstructed_task":
            return view.text
    for view in decoded_views:
        if view.view == "decoded_payload":
            return view.text
    return text


class MultiViewCanonicalizer:
    """Runs the detector, the deterministic side, and the LLM side in order,
    and merges their `CanonicalView`s into one `Canonicalization`.

    `enable` names which of `"deterministic"` and `"llm"` actually run, so
    the same pipeline can be measured with either arm switched off -- the
    detector itself always runs, since `Canonicalization.detection` is a
    required field no `enable` setting can omit. `code_analyzer` is accepted
    for parity with `deterministic`, which already routes `Family.CODE`
    (including as one layer of a multi-step decode) through its own
    `RestrictedCodeAnalyzer` instance internally; `run` does not call it
    directly today, but a caller that wants this pipeline's own analyzer
    instance -- to reuse it, or to substitute a spy in a test -- has it on
    hand as `self.code_analyzer`.
    """

    def __init__(
        self,
        detector: TransformationDetector,
        deterministic: DeterministicCanonicalizer,
        code_analyzer: RestrictedCodeAnalyzer,
        llm: LLMCanonicalizer | None,
        enable: set[str],
    ) -> None:
        self.detector = detector
        self.deterministic = deterministic
        self.code_analyzer = code_analyzer
        self.llm = llm
        self.enable = enable

    def run(self, text: str) -> Canonicalization:
        detection = self.detector.detect(text)
        decoded_views = self._decoded_views(text, detection)
        llm_views, llm_errors, llm_judgement = self._llm_views(text, decoded_views)

        views = decoded_views + llm_views
        uncertainties = self._uncertainties(llm_views, decoded_views, llm_judgement)
        confidence = max(
            (view.confidence * _weight_for(view.source) for view in views), default=0.0
        )
        policy_categories = _policy_categories(llm_judgement)
        contains_transformation = detection.has_transformation or (
            llm_judgement.contains_transformation if llm_judgement is not None else False
        )

        return Canonicalization(
            detection=detection,
            views=views,
            contains_transformation=contains_transformation,
            reconstructed_intent=_reconstructed_intent(text, llm_views, decoded_views),
            policy_categories=policy_categories,
            confidence=confidence,
            uncertainties=uncertainties,
            errors=llm_errors,
        )

    def _decoded_views(self, text: str, detection: DetectionResult) -> tuple[CanonicalView, ...]:
        if "deterministic" not in self.enable or not detection.has_transformation:
            return ()
        return self.deterministic.canonicalize(text, detection)

    def _llm_views(
        self, text: str, decoded_views: tuple[CanonicalView, ...]
    ) -> tuple[tuple[CanonicalView, ...], tuple[str, ...], CanonJudgement | None]:
        if "llm" not in self.enable or self.llm is None:
            return (), (), None
        result = self.llm.canonicalize(text, decoded_views)
        if isinstance(result, CanonError):
            return result.views, result.errors, self.llm.last_judgement
        return result, (), self.llm.last_judgement

    @staticmethod
    def _uncertainties(
        llm_views: tuple[CanonicalView, ...],
        decoded_views: tuple[CanonicalView, ...],
        llm_judgement: CanonJudgement | None,
    ) -> tuple[str, ...]:
        markers: list[str] = list(llm_judgement.uncertainties) if llm_judgement else []
        if not llm_views:
            markers.append(NO_LLM_VIEW)
        if not decoded_views:
            markers.append(NO_DECODED_VIEW)
        return tuple(markers)
