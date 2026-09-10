"""Decision rules: how a policy gate turns risk scores into a call.

`ThresholdGuard` compares one score against one threshold. TIP-Guard has up
to three scores in play at once -- the original prompt, the reconstructed
intent, and (later) the model's own answer -- and `Decision.CLARIFY` is a
third outcome neither `ThresholdGuard` nor `OutputGuard` ever produces. A
`DecisionRule` is the seam that lets how those scores combine vary
independently of the pipeline that gathers them.
"""

from typing import Protocol

from tipguard.base import FrozenModel
from tipguard.benchmark.schema import Decision
from tipguard.classifiers.types import RiskScore


class RiskInputs(FrozenModel):
    """Every score a `DecisionRule` may read, one field per pipeline stage.

    A stage that did not run -- an ablated classifier, or the output stage
    before the main model has answered -- reports `None` rather than a
    fabricated score, so a rule can tell "not evaluated" apart from
    "evaluated as safe".
    """

    original: RiskScore | None
    canonical: RiskScore | None
    canonical_confidence: float
    output: RiskScore | None = None


class RuleDecision(FrozenModel):
    decision: Decision
    reasons: tuple[str, ...]


class DecisionRule(Protocol):
    def decide(self, scores: RiskInputs) -> RuleDecision: ...


def _available_scores(inputs: RiskInputs) -> list[RiskScore]:
    return [
        score for score in (inputs.original, inputs.canonical, inputs.output) if score is not None
    ]


def _max_risk_decision(inputs: RiskInputs, threshold: float) -> RuleDecision:
    """Block when the single highest available score meets `threshold`.

    Shared by `MaxRiskRule` and, as its fallback once the ambiguous cases are
    carved out, `ConfidenceEscalationRule`.
    """
    available = _available_scores(inputs)
    if not available:
        return RuleDecision(decision=Decision.ALLOW, reasons=())
    top = max(available, key=lambda score: score.score)
    if top.score >= threshold:
        return RuleDecision(
            decision=Decision.BLOCK,
            reasons=(f"{top.classifier}:risk_above_threshold:{top.score}",),
        )
    return RuleDecision(decision=Decision.ALLOW, reasons=())


class MaxRiskRule:
    """Blocks when the highest available score meets `threshold`.

    The simplest combination rule: it does not weigh the original prompt
    against the reconstructed intent, it just asks whether *any* stage saw
    enough risk to refuse.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def decide(self, scores: RiskInputs) -> RuleDecision:
        return _max_risk_decision(scores, self.threshold)


class ConfidenceEscalationRule:
    """Escalates an ambiguous verdict to `Decision.CLARIFY` instead of
    guessing.

    Two situations count as ambiguous: the canonical score sits inside
    `clarify_band`, where neither "safe" nor "risky" is a confident read; or
    the canonicalizer itself had low confidence (`canonical_confidence < 0.4`)
    while the original prompt alone looked safe -- a case where the
    reconstruction cannot be trusted to vouch for the request either way.
    Outside both, this falls back to the same highest-score comparison
    `MaxRiskRule` makes.
    """

    #: Below this, the canonicalizer's own reading of the prompt is not
    #: trusted enough to let a low original score stand unquestioned.
    LOW_CONFIDENCE = 0.4

    def __init__(
        self, threshold: float = 0.5, clarify_band: tuple[float, float] = (0.35, 0.65)
    ) -> None:
        self.threshold = threshold
        self.clarify_band = clarify_band

    def decide(self, scores: RiskInputs) -> RuleDecision:
        canonical = scores.canonical
        low, high = self.clarify_band
        if canonical is not None and low <= canonical.score <= high:
            return RuleDecision(
                decision=Decision.CLARIFY,
                reasons=(f"{canonical.classifier}:ambiguous_risk:{canonical.score}",),
            )
        original = scores.original
        if (
            scores.canonical_confidence < self.LOW_CONFIDENCE
            and original is not None
            and original.score < self.threshold
        ):
            return RuleDecision(
                decision=Decision.CLARIFY,
                reasons=(f"{original.classifier}:low_canonicalizer_confidence",),
            )
        return _max_risk_decision(scores, self.threshold)
