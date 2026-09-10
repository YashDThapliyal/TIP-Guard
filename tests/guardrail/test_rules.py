"""Unit tests for `DecisionRule`s: `MaxRiskRule` and `ConfidenceEscalationRule`."""

import re

import pytest

from tipguard.benchmark.schema import Decision
from tipguard.classifiers.types import RiskScore
from tipguard.guardrail.rules import ConfidenceEscalationRule, MaxRiskRule, RiskInputs

REASON_PATTERN = re.compile(r"^[a-z_]+:[a-z_]+(:.*)?$")


def score(
    value: float, classifier: str = "canonical_classifier", categories: tuple[str, ...] = ()
) -> RiskScore:
    return RiskScore(score=value, categories=categories, evidence=(), classifier=classifier)


def inputs(
    *,
    original: RiskScore | None = None,
    canonical: RiskScore | None = None,
    canonical_confidence: float = 1.0,
    output: RiskScore | None = None,
) -> RiskInputs:
    return RiskInputs(
        original=original,
        canonical=canonical,
        canonical_confidence=canonical_confidence,
        output=output,
    )


class TestMaxRiskRule:
    def test_allows_when_nothing_is_available(self) -> None:
        rule = MaxRiskRule(threshold=0.5)
        result = rule.decide(inputs())
        assert result.decision is Decision.ALLOW
        assert result.reasons == ()

    def test_allows_below_threshold(self) -> None:
        rule = MaxRiskRule(threshold=0.5)
        result = rule.decide(inputs(original=score(0.2), canonical=score(0.3)))
        assert result.decision is Decision.ALLOW

    def test_blocks_when_the_highest_score_meets_the_threshold(self) -> None:
        rule = MaxRiskRule(threshold=0.5)
        result = rule.decide(
            inputs(original=score(0.1), canonical=score(0.5, "canonical_classifier"))
        )
        assert result.decision is Decision.BLOCK
        assert result.reasons == ("canonical_classifier:risk_above_threshold:0.5",)
        assert all(REASON_PATTERN.match(reason) for reason in result.reasons)

    def test_the_original_score_can_be_the_one_that_blocks(self) -> None:
        rule = MaxRiskRule(threshold=0.5)
        result = rule.decide(
            inputs(original=score(0.9, "original_classifier"), canonical=score(0.1))
        )
        assert result.decision is Decision.BLOCK
        assert result.reasons == ("original_classifier:risk_above_threshold:0.9",)

    def test_ignores_a_score_that_was_never_produced(self) -> None:
        rule = MaxRiskRule(threshold=0.5)
        result = rule.decide(inputs(canonical=score(0.9)))
        assert result.decision is Decision.BLOCK


class TestConfidenceEscalationRule:
    def test_clarifies_when_the_canonical_score_sits_inside_the_band(self) -> None:
        rule = ConfidenceEscalationRule(threshold=0.5, clarify_band=(0.35, 0.65))
        result = rule.decide(inputs(canonical=score(0.5)))
        assert result.decision is Decision.CLARIFY
        assert all(REASON_PATTERN.match(reason) for reason in result.reasons)

    @pytest.mark.parametrize("value", [0.35, 0.5, 0.65])
    def test_the_band_is_inclusive_at_both_ends(self, value: float) -> None:
        rule = ConfidenceEscalationRule(threshold=0.5, clarify_band=(0.35, 0.65))
        result = rule.decide(inputs(canonical=score(value)))
        assert result.decision is Decision.CLARIFY

    def test_clarifies_on_low_canonicalizer_confidence_with_a_safe_original_score(self) -> None:
        rule = ConfidenceEscalationRule(threshold=0.5)
        result = rule.decide(
            inputs(
                original=score(0.1, "original_classifier"),
                canonical=score(0.9),
                canonical_confidence=0.1,
            )
        )
        # The canonical score of 0.9 is outside the clarify band and would
        # otherwise block outright -- but a low-confidence canonicalization
        # combined with a *safe* original score is exactly the case this
        # branch exists to catch before either extreme is trusted. Since the
        # low-confidence branch is checked first, it wins.
        assert result.decision is Decision.CLARIFY

    def test_does_not_clarify_on_low_confidence_when_the_original_score_is_already_risky(
        self,
    ) -> None:
        rule = ConfidenceEscalationRule(threshold=0.5)
        result = rule.decide(
            inputs(
                original=score(0.8, "original_classifier"),
                canonical=score(0.9, "canonical_classifier"),
                canonical_confidence=0.1,
            )
        )
        assert result.decision is Decision.BLOCK

    def test_falls_back_to_max_risk_blocking_outside_the_ambiguous_cases(self) -> None:
        rule = ConfidenceEscalationRule(threshold=0.5, clarify_band=(0.35, 0.65))
        result = rule.decide(
            inputs(original=score(0.9, "original_classifier"), canonical_confidence=1.0)
        )
        assert result.decision is Decision.BLOCK
        assert result.reasons == ("original_classifier:risk_above_threshold:0.9",)

    def test_allows_a_clearly_safe_case(self) -> None:
        rule = ConfidenceEscalationRule(threshold=0.5, clarify_band=(0.35, 0.65))
        result = rule.decide(
            inputs(original=score(0.1), canonical=score(0.1), canonical_confidence=1.0)
        )
        assert result.decision is Decision.ALLOW
