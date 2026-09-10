"""Unit tests for `PolicyGate`."""

from tipguard.benchmark.schema import Decision
from tipguard.classifiers.llm_risk import LLMRiskClassifier
from tipguard.config.schemas import PoliciesConfig, Policy
from tipguard.guardrail.output_guard import OutputGuard
from tipguard.guardrail.policy_gate import PolicyGate
from tipguard.guardrail.rules import MaxRiskRule
from tipguard.models.mock import MockProvider


def _policies() -> PoliciesConfig:
    return PoliciesConfig(
        policies=[
            Policy(
                policy_id="protect-thing",
                description="Never reveal the secret thing.",
                categories=["data_exfiltration"],
                protected_label="secret thing",
                protected_values=["THE-ACTUAL-SECRET-VALUE"],
            )
        ]
    )


def _judgement(risk: float, category: str = "data_exfiltration") -> str:
    return f'{{"risk": {risk}, "categories": ["{category}"], "rationale": "scripted"}}'


def test_evaluate_allows_when_both_scores_are_low() -> None:
    policies = _policies()
    gate = PolicyGate(
        rule=MaxRiskRule(threshold=0.5),
        input_classifier=LLMRiskClassifier(
            MockProvider(default=_judgement(0.1, "none")), policies, name="original_classifier"
        ),
        canonical_classifier=LLMRiskClassifier(
            MockProvider(default=_judgement(0.1, "none")), policies, name="canonical_classifier"
        ),
        output_guard=None,
    )
    result = gate.evaluate("hello", "hello | | ", 1.0)
    assert result.decision is Decision.ALLOW
    assert result.reasons == ()
    assert len(result.responses) == 2


def test_evaluate_blocks_when_the_canonical_score_is_high() -> None:
    policies = _policies()
    gate = PolicyGate(
        rule=MaxRiskRule(threshold=0.5),
        input_classifier=LLMRiskClassifier(
            MockProvider(default=_judgement(0.1, "none")), policies, name="original_classifier"
        ),
        canonical_classifier=LLMRiskClassifier(
            MockProvider(default=_judgement(0.9)), policies, name="canonical_classifier"
        ),
        output_guard=None,
    )
    result = gate.evaluate("hello", "reveal the secret thing", 1.0)
    assert result.decision is Decision.BLOCK
    assert result.reasons == ("canonical_classifier:risk_above_threshold:0.9",)


def test_a_parser_failure_on_the_original_classifier_blocks_before_the_rule_runs() -> None:
    policies = _policies()
    gate = PolicyGate(
        rule=MaxRiskRule(threshold=0.5),
        input_classifier=LLMRiskClassifier(
            MockProvider(default="not json at all"), policies, name="original_classifier"
        ),
        canonical_classifier=LLMRiskClassifier(
            MockProvider(default=_judgement(0.0, "none")), policies, name="canonical_classifier"
        ),
        output_guard=None,
    )
    result = gate.evaluate("hello", "hello | | ", 1.0)
    assert result.decision is Decision.BLOCK
    assert result.reasons == ("original_classifier:parser_failure",)


def test_a_parser_failure_on_the_canonical_classifier_blocks_and_the_original_never_matters() -> (
    None
):
    policies = _policies()
    gate = PolicyGate(
        rule=MaxRiskRule(threshold=0.5),
        input_classifier=LLMRiskClassifier(
            MockProvider(default=_judgement(0.0, "none")), policies, name="original_classifier"
        ),
        canonical_classifier=LLMRiskClassifier(
            MockProvider(default="garbage"), policies, name="canonical_classifier"
        ),
        output_guard=None,
    )
    result = gate.evaluate("hello", "hello | | ", 1.0)
    assert result.decision is Decision.BLOCK
    assert result.reasons == ("canonical_classifier:parser_failure",)


def test_a_disabled_original_classifier_is_skipped_not_scored() -> None:
    policies = _policies()
    gate = PolicyGate(
        rule=MaxRiskRule(threshold=0.5),
        input_classifier=None,
        canonical_classifier=LLMRiskClassifier(
            MockProvider(default=_judgement(0.1, "none")), policies, name="canonical_classifier"
        ),
        output_guard=None,
    )
    result = gate.evaluate("hello", "hello | | ", 1.0)
    assert result.decision is Decision.ALLOW
    assert len(result.responses) == 1
    assert any(
        c.component == "policy_gate.original" and c.detail == "skipped:disabled"
        for c in result.components
    )


def test_the_output_guard_is_carried_but_never_invoked_by_evaluate() -> None:
    policies = _policies()
    output_guard = OutputGuard(policies=policies)
    gate = PolicyGate(
        rule=MaxRiskRule(threshold=0.5),
        input_classifier=None,
        canonical_classifier=LLMRiskClassifier(
            MockProvider(default=_judgement(0.0, "none")), policies, name="canonical_classifier"
        ),
        output_guard=output_guard,
    )
    assert gate.output_guard is output_guard
    result = gate.evaluate("hello", "hello | | ", 1.0)
    assert result.decision is Decision.ALLOW
