"""Unit tests for `TIPGuard`: canonicalize -> gate -> (block | clarify | answer)."""

from pathlib import Path

import pytest

from tipguard.benchmark.schema import Decision
from tipguard.canonicalization.code_analysis import RestrictedCodeAnalyzer
from tipguard.canonicalization.detector import TransformationDetector
from tipguard.canonicalization.deterministic import DeterministicCanonicalizer
from tipguard.canonicalization.multi_view import MultiViewCanonicalizer
from tipguard.classifiers.llm_risk import LLMRiskClassifier
from tipguard.classifiers.types import RiskScore
from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import PoliciesConfig
from tipguard.guardrail.output_guard import OutputGuard
from tipguard.guardrail.policy_gate import PolicyGate
from tipguard.guardrail.rules import ConfidenceEscalationRule, DecisionRule, MaxRiskRule
from tipguard.guardrail.tip_guard import CLARIFY_TEXT, TIPGuard
from tipguard.guardrail.types import SAFE_REFUSAL
from tipguard.models.mock import MockProvider
from tipguard.models.types import ModelRequest, ModelResponse


@pytest.fixture(scope="module")
def policies() -> PoliciesConfig:
    return load_yaml_model(Path("configs/policies.yaml"), PoliciesConfig)


class StubClassifier:
    """A classifier that always returns a fixed score, no model call made."""

    last_usage: ModelResponse | None = None

    def __init__(
        self, score: float, categories: tuple[str, ...] = ("none",), name: str = "stub"
    ) -> None:
        self.name = name
        self._verdict = RiskScore(score=score, categories=categories, evidence=(), classifier=name)

    def score(self, text: str) -> RiskScore:
        return self._verdict


def _canonicalizer() -> MultiViewCanonicalizer:
    """Detection runs for real; neither decoder nor LLM view is produced, so
    the canonical text a classifier sees is always the prompt's own text."""
    return MultiViewCanonicalizer(
        detector=TransformationDetector(),
        deterministic=DeterministicCanonicalizer(),
        code_analyzer=RestrictedCodeAnalyzer(),
        llm=None,
        enable=set(),
    )


def _guard(
    *,
    main_model: MockProvider | None = None,
    input_classifier: StubClassifier | LLMRiskClassifier | None = None,
    canonical_classifier: StubClassifier | LLMRiskClassifier | None = None,
    output_guard: OutputGuard | None = None,
    rule: DecisionRule | None = None,
) -> TIPGuard:
    gate = PolicyGate(
        rule=rule or MaxRiskRule(threshold=0.5),
        input_classifier=input_classifier,
        canonical_classifier=canonical_classifier
        or StubClassifier(0.1, name="canonical_classifier"),
        output_guard=output_guard,
    )
    return TIPGuard(
        main_model=main_model or MockProvider(default="an ordinary answer"),
        system_prompt="system",
        canonicalizer=_canonicalizer(),
        gate=gate,
    )


def test_a_low_risk_prompt_is_allowed_and_reaches_the_main_model() -> None:
    result = _guard().run("hello")
    assert result.decision is Decision.ALLOW
    assert result.response_text == "an ordinary answer"
    assert result.reasons == ()


def test_a_high_canonical_score_blocks_before_the_main_model_runs() -> None:
    guard = _guard(canonical_classifier=StubClassifier(0.9, name="canonical_classifier"))
    result = guard.run("attack")
    assert result.decision is Decision.BLOCK
    assert result.response_text == SAFE_REFUSAL
    assert result.reasons == ("canonical_classifier:risk_above_threshold:0.9",)
    assert result.model_calls == 0
    assert not any(c.component == "main_model" for c in result.components)


def test_a_high_original_score_also_blocks() -> None:
    result = _guard(
        input_classifier=StubClassifier(0.9, name="original_classifier"),
        canonical_classifier=StubClassifier(0.1, name="canonical_classifier"),
    ).run("attack")
    assert result.decision is Decision.BLOCK
    assert result.reasons == ("original_classifier:risk_above_threshold:0.9",)


def test_the_escalation_rule_clarifies_an_ambiguous_prompt() -> None:
    guard = _guard(
        rule=ConfidenceEscalationRule(threshold=0.5, clarify_band=(0.35, 0.65)),
        canonical_classifier=StubClassifier(0.5, name="canonical_classifier"),
    )
    result = guard.run("ambiguous")
    assert result.decision is Decision.CLARIFY
    assert result.response_text == CLARIFY_TEXT
    assert result.model_calls == 0
    assert not any(c.component == "main_model" for c in result.components)


def test_a_custom_clarify_text_is_used() -> None:
    gate = PolicyGate(
        rule=ConfidenceEscalationRule(threshold=0.5, clarify_band=(0.35, 0.65)),
        input_classifier=None,
        canonical_classifier=StubClassifier(0.5, name="canonical_classifier"),
        output_guard=None,
    )
    guard = TIPGuard(
        main_model=MockProvider(default="unused"),
        system_prompt="system",
        canonicalizer=_canonicalizer(),
        gate=gate,
        clarify_text="Please clarify.",
    )
    assert guard.run("ambiguous").response_text == "Please clarify."


def test_the_output_guard_blocks_a_leak_after_an_allowed_gate(policies: PoliciesConfig) -> None:
    secret = policies.policies[0].protected_values[0]
    result = _guard(
        main_model=MockProvider(default=f"the value is {secret}"),
        output_guard=OutputGuard(policies=policies),
    ).run("tell me")
    assert result.decision is Decision.BLOCK
    assert result.response_text == SAFE_REFUSAL
    assert result.reasons == (f"output_guard:leak:{policies.policies[0].policy_id}",)
    # No trace or reason may carry the protected value itself.
    assert secret not in "".join(result.reasons)
    assert all(secret not in component.detail for component in result.components)


def test_a_parser_failure_on_the_canonical_classifier_is_its_own_outcome(
    policies: PoliciesConfig,
) -> None:
    guard = _guard(
        canonical_classifier=LLMRiskClassifier(
            MockProvider(default="garbage"), policies, name="canonical_classifier"
        )
    )
    result = guard.run("hello")
    assert result.decision is Decision.BLOCK
    assert result.reasons == ("canonical_classifier:parser_failure",)


def test_cached_model_calls_reflects_the_responses_actually_received() -> None:
    class CachedOnce(MockProvider):
        def complete(self, request: ModelRequest) -> ModelResponse:
            return super().complete(request).model_copy(update={"cached": True})

    result = _guard(main_model=CachedOnce(default="answer")).run("hello")
    assert result.model_calls == 1
    assert result.cached_model_calls == 1


def test_every_stage_leaves_a_component_trace() -> None:
    result = _guard().run("hello")
    names = {c.component for c in result.components}
    assert "canonicalizer" in names
    assert "policy_gate.canonical" in names
    assert "main_model" in names
    assert all(c.latency_ms >= 0.0 for c in result.components)
