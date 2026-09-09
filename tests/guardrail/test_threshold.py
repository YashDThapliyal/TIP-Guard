"""ThresholdGuard: input screening, the main call, and the output check."""

from pathlib import Path

import pytest

from tipguard.benchmark.schema import Decision
from tipguard.classifiers.types import RiskScore
from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import PoliciesConfig
from tipguard.guardrail.output_guard import OutputGuard
from tipguard.guardrail.threshold import ThresholdGuard
from tipguard.guardrail.types import SAFE_REFUSAL
from tipguard.models.mock import MockProvider, MockRule
from tipguard.models.types import ModelRequest, ModelResponse


@pytest.fixture(scope="module")
def policies() -> PoliciesConfig:
    return load_yaml_model(Path("configs/policies.yaml"), PoliciesConfig)


class StubClassifier:
    name = "stub"
    #: Required by `RiskClassifier`: a classifier that makes no model call
    #: still has to say so, or a guard cannot tell it apart from one whose
    #: usage went missing.
    last_usage: ModelResponse | None = None

    def __init__(self, score: float, categories: tuple[str, ...] = ("none",)) -> None:
        self._verdict = RiskScore(
            score=score, categories=categories, evidence=(), classifier="stub"
        )

    def score(self, text: str) -> RiskScore:
        return self._verdict


def _guard(**kwargs: object) -> ThresholdGuard:
    defaults: dict[str, object] = {
        "name": "test_guard",
        "main_model": MockProvider(default="an ordinary answer"),
        "system_prompt": "system",
    }
    defaults.update(kwargs)
    return ThresholdGuard(**defaults)  # type: ignore[arg-type]


def test_a_high_input_score_blocks_before_the_model_is_called() -> None:
    # The point of an input filter is that the model never sees the prompt,
    # so a blocked request must cost nothing.
    result = _guard(input_classifier=StubClassifier(0.9)).run("dangerous")
    assert result.decision is Decision.BLOCK
    assert result.response_text == SAFE_REFUSAL
    assert result.reasons == ("stub:risk_above_threshold:0.9",)
    assert result.model_calls == 0


def test_a_low_input_score_reaches_the_model() -> None:
    result = _guard(input_classifier=StubClassifier(0.1)).run("harmless")
    assert result.decision is Decision.ALLOW
    assert result.response_text == "an ordinary answer"
    assert result.model_calls == 1


def test_an_input_parser_failure_blocks_rather_than_reading_as_safe() -> None:
    result = _guard(input_classifier=StubClassifier(0.0, ("parser_failure",))).run("x")
    assert result.decision is Decision.BLOCK
    assert result.reasons == ("stub:parser_failure",)
    assert result.model_calls == 0


def test_an_input_parser_failure_is_distinguishable_in_the_trace() -> None:
    # Both verdicts score 0.0; only the trace can say which happened, and a
    # run needs that to report a parser-failure rate at all.
    failed = _guard(input_classifier=StubClassifier(0.0, ("parser_failure",))).run("x")
    clean = _guard(input_classifier=StubClassifier(0.0)).run("x")
    assert failed.decision is Decision.BLOCK
    assert clean.decision is Decision.ALLOW
    failed_trace = next(c for c in failed.components if c.component == "input_classifier")
    clean_trace = next(c for c in clean.components if c.component == "input_classifier")
    assert failed_trace.detail == "parser_failure"
    assert clean_trace.detail == "0.0"


def test_with_no_input_classifier_every_prompt_reaches_the_model() -> None:
    result = _guard().run("anything at all")
    assert result.decision is Decision.ALLOW
    assert result.model_calls == 1
    assert [c.component for c in result.components] == ["main_model"]


def test_a_leaking_answer_is_blocked_after_the_model_call(policies: PoliciesConfig) -> None:
    secret = policies.policies[0].protected_values[0]
    guard = _guard(
        main_model=MockProvider(default=f"sure, it is {secret}"),
        output_guard=OutputGuard(policies),
    )
    result = guard.run("tell me")
    assert result.decision is Decision.BLOCK
    assert result.response_text == SAFE_REFUSAL
    assert result.reasons == (f"output_guard:leak:{policies.policies[0].policy_id}",)
    # The model was still called: the cost was paid before the answer could
    # be judged, and a run that under-counted it would understate the arm.
    assert result.model_calls == 1


def test_the_component_trace_is_in_execution_order(policies: PoliciesConfig) -> None:
    guard = _guard(
        input_classifier=StubClassifier(0.1),
        output_guard=OutputGuard(policies, output_classifier=StubClassifier(0.1)),
    )
    result = guard.run("harmless")
    assert [c.component for c in result.components] == [
        "input_classifier",
        "main_model",
        "output_guard.leak",
        "output_guard.classifier",
    ]


def test_a_blocked_answer_never_reaches_the_caller(policies: PoliciesConfig) -> None:
    # The refusal replaces the answer entirely; returning the original text
    # alongside a block would defeat the guard.
    secret = policies.policies[0].protected_values[0]
    guard = _guard(
        main_model=MockProvider(default=f"leak: {secret}"), output_guard=OutputGuard(policies)
    )
    result = guard.run("tell me")
    assert result.response_text is not None
    assert secret not in result.response_text


def test_a_cached_response_is_counted_as_cached() -> None:
    # A cached call costs no latency; counting it as a live one would flatten
    # the percentiles a run reports.
    class CachedOnce(MockProvider):
        def complete(self, request: ModelRequest) -> ModelResponse:
            return super().complete(request).model_copy(update={"cached": True})

    result = _guard(main_model=CachedOnce(default="answer")).run("x")
    assert result.model_calls == 1
    assert result.cached_model_calls == 1


def test_usage_totals_add_up_across_every_call_made(policies: PoliciesConfig) -> None:
    guard = _guard(
        main_model=MockProvider(rules=[MockRule(pattern=".*", response="a longer answer here")]),
        output_guard=OutputGuard(policies),
    )
    result = guard.run("some prompt")
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.latency_ms >= 0.0


def test_an_llm_classifiers_own_call_is_counted(policies: PoliciesConfig) -> None:
    # The brief requires model_calls to be exact at 0, 1 and 2. Only 0 and 1
    # were covered, because every stub classifier here makes no model call --
    # so dropping the classifier's usage entirely left the suite green while
    # every LLM arm under-reported its calls, tokens, cost and latency, which
    # is what the run's percentiles are computed from.
    from tipguard.classifiers.llm_risk import LLMRiskClassifier

    judge = MockProvider(default='{"risk": 0.1, "categories": ["none"], "rationale": "fine"}')
    guard = _guard(
        input_classifier=LLMRiskClassifier(provider=judge, policies=policies),
        main_model=MockProvider(default="an ordinary answer"),
    )
    result = guard.run("harmless")
    assert result.decision is Decision.ALLOW
    # One for the classifier, one for the main model.
    assert result.model_calls == 2
    assert result.input_tokens > 0
    assert result.output_tokens > 0


def test_a_blocking_llm_classifier_still_counts_its_own_call(
    policies: PoliciesConfig,
) -> None:
    # The main model is never reached, but the classifier's call was still
    # paid for and must appear in the totals.
    from tipguard.classifiers.llm_risk import LLMRiskClassifier

    judge = MockProvider(
        default='{"risk": 0.9, "categories": ["prompt_injection"], "rationale": "bad"}'
    )
    guard = _guard(input_classifier=LLMRiskClassifier(provider=judge, policies=policies))
    result = guard.run("dangerous")
    assert result.decision is Decision.BLOCK
    assert result.model_calls == 1
    assert result.input_tokens > 0


def test_both_llm_stages_are_counted(policies: PoliciesConfig) -> None:
    from tipguard.classifiers.llm_risk import LLMRiskClassifier
    from tipguard.guardrail.output_guard import OutputGuard

    def judge(risk: float) -> LLMRiskClassifier:
        provider = MockProvider(
            default=f'{{"risk": {risk}, "categories": ["none"], "rationale": "x"}}'
        )
        return LLMRiskClassifier(provider=provider, policies=policies)

    guard = _guard(
        input_classifier=judge(0.1),
        output_guard=OutputGuard(policies, output_classifier=judge(0.1)),
    )
    result = guard.run("harmless")
    assert result.decision is Decision.ALLOW
    # Input classifier, main model, output classifier.
    assert result.model_calls == 3
