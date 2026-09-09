"""The conventional filtering arms this study measures TIP-Guard against.

Each is a `ThresholdGuard` with a different classifier in a different slot,
so the comparison is between what a defence can *see* rather than between
implementations. Every knob a Phase 7 ablation touches is read from
`DefenseConfig.params` here, so switching a component off never needs a code
change.
"""

from typing import Any

from tipguard.classifiers.keyword import KeywordClassifier
from tipguard.classifiers.llm_risk import LLMRiskClassifier
from tipguard.classifiers.pattern import PatternClassifier
from tipguard.config.schemas import PoliciesConfig
from tipguard.models.registry import ProviderRegistry

from .output_guard import OutputGuard
from .threshold import ThresholdGuard

#: Which model rates risk when a defence needs one. A mock by default so the
#: whole baseline suite runs offline and in CI; real aliases are selected by
#: the experiment configs.
DEFAULT_CLASSIFIER_MODEL = "mock-judge"

DEFAULT_THRESHOLD = 0.5


def _input_threshold(params: dict[str, Any]) -> float:
    return float(params.get("input_threshold", DEFAULT_THRESHOLD))


def _output_threshold(params: dict[str, Any]) -> float:
    return float(params.get("output_threshold", DEFAULT_THRESHOLD))


def _leak_check(params: dict[str, Any]) -> bool:
    return bool(params.get("leak_check", True))


def _llm_classifier(
    params: dict[str, Any], registry: ProviderRegistry, policies: PoliciesConfig, name: str
) -> LLMRiskClassifier:
    alias = str(params.get("classifier_model", DEFAULT_CLASSIFIER_MODEL))
    return LLMRiskClassifier(provider=registry.get(alias), policies=policies, name=name)


def keyword_filter(
    params: dict[str, Any], registry: ProviderRegistry, main_model: str, system_prompt: str
) -> ThresholdGuard:
    return ThresholdGuard(
        name="keyword_filter",
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        input_classifier=KeywordClassifier(),
        input_threshold=_input_threshold(params),
    )


def pattern_detector(
    params: dict[str, Any], registry: ProviderRegistry, main_model: str, system_prompt: str
) -> ThresholdGuard:
    return ThresholdGuard(
        name="pattern_detector",
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        input_classifier=PatternClassifier(),
        input_threshold=_input_threshold(params),
    )


def input_classifier(
    params: dict[str, Any],
    registry: ProviderRegistry,
    main_model: str,
    system_prompt: str,
    policies: PoliciesConfig,
) -> ThresholdGuard:
    return ThresholdGuard(
        name="input_classifier",
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        input_classifier=_llm_classifier(params, registry, policies, "input_classifier"),
        input_threshold=_input_threshold(params),
    )


def output_classifier(
    params: dict[str, Any],
    registry: ProviderRegistry,
    main_model: str,
    system_prompt: str,
    policies: PoliciesConfig,
) -> ThresholdGuard:
    """No input screening at all: every prompt reaches the model, and only
    the answer is judged. This is the arm that shows what an input filter
    was worth, and the only baseline that can catch an attack whose request
    looks entirely benign."""
    return ThresholdGuard(
        name="output_classifier",
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        output_guard=OutputGuard(
            policies=policies,
            output_classifier=_llm_classifier(params, registry, policies, "output_classifier"),
            threshold=_output_threshold(params),
            leak_check=_leak_check(params),
        ),
    )


def input_output_classifier(
    params: dict[str, Any],
    registry: ProviderRegistry,
    main_model: str,
    system_prompt: str,
    policies: PoliciesConfig,
) -> ThresholdGuard:
    return ThresholdGuard(
        name="input_output_classifier",
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        input_classifier=_llm_classifier(params, registry, policies, "input_classifier"),
        output_guard=OutputGuard(
            policies=policies,
            output_classifier=_llm_classifier(params, registry, policies, "output_classifier"),
            threshold=_output_threshold(params),
            leak_check=_leak_check(params),
        ),
        input_threshold=_input_threshold(params),
    )
