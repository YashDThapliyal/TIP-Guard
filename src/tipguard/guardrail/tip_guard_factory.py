"""Build a `TIPGuard` pipeline from a `DefenseConfig`'s params.

Kept out of `guardrail.factory` and `guardrail.baselines` because this
defense, unlike the baselines, assembles a whole canonicalization pipeline as
well as a classifier -- and out of `guardrail.tip_guard` itself, which stays
free of config parsing so it can be constructed directly in tests.
"""

from typing import Any

from tipguard.canonicalization.code_analysis import RestrictedCodeAnalyzer
from tipguard.canonicalization.detector import TransformationDetector
from tipguard.canonicalization.deterministic import DeterministicCanonicalizer
from tipguard.canonicalization.llm_canonicalizer import LLMCanonicalizer
from tipguard.canonicalization.multi_view import MultiViewCanonicalizer
from tipguard.canonicalization.types import DetectionResult
from tipguard.classifiers.llm_risk import LLMRiskClassifier
from tipguard.classifiers.prompts import PROMPT_VERSION
from tipguard.config.loader import ConfigError
from tipguard.config.schemas import PoliciesConfig
from tipguard.models.registry import ProviderRegistry

from .output_guard import OutputGuard
from .params import check_known_params, parse_bool, parse_threshold
from .policy_gate import PolicyGate
from .rules import ConfidenceEscalationRule, DecisionRule, MaxRiskRule
from .tip_guard import TIPGuard

#: Same reasoning as `guardrail.baselines.DEFAULT_CLASSIFIER_MODEL`: a mock
#: by default so the pipeline runs offline and in CI.
DEFAULT_CANONICALIZER_MODEL = "mock-judge"
DEFAULT_CLASSIFIER_MODEL = "mock-judge"
DEFAULT_THRESHOLD = 0.5
DEFAULT_RULE = "max"

KNOWN_PARAMS = frozenset(
    {
        "canonicalizer_model",
        "classifier_model",
        "prompt_version",
        "rule",
        "threshold",
        "enable_detector",
        "enable_decoders",
        "enable_code_analyzer",
        "enable_llm_canonicalizer",
        "enable_original_classifier",
        "enable_output_guard",
    }
)

_RULES: dict[str, type[MaxRiskRule | ConfidenceEscalationRule]] = {
    "max": MaxRiskRule,
    "escalation": ConfidenceEscalationRule,
}


class _NoOpDetector(TransformationDetector):
    """A detector that never sees a transformation, for `enable_detector=False`.

    `Canonicalization.detection` is a required field no `enable` switch can
    omit (see `MultiViewCanonicalizer`'s own docstring), and
    `TransformationDetector` has no switch of its own -- so ablating
    detection means substituting a detector that always reports nothing,
    rather than removing a step from the pipeline.
    """

    def detect(self, text: str) -> DetectionResult:
        return DetectionResult(
            has_transformation=False,
            family=None,
            implies_code=False,
            confidence=0.0,
            evidence=(),
        )


def _rule(params: dict[str, Any], threshold: float, defense: str) -> DecisionRule:
    name = str(params.get("rule", DEFAULT_RULE))
    if name not in _RULES:
        raise ConfigError(
            f"defense {defense!r}: rule must be one of {sorted(_RULES)}, got {name!r}"
        )
    return _RULES[name](threshold=threshold)


def build_tip_guard(
    params: dict[str, Any],
    registry: ProviderRegistry,
    main_model: str,
    system_prompt: str,
    policies: PoliciesConfig,
) -> TIPGuard:
    name = "tip_guard"
    check_known_params(params, KNOWN_PARAMS, name)

    threshold = parse_threshold(params, "threshold", DEFAULT_THRESHOLD, name)
    canonicalizer_alias = str(params.get("canonicalizer_model", DEFAULT_CANONICALIZER_MODEL))
    classifier_alias = str(params.get("classifier_model", DEFAULT_CLASSIFIER_MODEL))
    # One version for all three classifiers this builds. The study compares
    # TIP-Guard against the calibrated-prompt baseline, so leaving these on
    # the defective v1 prompt would measure that defect instead of
    # canonicalization; letting the three stages disagree would measure
    # neither version.
    prompt_version = str(params.get("prompt_version", PROMPT_VERSION))

    enable_detector = parse_bool(params, "enable_detector", True, name)
    enable_decoders = parse_bool(params, "enable_decoders", True, name)
    # `RestrictedCodeAnalyzer` is always built: `MultiViewCanonicalizer.run`
    # does not call it directly today (see its own docstring), so there is
    # nothing yet for this switch to turn off. Still parsed and validated so
    # a config naming it gets a real boolean check rather than a silent
    # "unknown param" rejection once a later phase wires the analyzer in.
    parse_bool(params, "enable_code_analyzer", True, name)
    enable_llm_canonicalizer = parse_bool(params, "enable_llm_canonicalizer", True, name)
    enable_original_classifier = parse_bool(params, "enable_original_classifier", True, name)
    enable_output_guard = parse_bool(params, "enable_output_guard", True, name)

    canonicalize_enable: set[str] = set()
    if enable_decoders:
        canonicalize_enable.add("deterministic")
    llm_canonicalizer: LLMCanonicalizer | None = None
    if enable_llm_canonicalizer:
        canonicalize_enable.add("llm")
        llm_canonicalizer = LLMCanonicalizer(
            provider=registry.get(canonicalizer_alias),
            policies=policies,
            name="llm_canonicalizer",
        )

    canonicalizer = MultiViewCanonicalizer(
        detector=TransformationDetector() if enable_detector else _NoOpDetector(),
        deterministic=DeterministicCanonicalizer(),
        code_analyzer=RestrictedCodeAnalyzer(),
        llm=llm_canonicalizer,
        enable=canonicalize_enable,
    )

    input_classifier = None
    if enable_original_classifier:
        input_classifier = LLMRiskClassifier(
            provider=registry.get(classifier_alias),
            policies=policies,
            name="original_classifier",
            prompt_version=prompt_version,
        )
    canonical_classifier = LLMRiskClassifier(
        provider=registry.get(classifier_alias),
        policies=policies,
        name="canonical_classifier",
        prompt_version=prompt_version,
    )

    output_guard: OutputGuard | None = None
    if enable_output_guard:
        output_guard = OutputGuard(
            policies=policies,
            output_classifier=LLMRiskClassifier(
                provider=registry.get(classifier_alias),
                policies=policies,
                name="output_classifier",
                prompt_version=prompt_version,
            ),
            threshold=threshold,
        )

    gate = PolicyGate(
        rule=_rule(params, threshold, name),
        input_classifier=input_classifier,
        canonical_classifier=canonical_classifier,
        output_guard=output_guard,
    )

    return TIPGuard(
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        canonicalizer=canonicalizer,
        gate=gate,
    )
