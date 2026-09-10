"""The conventional filtering arms this study measures TIP-Guard against.

Each is a `ThresholdGuard` with a different classifier in a different slot,
so the comparison is between what a defence can *see* rather than between
implementations. Every knob a Phase 7 ablation touches is read from
`DefenseConfig.params` here, so switching a component off never needs a code
change.
"""

from collections.abc import Mapping
from typing import Any

from tipguard.classifiers.keyword import KeywordClassifier
from tipguard.classifiers.llm_risk import LLMRiskClassifier
from tipguard.classifiers.pattern import PatternClassifier
from tipguard.classifiers.prompts import PROMPT_VERSION
from tipguard.config.loader import ConfigError
from tipguard.config.schemas import PoliciesConfig
from tipguard.models.registry import ProviderRegistry

from .output_guard import OutputGuard
from .threshold import ThresholdGuard

#: Every parameter a baseline reads, and what it resolves to when a config
#: omits it. This is the single source of both, so that a run's manifest can
#: record what an omitted parameter actually became -- an earlier format
#: stored the config as written, which meant the threshold every study arm ran
#: at was never recorded anywhere and had to be inferred from the decisions.
#:
#: The parse calls below read from this mapping rather than from literals, so
#: a default cannot be changed in one place and reported from another.
DEFAULTS: dict[str, Any] = {
    "classifier_model": "mock-judge",
    "prompt_version": PROMPT_VERSION,
    "input_threshold": 0.5,
    "output_threshold": 0.5,
    "leak_check": True,
}

KNOWN_PARAMS = frozenset(DEFAULTS)

#: Which of those parameters each baseline actually reads.
#:
#: `KNOWN_PARAMS` is the union, and validating every arm against the union was
#: too loose in both directions: a config could set `output_threshold` on
#: `keyword_filter` and have it silently ignored, and a run's manifest would
#: then report `classifier_model` and `prompt_version` as that filter's
#: effective settings when it never consults a model at all. A manifest that
#: names settings which had no effect is not a record of what ran.
PARAMS_BY_DEFENSE: dict[str, frozenset[str]] = {
    "keyword_filter": frozenset({"input_threshold"}),
    "pattern_detector": frozenset({"input_threshold"}),
    "input_classifier": frozenset({"classifier_model", "prompt_version", "input_threshold"}),
    "output_classifier": frozenset(
        {"classifier_model", "prompt_version", "output_threshold", "leak_check"}
    ),
    "input_output_classifier": KNOWN_PARAMS,
}

#: Which model rates risk when a defence needs one. A mock by default so the
#: whole baseline suite runs offline and in CI; real aliases are selected by
#: the experiment configs.
DEFAULT_CLASSIFIER_MODEL = DEFAULTS["classifier_model"]

DEFAULT_THRESHOLD = DEFAULTS["input_threshold"]


#: Every key a baseline understands. A `DefenseConfig.params` is an open
#: mapping, so a mistyped key would otherwise fall back to the default in
#: silence -- and a Phase 7 threshold sweep that mistyped one would report the
#: default arm as though it had swept.

_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"false", "no", "off", "0"})


def _check_params(params: Mapping[str, Any], defense: str) -> None:
    accepted = PARAMS_BY_DEFENSE[defense]
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise ConfigError(
            f"defense {defense!r} got unknown params {unknown}; known params: {sorted(accepted)}"
        )


def _threshold(params: Mapping[str, Any], key: str, defense: str) -> float:
    raw = params.get(key, DEFAULTS[key])
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"defense {defense!r}: {key} must be a number, got {raw!r}") from exc
    if not 0.0 <= value <= 1.01:
        # 1.01 rather than 1.0 so a config can express "never block": a score
        # is capped at 1.0 and the comparison is `>=`, so 1.0 would still fire
        # on a maximum-risk verdict.
        raise ConfigError(f"defense {defense!r}: {key} must be within [0, 1.01], got {value}")
    return value


def _leak_check(params: Mapping[str, Any], defense: str) -> bool:
    """Whether the output guard runs its leak scan.

    Spelled out rather than passed through `bool`, because `bool("false")` is
    True: a config saying `leak_check: "false"` would leave the scan running
    while the run reported it as ablated, which corrupts the measurement the
    ablation exists to produce rather than merely failing.
    """
    raw = params.get("leak_check", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        if raw.strip().lower() in _TRUE:
            return True
        if raw.strip().lower() in _FALSE:
            return False
    raise ConfigError(f"defense {defense!r}: leak_check must be a boolean, got {raw!r}")


def _llm_classifier(
    params: dict[str, Any], registry: ProviderRegistry, policies: PoliciesConfig, name: str
) -> LLMRiskClassifier:
    """Build the risk classifier an arm screens with.

    `prompt_version` is a config parameter rather than a constant because the
    study runs the same defence under two risk prompts, and which one a run
    used has to survive into its artifacts -- an arm that reported `risk-v2`
    while scoring with `risk-v1` would invert the finding it exists to
    support. `LLMRiskClassifier` rejects an unknown version, so a typo fails
    the build instead of quietly scoring with the default.
    """
    alias = str(params.get("classifier_model", DEFAULTS["classifier_model"]))
    return LLMRiskClassifier(
        provider=registry.get(alias),
        policies=policies,
        name=name,
        prompt_version=str(params.get("prompt_version", DEFAULTS["prompt_version"])),
    )


def keyword_filter(
    params: dict[str, Any], registry: ProviderRegistry, main_model: str, system_prompt: str
) -> ThresholdGuard:
    name = "keyword_filter"
    _check_params(params, name)
    return ThresholdGuard(
        name=name,
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        input_classifier=KeywordClassifier(),
        input_threshold=_threshold(params, "input_threshold", name),
    )


def pattern_detector(
    params: dict[str, Any], registry: ProviderRegistry, main_model: str, system_prompt: str
) -> ThresholdGuard:
    name = "pattern_detector"
    _check_params(params, name)
    return ThresholdGuard(
        name=name,
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        input_classifier=PatternClassifier(),
        input_threshold=_threshold(params, "input_threshold", name),
    )


def input_classifier(
    params: dict[str, Any],
    registry: ProviderRegistry,
    main_model: str,
    system_prompt: str,
    policies: PoliciesConfig,
) -> ThresholdGuard:
    name = "input_classifier"
    _check_params(params, name)
    return ThresholdGuard(
        name=name,
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        input_classifier=_llm_classifier(params, registry, policies, "input_classifier"),
        input_threshold=_threshold(params, "input_threshold", name),
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
    name = "output_classifier"
    _check_params(params, name)
    return ThresholdGuard(
        name=name,
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        output_guard=OutputGuard(
            policies=policies,
            output_classifier=_llm_classifier(params, registry, policies, "output_classifier"),
            threshold=_threshold(params, "output_threshold", name),
            leak_check=_leak_check(params, name),
        ),
    )


def input_output_classifier(
    params: dict[str, Any],
    registry: ProviderRegistry,
    main_model: str,
    system_prompt: str,
    policies: PoliciesConfig,
) -> ThresholdGuard:
    name = "input_output_classifier"
    _check_params(params, name)
    return ThresholdGuard(
        name=name,
        main_model=registry.get(main_model),
        system_prompt=system_prompt,
        input_classifier=_llm_classifier(params, registry, policies, "input_classifier"),
        output_guard=OutputGuard(
            policies=policies,
            output_classifier=_llm_classifier(params, registry, policies, "output_classifier"),
            threshold=_threshold(params, "output_threshold", name),
            leak_check=_leak_check(params, name),
        ),
        input_threshold=_threshold(params, "input_threshold", name),
    )
