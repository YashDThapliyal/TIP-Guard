"""Construct a guardrail from a DefenseConfig."""

from tipguard.config.loader import ConfigError
from tipguard.config.schemas import DefenseConfig, PoliciesConfig
from tipguard.guardrail import baselines, tip_guard_factory
from tipguard.guardrail.no_defense import NoDefense
from tipguard.guardrail.tip_guard_factory import build_tip_guard
from tipguard.guardrail.types import (
    Guardrail,
    SystemPromptCondition,
    build_system_prompt,
)
from tipguard.models.registry import ProviderRegistry


def resolve_params(defense: DefenseConfig) -> dict[str, object]:
    """The parameters a defence will actually run with, defaults filled in.

    A `DefenseConfig` records only what its YAML wrote, so a run's manifest
    built from it cannot say what an omitted parameter became. That is not
    hypothetical: every study arm omitted `input_threshold`, so the threshold
    they all ran at was never stored, and recovering it later meant inferring
    it from the recorded decisions. Recording this alongside the config closes
    that gap for future runs.

    Unknown defence names return the config's own params rather than raising,
    because this is called to describe a run, not to validate one --
    `build_guardrail` owns the validation and will reject the name anyway.
    """
    if defense.name == "tip_guard":
        accepted, defaults = tip_guard_factory.KNOWN_PARAMS, tip_guard_factory.DEFAULTS
    elif defense.name in baselines.PARAMS_BY_DEFENSE:
        accepted, defaults = baselines.PARAMS_BY_DEFENSE[defense.name], baselines.DEFAULTS
    elif defense.name == "no_defense":
        # Reads nothing, so nothing it was handed took effect.
        return {}
    else:
        return dict(defense.params)
    # Restricted to what this defence reads. Reporting the whole family's
    # parameters would credit `keyword_filter` with a `classifier_model` it
    # never consults.
    resolved = {**defaults, **defense.params}
    return {key: value for key, value in resolved.items() if key in accepted}


def build_guardrail(
    defense: DefenseConfig,
    registry: ProviderRegistry,
    main_model: str,
    policies: PoliciesConfig,
    system_prompt_condition: SystemPromptCondition = "forbidden",
) -> Guardrail:
    """Build the configured guardrail.

    Takes the registry and the main model's *alias* rather than a built
    provider so that multi-component defenses can resolve their own extra
    models (judge, canonicaliser) from the same registry and share its cache.
    """
    system_prompt = build_system_prompt(policies, system_prompt_condition)
    params = defense.params
    if defense.name == "no_defense":
        # Validated even though there is nothing to configure: silently
        # accepting a parameter this arm cannot act on would let a config
        # claim a setting the control never had.
        if params:
            raise ConfigError(f"defense 'no_defense' takes no params, got {sorted(params)}")
        return NoDefense(registry.get(main_model), system_prompt)
    if defense.name == "keyword_filter":
        return baselines.keyword_filter(params, registry, main_model, system_prompt)
    if defense.name == "pattern_detector":
        return baselines.pattern_detector(params, registry, main_model, system_prompt)
    if defense.name == "input_classifier":
        return baselines.input_classifier(params, registry, main_model, system_prompt, policies)
    if defense.name == "output_classifier":
        return baselines.output_classifier(params, registry, main_model, system_prompt, policies)
    if defense.name == "input_output_classifier":
        return baselines.input_output_classifier(
            params, registry, main_model, system_prompt, policies
        )
    if defense.name == "tip_guard":
        return build_tip_guard(params, registry, main_model, system_prompt, policies)
    raise ConfigError(f"unknown defense {defense.name!r}")
