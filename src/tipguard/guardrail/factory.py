"""Construct a guardrail from a DefenseConfig."""

from tipguard.config.loader import ConfigError
from tipguard.config.schemas import DefenseConfig, PoliciesConfig
from tipguard.guardrail.no_defense import NoDefense
from tipguard.guardrail.types import Guardrail, build_system_prompt
from tipguard.models.registry import ProviderRegistry


def build_guardrail(
    defense: DefenseConfig,
    registry: ProviderRegistry,
    main_model: str,
    policies: PoliciesConfig,
) -> Guardrail:
    """Build the configured guardrail.

    Takes the registry and the main model's *alias* rather than a built
    provider so that multi-component defenses can resolve their own extra
    models (judge, canonicaliser) from the same registry and share its cache.
    """
    system_prompt = build_system_prompt(policies)
    if defense.name == "no_defense":
        return NoDefense(registry.get(main_model), system_prompt)
    raise ConfigError(f"unknown defense {defense.name!r}")
