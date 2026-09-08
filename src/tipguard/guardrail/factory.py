"""Construct a guardrail from a DefenseConfig."""

from tipguard.config.loader import ConfigError
from tipguard.config.schemas import DefenseConfig, PoliciesConfig
from tipguard.guardrail.no_defense import NoDefense
from tipguard.guardrail.types import Guardrail, build_system_prompt
from tipguard.models.types import ModelProvider


def build_guardrail(
    defense: DefenseConfig, main_model: ModelProvider, policies: PoliciesConfig
) -> Guardrail:
    system_prompt = build_system_prompt(policies)
    if defense.name == "no_defense":
        return NoDefense(main_model, system_prompt)
    raise ConfigError(f"unknown defense {defense.name!r}")
