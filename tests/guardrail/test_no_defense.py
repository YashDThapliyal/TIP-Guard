from pathlib import Path

import pytest

from tipguard.benchmark.schema import Decision
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import DefenseConfig, ModelsConfig, ModelSpec, PoliciesConfig
from tipguard.guardrail.factory import build_guardrail
from tipguard.guardrail.no_defense import NoDefense
from tipguard.guardrail.types import build_system_prompt
from tipguard.models.cache import ResponseCache
from tipguard.models.mock import MockProvider
from tipguard.models.registry import ProviderRegistry


def _registry() -> ProviderRegistry:
    models = ModelsConfig(models={"m": ModelSpec(provider="mock", model="mock-echo")})
    return ProviderRegistry(models)


def test_no_defense_allows_and_returns_model_text(repo_root: Path) -> None:
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    guard = NoDefense(MockProvider(), build_system_prompt(policies))
    result = guard.run("hello")
    assert result.decision is Decision.ALLOW
    assert result.response_text == "ECHO: hello"
    assert result.model_calls == 1
    assert result.cached_model_calls == 0
    assert [c.component for c in result.components] == ["main_model"]


def test_no_defense_counts_a_cached_model_call(repo_root: Path, tmp_path: Path) -> None:
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    registry = ProviderRegistry(
        ModelsConfig(models={"m": ModelSpec(provider="mock", model="mock-echo")}),
        cache=ResponseCache(tmp_path / "c.sqlite"),
    )
    guard = NoDefense(registry.get("m"), build_system_prompt(policies))
    assert guard.run("hello").cached_model_calls == 0
    assert guard.run("hello").cached_model_calls == 1


def test_system_prompt_lists_every_policy(repo_root: Path) -> None:
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    text = build_system_prompt(policies)
    for policy in policies.policies:
        assert policy.protected_values[0] in text


def test_factory_unknown_defense(repo_root: Path) -> None:
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    with pytest.raises(ConfigError, match="unknown defense"):
        build_guardrail(DefenseConfig(name="nope"), _registry(), "m", policies)


def test_factory_builds_no_defense(repo_root: Path) -> None:
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    guard = build_guardrail(DefenseConfig(name="no_defense"), _registry(), "m", policies)
    assert guard.name == "no_defense"
    assert guard.run("hello").response_text == "ECHO: hello"


def test_factory_reports_unknown_main_model_alias(repo_root: Path) -> None:
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    with pytest.raises(ConfigError, match="unknown model alias"):
        build_guardrail(DefenseConfig(name="no_defense"), _registry(), "nope", policies)
