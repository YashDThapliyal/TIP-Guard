from pathlib import Path

import pytest

from tipguard.benchmark.schema import Decision
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import DefenseConfig, PoliciesConfig
from tipguard.guardrail.factory import build_guardrail
from tipguard.guardrail.no_defense import NoDefense
from tipguard.guardrail.types import build_system_prompt
from tipguard.models.mock import MockProvider


def test_no_defense_allows_and_returns_model_text(repo_root: Path) -> None:
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    guard = NoDefense(MockProvider(), build_system_prompt(policies))
    result = guard.run("hello")
    assert result.decision is Decision.ALLOW
    assert result.response_text == "ECHO: hello"
    assert result.model_calls == 1
    assert [c.component for c in result.components] == ["main_model"]


def test_system_prompt_lists_every_policy(repo_root: Path) -> None:
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    text = build_system_prompt(policies)
    for policy in policies.policies:
        assert policy.protected_values[0] in text


def test_factory_unknown_defense(repo_root: Path) -> None:
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    with pytest.raises(ConfigError):
        build_guardrail(DefenseConfig(name="nope"), MockProvider(), policies)


def test_factory_builds_no_defense(repo_root: Path) -> None:
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    assert (
        build_guardrail(DefenseConfig(name="no_defense"), MockProvider(), policies).name
        == "no_defense"
    )
