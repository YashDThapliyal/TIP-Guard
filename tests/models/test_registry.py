from pathlib import Path

import pytest

from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import ModelsConfig
from tipguard.models.cache import ResponseCache
from tipguard.models.registry import ProviderRegistry


def test_registry_builds_mock_and_memoises(repo_root: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "models.yaml", ModelsConfig)
    registry = ProviderRegistry(cfg)
    assert registry.get("mock-main") is registry.get("mock-main")
    assert registry.get("mock-main").name == "mock"


def test_registry_unknown_alias(repo_root: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "models.yaml", ModelsConfig)
    with pytest.raises(KeyError):
        ProviderRegistry(cfg).get("nope")


def test_registry_wraps_with_cache(repo_root: Path, tmp_path: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "models.yaml", ModelsConfig)
    registry = ProviderRegistry(cfg, cache=ResponseCache(tmp_path / "c.sqlite"))
    assert type(registry.get("mock-main")).__name__ == "CachedProvider"


def test_registry_builds_cloud_providers_without_network(repo_root: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "models.yaml", ModelsConfig)
    registry = ProviderRegistry(cfg)
    assert registry.get("gpt-4o-mini").name == "openai"
    assert registry.get("claude-haiku-4-5").name == "anthropic"
    assert registry.get("ollama-qwen2.5-7b").name == "ollama"
