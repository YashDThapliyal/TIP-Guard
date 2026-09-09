"""Build providers from the models config, memoised by alias."""

import json
from typing import assert_never

from tipguard.config.loader import ConfigError
from tipguard.config.schemas import ModelsConfig, ModelSpec
from tipguard.models.anthropic_provider import AnthropicProvider
from tipguard.models.cache import CachedProvider, ResponseCache
from tipguard.models.mock import JUDGE_DEFAULT, JUDGE_RULES, MockProvider
from tipguard.models.ollama_provider import OllamaProvider
from tipguard.models.openai_provider import OpenAIProvider
from tipguard.models.types import ModelProvider


def build_provider(spec: ModelSpec) -> ModelProvider:
    """The provider named by `spec`, with no default.

    Every branch is explicit and the tail is `assert_never`, so adding a name
    to `ProviderName` without a branch here is a mypy error rather than a
    silent fall-through. It used to fall through to Ollama, which pointed an
    unrecognised name at localhost instead of failing.
    """
    if spec.provider == "mock":
        if spec.model == "mock-judge":
            # The judge alias answers in the JSON a risk classifier expects.
            # A bare mock echoes its prompt, which is a parser failure, and a
            # guard built on that blocks every case including all benign
            # traffic -- a degenerate arm rather than a cheap one.
            return MockProvider(model=spec.model, rules=JUDGE_RULES, default=JUDGE_DEFAULT)
        return MockProvider(model=spec.model)
    if spec.provider == "openai":
        return OpenAIProvider(spec)
    if spec.provider == "anthropic":
        return AnthropicProvider(spec)
    if spec.provider == "ollama":
        return OllamaProvider(spec)
    assert_never(spec.provider)


def _cache_namespace(spec: ModelSpec) -> str:
    """Namespace a cached response by every request-shaping field a provider
    resolves from the spec, so specs that differ only in temperature or
    max_tokens never share a cached response (pricing is not part of the
    request, so it is excluded).
    """
    identity = {
        "base_url": spec.base_url,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }
    return json.dumps(identity, sort_keys=True)


class ProviderRegistry:
    def __init__(self, models_config: ModelsConfig, cache: ResponseCache | None = None) -> None:
        self._config = models_config
        self._cache = cache
        self._providers: dict[str, ModelProvider] = {}

    def spec(self, alias: str) -> ModelSpec:
        try:
            return self._config.models[alias]
        except KeyError as exc:
            raise ConfigError(
                f"unknown model alias {alias!r}; known aliases: {sorted(self._config.models)}"
            ) from exc

    def get(self, alias: str) -> ModelProvider:
        if alias not in self._providers:
            spec = self.spec(alias)
            provider = build_provider(spec)
            if self._cache is not None:
                provider = CachedProvider(
                    provider, self._cache, namespace=_cache_namespace(spec), spec=spec
                )
            self._providers[alias] = provider
        return self._providers[alias]
