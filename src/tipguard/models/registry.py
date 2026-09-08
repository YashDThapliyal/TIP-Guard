"""Build providers from the models config, memoised by alias."""

from tipguard.config.schemas import ModelsConfig, ModelSpec
from tipguard.models.anthropic_provider import AnthropicProvider
from tipguard.models.cache import CachedProvider, ResponseCache
from tipguard.models.mock import MockProvider
from tipguard.models.ollama_provider import OllamaProvider
from tipguard.models.openai_provider import OpenAIProvider
from tipguard.models.types import ModelProvider


def build_provider(spec: ModelSpec) -> ModelProvider:
    if spec.provider == "mock":
        return MockProvider(model=spec.model)
    if spec.provider == "openai":
        return OpenAIProvider(spec)
    if spec.provider == "anthropic":
        return AnthropicProvider(spec)
    return OllamaProvider(spec)


class ProviderRegistry:
    def __init__(self, models_config: ModelsConfig, cache: ResponseCache | None = None) -> None:
        self._config = models_config
        self._cache = cache
        self._providers: dict[str, ModelProvider] = {}

    def spec(self, alias: str) -> ModelSpec:
        return self._config.models[alias]

    def get(self, alias: str) -> ModelProvider:
        if alias not in self._providers:
            provider = build_provider(self.spec(alias))
            if self._cache is not None:
                provider = CachedProvider(provider, self._cache)
            self._providers[alias] = provider
        return self._providers[alias]
