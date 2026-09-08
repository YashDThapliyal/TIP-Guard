"""Local Ollama provider via its OpenAI-compatible endpoint."""

from typing import Any

from tipguard.models.openai_provider import OpenAIProvider

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider(OpenAIProvider):
    name = "ollama"

    def _build_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(base_url=self.spec.base_url or DEFAULT_OLLAMA_BASE_URL, api_key="ollama")
