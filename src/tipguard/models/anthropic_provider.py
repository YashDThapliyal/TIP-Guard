"""Anthropic Messages API provider."""

import time
from typing import Any

from tipguard.config.schemas import ModelSpec
from tipguard.models.pricing import estimate_cost
from tipguard.models.types import ModelRequest, ModelResponse, ProviderError

JSON_INSTRUCTION = "\nRespond with a single JSON object and nothing else."


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, spec: ModelSpec, client: Any | None = None) -> None:
        self.spec = spec
        self.model = spec.model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic()
        return self._client

    def complete(self, request: ModelRequest) -> ModelResponse:
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        if request.response_format == "json":
            system = system + JSON_INSTRUCTION
        messages = [
            {"role": m.role, "content": m.content} for m in request.messages if m.role != "system"
        ]
        started = time.perf_counter()
        try:
            result = self.client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except Exception as exc:
            raise ProviderError(f"{self.name}/{self.model}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        text = "".join(
            block.text for block in result.content if getattr(block, "type", "") == "text"
        )
        input_tokens = int(result.usage.input_tokens)
        output_tokens = int(result.usage.output_tokens)
        return ModelResponse(
            text=text,
            model=self.model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=estimate_cost(self.spec, input_tokens, output_tokens),
        )
