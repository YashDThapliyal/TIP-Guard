"""Anthropic Messages API provider.

Note: the locked `anthropic` SDK's `Messages.create()` has no `temperature`
parameter (sampling controls were removed from the API for current Claude
models), so `request.temperature` is intentionally ignored here.
"""

import time
from typing import Any

from tipguard.config.schemas import ModelSpec
from tipguard.models.pricing import estimate_cost
from tipguard.models.types import JSON_INSTRUCTION, ModelRequest, ModelResponse, ProviderError

ChatMessage = dict[str, str]


def _append_to_last_user(messages: list[ChatMessage]) -> list[ChatMessage]:
    target_index = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), None
    )
    if target_index is None:
        return messages
    updated = messages[target_index]
    new_messages = list(messages)
    new_messages[target_index] = {**updated, "content": updated["content"] + JSON_INSTRUCTION}
    return new_messages


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
        has_system = any(m.role == "system" for m in request.messages)
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        messages = [
            {"role": m.role, "content": m.content} for m in request.messages if m.role != "system"
        ]
        if request.response_format == "json":
            if has_system:
                system = system + JSON_INSTRUCTION
            else:
                messages = _append_to_last_user(messages)
        max_tokens = request.max_tokens if request.max_tokens is not None else self.spec.max_tokens
        started = time.perf_counter()
        try:
            result = self.client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
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
