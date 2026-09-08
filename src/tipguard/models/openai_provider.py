"""OpenAI-compatible chat completions provider."""

import time
from typing import Any

from tipguard.config.schemas import ModelSpec
from tipguard.models.pricing import estimate_cost
from tipguard.models.types import JSON_INSTRUCTION, ModelRequest, ModelResponse, ProviderError

ChatMessage = dict[str, str]


def _with_json_instruction(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Append JSON_INSTRUCTION to the system message, or the last user message if none."""
    target_index = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
    if target_index is None:
        target_index = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), None
        )
    if target_index is None:
        return messages
    updated = messages[target_index]
    new_messages = list(messages)
    new_messages[target_index] = {**updated, "content": updated["content"] + JSON_INSTRUCTION}
    return new_messages


class OpenAIProvider:
    name = "openai"

    def __init__(self, spec: ModelSpec, client: Any | None = None) -> None:
        self.spec = spec
        self.model = spec.model
        self._client = client

    def _build_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(base_url=self.spec.base_url)

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def complete(self, request: ModelRequest) -> ModelResponse:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        is_json = request.response_format == "json"
        if is_json:
            messages = _with_json_instruction(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if is_json:
            kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(f"{self.name}/{self.model}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        usage = completion.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return ModelResponse(
            text=completion.choices[0].message.content or "",
            model=self.model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=estimate_cost(self.spec, input_tokens, output_tokens),
        )
