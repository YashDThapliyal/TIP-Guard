"""Deterministic scripted provider for tests and offline smoke runs."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from tipguard.models.types import ModelRequest, ModelResponse


@dataclass(frozen=True)
class MockRule:
    pattern: str
    response: str


def _token_count(text: str) -> int:
    return len(text.split())


class MockProvider:
    name = "mock"

    def __init__(
        self,
        model: str = "mock-echo",
        rules: Sequence[MockRule] = (),
        default: str | None = None,
    ) -> None:
        self.model = model
        self._rules = tuple(rules)
        self._default = default

    def _respond_to(self, user_text: str) -> str:
        for rule in self._rules:
            if re.search(rule.pattern, user_text, flags=re.IGNORECASE):
                return rule.response
        return self._default if self._default is not None else f"ECHO: {user_text}"

    def complete(self, request: ModelRequest) -> ModelResponse:
        prompt_text = " ".join(m.content for m in request.messages)
        text = self._respond_to(request.last_user_content())
        return ModelResponse(
            text=text,
            model=self.model,
            provider=self.name,
            input_tokens=_token_count(prompt_text),
            output_tokens=_token_count(text),
            latency_ms=0.0,
        )
