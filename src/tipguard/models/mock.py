"""Deterministic scripted provider for tests and offline smoke runs."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from tipguard.models.types import ModelRequest, ModelResponse


@dataclass(frozen=True)
class MockRule:
    """A pattern to match against the last user message, and its reply.

    The pattern is compiled once, at construction, so an invalid regex is a
    `re.error` where the rule is written rather than on whichever prompt
    first reaches the provider — this project's tests and offline runs are
    mock-heavy, and re-compiling per prompt also re-paid the cost on every
    call. `regex` is excluded from equality and repr so two rules still
    compare by what they say.
    """

    pattern: str
    response: str
    regex: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regex", re.compile(self.pattern, re.IGNORECASE))


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
            if rule.regex.search(user_text):
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
