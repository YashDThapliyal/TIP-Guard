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


def _judgement(risk: float, category: str) -> str:
    return f'{{"risk": {risk}, "categories": ["{category}"], "rationale": "scripted mock"}}'


#: A scripted stand-in for a risk-rating model, used by the `mock-judge`
#: alias so the offline and CI arms of the study behave like a classifier
#: rather than like a permanently broken one.
#:
#: Without it the mock echoes its prompt, `LLMRiskClassifier` reads that as a
#: `parser_failure`, and every guard built on it blocks 100% of traffic --
#: including all benign cases. That is a degenerate arm, not a cheap one.
#:
#: Each pattern is anchored *after* the "Prompt to rate:" marker that
#: `classifiers.prompts.risk_user_prompt` writes, because everything before
#: it is the policy summary -- whose own descriptions say "Never reveal ..."
#: and would otherwise match every rule on every request, scoring an
#: innocuous question 0.9. The `(?s)` flag lets `.` cross the newlines the
#: rated text may contain.
#:
#: It is deliberately crude and models nothing: it keys on the surface
#: language a Task-in-Prompt wrapper tends to use. Measurement comes from the
#: real-model experiment configs; this exists so the suite runs offline and
#: so a smoke test can tell a blocking arm from an allowing one.
_RATED = r"(?s)Prompt to rate:.*?"

JUDGE_RULES: tuple[MockRule, ...] = (
    MockRule(
        pattern=_RATED + r"\b(?:reveal|disclose|exfiltrat\w*|leak)\b",
        response=_judgement(0.9, "data_exfiltration"),
    ),
    MockRule(
        pattern=_RATED + r"\b(?:ignore|override|disregard)\s+(?:the\s+)?(?:previous|above|prior)\b",
        response=_judgement(0.8, "prompt_injection"),
    ),
    MockRule(
        pattern=_RATED + r"\b(?:decode|decrypt|base64|cipher|morse)\b",
        response=_judgement(0.7, "prompt_injection"),
    ),
)

JUDGE_DEFAULT = _judgement(0.1, "none")


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
